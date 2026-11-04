"""Screenshot-only scriptWoB browser backed by MiniWoB++ and a private interpreter.

The page renders public world state and reports physical control clicks. Procedure
execution and validation live exclusively in Python; no AST, expected action, or
program counter is sent to JavaScript.
"""

import functools
import random
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from miniwob.action import ActionSpaceConfig, ActionTypes
from miniwob.environment import MiniWoBEnvironment
from miniwob.screenshot import get_screenshot
from miniwob.selenium_instance import HTML_DIR
from selenium.webdriver.support.ui import WebDriverWait

from .contracts import ClickAction, HistoryEntry, Observation
from .engine import Engine
from .schema import Episode, SemanticAction

WIDTH, HEIGHT = 1000, 760


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class _NativeEnvironment(MiniWoBEnvironment):
    def _hard_reset_instance(self):
        super()._hard_reset_instance()
        instance = self.instance
        instance.task_width, instance.task_height = WIDTH, HEIGHT
        instance.driver.set_window_size(WIDTH + 60, HEIGHT + 140)
        instance.driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": 1, "mobile": False},
        )
        instance.inner_width, instance.inner_height = instance.driver.execute_script(
            "return [window.innerWidth, window.innerHeight]"
        )
        if (instance.inner_width, instance.inner_height) != (WIDTH, HEIGHT):
            raise RuntimeError("Browser viewport does not match the screenshot action space")


class BrowserEnvironment:
    """Reuse one isolated Chrome session across deterministic script episodes.

    Only :class:`Observation` objects belong to the policy interface. ``engine``,
    ``oracle_action`` and ``private_result`` are privileged evaluation facilities.
    """

    def __init__(self, *, headless: bool = True):
        self.native = None
        self._server = None
        self._thread = None
        self._temp = tempfile.TemporaryDirectory(prefix="scriptwob-")
        self._closed = False
        self.episode = None
        self.steps = 0
        self.terminated = False
        self.history: list[HistoryEntry] = []
        self._events: list[dict] = []
        self._budget_error = None
        try:
            root = Path(self._temp.name)
            (root / "upstream").symlink_to(HTML_DIR, target_is_directory=True)
            (root / "tasks").mkdir()
            (root / "tasks" / "scriptwob.html").write_text(
                files("scriptwob").joinpath("assets/scriptwob.html").read_text(),
                encoding="utf-8",
            )
            self._server = ThreadingHTTPServer(
                ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root))
            )
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self.native = _NativeEnvironment(
                subdomain="scriptwob",
                render_mode=None if headless else "human",
                base_url=f"http://127.0.0.1:{self._server.server_port}/tasks/",
                field_extractor=lambda _: [],
                block_on_reset=False,
                wait_ms=0,
                action_space_config=ActionSpaceConfig(
                    action_types=[ActionTypes.MOVE_COORDS, ActionTypes.CLICK_COORDS]
                ),
            )
        except Exception:
            self.close()
            raise

    def reset(self, episode: Episode) -> Observation:
        if self._closed:
            raise RuntimeError("Environment has been closed")
        self.episode = episode
        self.engine = Engine(episode)
        self.steps = 0
        self.terminated = False
        self.history = []
        self._events = []
        self._budget_error = None
        layout = [widget.id for widget in self.engine.world.widgets]
        random.Random(episode.layout_seed).shuffle(layout)
        self._layout = layout
        driver = self.native.instance.driver
        driver.execute_script(
            "window.scriptwobConfig = arguments[0]",
            {"world": self._browser_world(), "layout": layout},
        )
        self.native.reset(seed=episode.seed)
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return WOB_TASK_READY === true")
        )
        # Wall time is irrelevant to procedural difficulty. Disable native timers
        # and keep a non-null sentinel for MiniWoB's end-of-episode protocol.
        driver.execute_script(
            "clearTimeout(core.EP_TIMER); clearInterval(core.CD_TIMER); core.EP_TIMER = -1;"
        )
        self.native.instance.perform(
            self.native.create_action(ActionTypes.MOVE_COORDS, coords=[0, 0]),
            self.native.action_space_config,
        )
        self._check_visible_controls()
        return self.observe()

    def _browser_world(self) -> dict:
        """Reveal queue records on arrival, keeping future contents out of the page.

        Already visited records remain visible. Queue length/cursor are public,
        while labels, predicates, and results of unopened records are not.
        """
        world = self.engine.world.public_dict()
        for widget in world["widgets"]:
            if widget["kind"] != "queue":
                continue
            for index in range(widget["cursor"] + 1, len(widget["items"])):
                widget["items"][index] = {
                    "id": f"unopened-{index}",
                    "label": "Unopened",
                    "category": "Hidden",
                    "status": "Hidden",
                    "processed": False,
                    "result": "",
                }
        return world

    def _check_visible_controls(self):
        """Reject unsupported page sizes instead of silently cropping task state."""
        overflow = self.native.instance.driver.execute_script(
            """
            return [...document.querySelectorAll('.card, [data-action]')].some(e => {
              const r = e.getBoundingClientRect();
              return r.left < 0 || r.top < 0 || r.right > innerWidth || r.bottom > innerHeight;
            }) || [...document.querySelectorAll('.card')].some(
              e => e.scrollHeight > e.clientHeight + 1 || e.scrollWidth > e.clientWidth + 1);
            """
        )
        if overflow:
            raise ValueError("Episode widgets exceed the visible scriptWoB viewport")

    def observe(self) -> Observation:
        if self.episode is None:
            raise RuntimeError("Call reset before observing")
        instance = self.native.instance
        screenshot = get_screenshot(
            instance.driver, instance.inner_width, instance.inner_height, WIDTH, HEIGHT
        )
        return Observation(self.episode.instruction, screenshot, tuple(self.history))

    def step(self, action: ClickAction) -> Observation:
        if self.episode is None:
            raise RuntimeError("Call reset before acting")
        if self.terminated:
            raise RuntimeError("Cannot act after episode termination")
        if not isinstance(action, ClickAction):
            action = ClickAction.model_validate(action)
        self.steps += 1
        driver = self.native.instance.driver
        driver.execute_script("window.scriptwobClicks = []")
        pixel_x, pixel_y = action.pixels(WIDTH, HEIGHT)
        self.native.instance.perform(
            self.native.create_action(
                ActionTypes.CLICK_COORDS, coords=[pixel_x, pixel_y]
            ),
            self.native.action_space_config,
        )
        clicks = driver.execute_script("return window.scriptwobClicks")
        if len(clicks) == 1:
            semantic = SemanticAction.model_validate(clicks[0])
            self.engine.step(semantic)
            self._events.append({"step": self.steps, "action": semantic.model_dump()})
        else:
            self.engine.invalid_action("Click did not hit an enabled control")
            self._events.append({"step": self.steps, "action": None})
        # Never put semantic correctness, reference steps, or private state into
        # the policy's feedback. The screenshot shows the physical consequences.
        executed = ClickAction(x=pixel_x / (WIDTH - 1), y=pixel_y / (HEIGHT - 1))
        self.history.append(HistoryEntry(step=self.steps, action=executed, feedback="clicked"))
        if self.steps >= self.episode.max_steps and not self.engine.done:
            self._budget_error = "Primitive action budget exhausted"
        self.terminated = self.engine.done or self._budget_error is not None
        driver.execute_script(
            "window.renderWorld(arguments[0], arguments[1])",
            self._browser_world(),
            self._layout,
        )
        if self.terminated:
            # Native metadata is private, while the final screenshot remains the
            # actual page state rather than MiniWoB's blank terminal observation.
            driver.execute_script(
                "WOB_DONE_GLOBAL = true; WOB_RAW_REWARD_GLOBAL = arguments[0];"
                "WOB_REWARD_GLOBAL = arguments[0];",
                int(self.engine.success and self._budget_error is None),
            )
        return self.observe()

    def oracle_action(self) -> ClickAction:
        """Privileged fixture: locate the interpreter's next action in the DOM.

        Frontier experts do not receive this method or its semantic target.
        """
        if self.episode is None or self.terminated:
            raise RuntimeError("Oracle requires an active episode")
        semantic = self.engine.expected_action()
        if semantic is None:
            raise RuntimeError("Reference interpreter has no available next action")
        key = f"{semantic.target}:{semantic.op}"
        coords = self.native.instance.driver.execute_script(
            """
            const e = [...document.querySelectorAll('[data-action]')]
                .find(e => e.dataset.action === arguments[0]);
            if (!e || e.disabled) throw Error('Oracle control missing or disabled');
            const r = e.getBoundingClientRect();
            const x = r.x + r.width / 2, y = r.y + r.height / 2;
            const hit = document.elementFromPoint(x, y);
            if (hit !== e && !e.contains(hit)) throw Error('Oracle control occluded');
            return [x, y];
            """,
            key,
        )
        return ClickAction(x=coords[0] / (WIDTH - 1), y=coords[1] / (HEIGHT - 1))

    def private_result(self, reason: str = "") -> dict:
        if self.episode is None:
            raise RuntimeError("No episode has been reset")
        return {
            "episode_id": self.episode.id,
            "success": bool(self.engine.success and self._budget_error is None and not reason),
            "terminated": self.terminated,
            "steps": self.steps,
            "error": reason or self._budget_error or self.engine.error,
            "events": list(self._events),
            "final_world": self.engine.world.public_dict(),
        }

    @property
    def browser_version(self) -> str:
        return self.native.instance.driver.capabilities["browserVersion"]

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.native is not None:
                self.native.close()
        finally:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=2)
            self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
