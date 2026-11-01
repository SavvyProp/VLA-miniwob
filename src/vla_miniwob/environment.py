"""Native MiniWoB++ Gymnasium/Selenium execution with private order validation."""

import functools
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

from .contracts import HistoryEntry, MouseAction, PolicyObservation, decode
from .tasks import Instance, Manifest
from .validation import Event, OrderValidator

WIDTH, HEIGHT = 720, 540


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class _NativeEnvironment(MiniWoBEnvironment):
    def _hard_reset_instance(self):
        super()._hard_reset_instance()
        self.instance.task_width, self.instance.task_height = WIDTH, HEIGHT
        self.instance.driver.set_window_size(WIDTH + 60, HEIGHT + 140)
        self.instance.driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": WIDTH,
                "height": HEIGHT,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        self.instance.inner_width, self.instance.inner_height = self.instance.driver.execute_script(
            "return [window.innerWidth, window.innerHeight]"
        )
        if self.instance.inner_width < WIDTH or self.instance.inner_height < HEIGHT:
            raise RuntimeError("Browser viewport is smaller than the benchmark screenshot")


class BrowserEnvironment:
    """Lifecycle for one isolated browser. Reuse between episodes via reset()."""

    def __init__(self, manifest: Manifest, *, headless: bool = True):
        self.manifest = manifest
        self._temp = tempfile.TemporaryDirectory(prefix="vla-miniwob-")
        root = Path(self._temp.name)
        (root / "upstream").symlink_to(HTML_DIR, target_is_directory=True)
        (root / "benchmark").mkdir()
        (root / "benchmark" / "composed.html").write_text(
            files("vla_miniwob").joinpath("assets/composed.html").read_text()
        )
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root))
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.native = None
        try:
            self.native = _NativeEnvironment(
                subdomain="composed",
                render_mode=None if headless else "human",
                base_url=f"http://127.0.0.1:{self._server.server_port}/benchmark/",
                field_extractor=lambda _: [],
                block_on_reset=False,
                wait_ms=manifest.settle_ms,
                action_space_config=ActionSpaceConfig(
                    action_types=[ActionTypes.MOVE_COORDS, ActionTypes.CLICK_COORDS]
                ),
            )
        except Exception:
            self.close()
            raise

    def reset(self, instance: Instance) -> PolicyObservation:
        self.instance = instance
        self.validator = OrderValidator(instance.required_order)
        self.history: list[HistoryEntry] = []
        self.steps = 0
        self.native_reward = 0.0
        self.terminated = False
        self._event_offset = 0
        self._state = (0.0, 0.0, 0.0)
        driver = self.native.instance.driver
        driver.execute_script("window.benchmarkConfig = arguments[0]", instance.browser_config())
        self.native.reset(seed=instance.values_seed)
        WebDriverWait(driver, 15).until(lambda d: d.execute_script("return WOB_TASK_READY"))
        driver.execute_script("clearTimeout(core.EP_TIMER); clearInterval(core.CD_TIMER);")
        # Retain a non-null timer sentinel: native endEpisode requires it to publish reward.
        driver.execute_script("core.EP_TIMER = -1;")
        self.instruction = driver.execute_script("return core.getUtterance()")
        # Reset the physical pointer as well as the recorded observable state.
        # Setup movement is outside the episode's policy-action budget.
        self.native.instance.perform(
            self.native.create_action(ActionTypes.MOVE_COORDS, coords=[0, 0]),
            self.native.action_space_config,
        )
        return self.observe()

    def observe(self) -> PolicyObservation:
        inst = self.native.instance
        # Native Gym returns a blank image after terminal events. Capture the real screen
        # here for the settled transition record; it never conditions the preceding action.
        image = get_screenshot(inst.driver, inst.inner_width, inst.inner_height, WIDTH, HEIGHT)
        return PolicyObservation(self.instruction, image, self._state, tuple(self.history))

    def step(self, action: MouseAction):
        if self.terminated or self.steps >= self.manifest.max_steps:
            raise RuntimeError("Cannot act after termination or primitive budget")
        executed = decode(action, WIDTH, HEIGHT)
        self.steps += 1
        self.native.instance.driver.execute_script(
            "window.benchmarkStep = arguments[0]", self.steps
        )
        native_action = self.native.create_action(
            ActionTypes.CLICK_COORDS if executed.kind == "click" else ActionTypes.MOVE_COORDS,
            coords=[executed.pixel_x, executed.pixel_y],
        )
        _, self.native_reward, self.terminated, _, _ = self.native.step(native_action)
        self._state = (executed.action.x, executed.action.y, 0.0)
        # Feedback describes execution only; no reward/target/order information.
        feedback = "executed; coordinates clipped" if executed.clipped else "executed"
        self.history.append(
            HistoryEntry(step=self.steps, action=executed.action, feedback=feedback)
        )
        events = self.native.instance.driver.execute_script("return window.benchmarkLedger")
        for event in events[self._event_offset :]:
            self.validator.ingest(Event.model_validate(event))
        self._event_offset = len(events)
        return self.observe(), executed, feedback

    def private_result(self, reason: str = "") -> dict:
        return {
            "required_order": self.instance.required_order,
            "events": [event.model_dump() for event in self.validator.events],
            "native_reward": self.native_reward,
            "native_component_rewards": self.native.instance.driver.execute_script(
                "return window.benchmarkRewards"
            ),
            "verdict": self.validator.verdict(
                self.steps, self.manifest.max_steps, self.terminated, reason
            ).model_dump(),
        }

    @property
    def browser_version(self) -> str:
        return self.native.instance.driver.capabilities["browserVersion"]

    def close(self):
        if self.native is not None:
            self.native.close()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
