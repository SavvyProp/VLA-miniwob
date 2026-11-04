"""Real Chrome integration. DOM inspection is restricted to test/oracle fixtures."""

import pytest

from scriptwob.contracts import ClickAction
from scriptwob.environment import HEIGHT, WIDTH, BrowserEnvironment
from scriptwob.generation import FAMILIES, generate_episode
from scriptwob.schema import (
    Episode,
    Item,
    Predicate,
    SemanticAction,
    Statement,
    Widget,
    World,
)

pytestmark = pytest.mark.browser


def action(op, target):
    return Statement(kind="action", action=SemanticAction(op=op, target=target))


def episode(program, *, widgets=None, layout_seed=3, max_steps=128):
    return Episode(
        id="browser-fixture",
        split="test",
        level=4,
        family="browser-fixture",
        seed=7,
        layout_seed=layout_seed,
        program=program,
        world=World(widgets=widgets or [
            Widget(id="alpha", label="Alpha", kind="button"),
            Widget(id="beta", label="Beta", kind="button"),
            Widget(id="counter", label="Count", kind="counter", value=0),
            Widget(id="toggle", label="Enabled", kind="toggle", value=False),
            Widget(id="status", label="Signal", kind="status", value="ready"),
            Widget(id="queue", label="Tickets", kind="queue", items=[
                Item(id=f"item{i}", label=f"Item {i + 1}", category="red" if i % 2 else "blue")
                for i in range(5)
            ]),
        ]),
        instruction="Follow the fixture procedure, then click Done.",
        structural_id="browser-fixture",
        max_steps=max_steps,
    )


def control(env, key):
    coords = env.native.instance.driver.execute_script(
        """
        const e = [...document.querySelectorAll('[data-action]')]
            .find(e => e.dataset.action === arguments[0]);
        const r = e.getBoundingClientRect();
        return [r.x + r.width / 2, r.y + r.height / 2];
        """,
        key,
    )
    return ClickAction(x=coords[0] / (WIDTH - 1), y=coords[1] / (HEIGHT - 1))


def test_every_generated_family_runs_in_real_browser():
    with BrowserEnvironment() as env:
        for level, families in FAMILIES.items():
            for family in families:
                task = generate_episode(43, level, "test", family)
                env.reset(task)
                while not env.terminated:
                    env.step(env.oracle_action())
                result = env.private_result()
                assert result["success"], (family, result)
                assert env.steps == task.complexity.reference_actions


def test_all_widgets_and_control_flow_execute_through_real_clicks():
    program = Statement(kind="sequence", steps=[
        action("click", "alpha"),
        Statement(
            kind="if", condition=Predicate(target="status", value="ready"),
            body=action("toggle", "toggle"), otherwise=action("click", "beta"),
        ),
        Statement(kind="repeat", count=2, body=action("increment", "counter")),
        Statement(
            kind="while", condition=Predicate(target="counter", operator="gt", value=0),
            body=action("decrement", "counter"),
        ),
        Statement(
            kind="until", condition=Predicate(target="counter", value=1),
            body=action("increment", "counter"),
        ),
        Statement(kind="foreach", collection="queue", body=Statement(
            kind="if", condition=Predicate(target="queue", field="category", value="blue"),
            body=action("approve", "queue"), otherwise=action("reject", "queue"),
        )),
    ])
    with BrowserEnvironment() as env:
        initial = env.reset(episode(program))
        assert initial.image.size == (WIDTH, HEIGHT)
        assert not initial.history
        while not env.terminated:
            env.step(env.oracle_action())
        result = env.private_result()
        assert result["success"], result
        assert env.steps == 18
        assert all(entry.feedback == "clicked" for entry in env.history)
        assert [item.result for item in env.engine.world.widget("queue").items] == [
            "approved", "rejected", "approved", "rejected", "approved"
        ]
        # Pixel-normalized recorded actions replay without another rounding drift.
        recorded = [entry.action for entry in env.history]
        env.reset(episode(program))
        for click in recorded:
            env.step(click)
        assert env.private_result()["success"]


def test_repeated_reset_is_deterministic_and_program_state_is_not_in_page():
    task = episode(action("click", "alpha"))
    with BrowserEnvironment() as env:
        first = env.reset(task)
        env.step(env.oracle_action())
        second = env.reset(task)
        assert first.image.tobytes() == second.image.tobytes()
        assert first.history == second.history == ()
        config = env.native.instance.driver.execute_script("return window.scriptwobConfig")
        assert set(config) == {"world", "layout"}
        public_widgets = {widget["id"]: widget for widget in config["world"]["widgets"]}
        assert public_widgets["alpha"] == task.world.widget("alpha").model_dump()
        current_item = task.world.widget("queue").items[0].model_dump()
        assert public_widgets["queue"]["items"][0] == current_item
        assert public_widgets["queue"]["items"][1]["category"] == "Hidden"
        assert "program" not in config and "expected_action" not in config
        assert len(config["layout"]) == 6
        changed = env.reset(task.model_copy(update={"layout_seed": 11}))
        assert changed.image.tobytes() != first.image.tobytes()


def test_queue_future_contents_are_hidden_until_next_in_payload_and_screenshot():
    program = Statement(
        kind="until", condition=Predicate(target="queue", field="category", value="red"),
        body=action("advance", "queue"),
    )
    original = episode(program, widgets=[Widget(
        id="queue", label="Tickets", kind="queue", items=[
            Item(id="first", label="First", category="blue"),
            Item(id="future-secret-a", label="Secret A", category="red", status="closed"),
            Item(id="last", label="Last", category="red"),
        ],
    )])
    alternative = original.model_copy(deep=True)
    future = alternative.world.widget("queue").items[1]
    future.id, future.label, future.category, future.status = (
        "future-secret-b", "Secret B", "blue", "open"
    )
    with BrowserEnvironment() as env:
        before_a = env.reset(original)
        config = env.native.instance.driver.execute_script("return window.scriptwobConfig")
        assert "Secret A" not in str(config) and "future-secret-a" not in str(config)
        assert config["world"]["widgets"][0]["items"][1]["status"] == "Hidden"
        after_a = env.step(env.oracle_action())
        visible_a = env.native.instance.driver.execute_script(
            "return document.querySelector('.queue').innerText"
        )
        assert "First" in visible_a and "Secret A" in visible_a
        assert "Last" not in visible_a
        before_b = env.reset(alternative)
        after_b = env.step(env.oracle_action())
        # Future content cannot disclose a loop's stopping position upfront.
        assert before_a.image.tobytes() == before_b.image.tobytes()
        assert after_a.image.tobytes() != after_b.image.tobytes()
        assert before_a.instruction == before_b.instruction
        assert env.engine.world.widget("queue").items[2].category == "red"


@pytest.mark.parametrize("bad_control,error", [
    ("beta:click", "wrong_action"),
    ("done:done", "premature_done"),
    (None, "invalid_action"),
])
def test_wrong_branch_premature_done_and_background_fail(bad_control, error):
    program = Statement(
        kind="if", condition=Predicate(target="status", value="ready"),
        body=action("click", "alpha"), otherwise=action("click", "beta"),
    )
    with BrowserEnvironment() as env:
        env.reset(episode(program))
        next_action = control(env, bad_control) if bad_control else ClickAction(x=0, y=0)
        observation = env.step(next_action)
        result = env.private_result()
        assert env.terminated and not result["success"]
        assert result["error"].startswith(error)
        assert observation.history[-1].feedback == "clicked"
        with pytest.raises(RuntimeError, match="termination"):
            env.step(next_action)


def test_done_is_required_and_extra_iteration_fails():
    with BrowserEnvironment() as env:
        env.reset(episode(Statement(kind="repeat", count=1, body=action("click", "alpha"))))
        env.step(env.oracle_action())
        assert not env.terminated
        assert not env.private_result()["success"]
        env.step(control(env, "alpha:click"))
        assert env.private_result()["error"] == "extra_action"


def test_zero_iteration_loop_requires_only_done():
    program = Statement(
        kind="until", condition=Predicate(target="counter", value=0),
        body=action("increment", "counter"),
    )
    with BrowserEnvironment() as env:
        env.reset(episode(program))
        env.step(env.oracle_action())
        assert env.steps == 1
        assert env.private_result()["success"]
