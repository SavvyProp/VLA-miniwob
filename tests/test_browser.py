"""DOM access below is a TEST ORACLE only, never a teacher or evaluation policy."""

import json
import re

import numpy as np
import pytest

from vla_miniwob.collection import run
from vla_miniwob.contracts import MouseAction
from vla_miniwob.dataset import ChunkDataset, export_hdf5
from vla_miniwob.environment import HEIGHT, WIDTH, BrowserEnvironment
from vla_miniwob.tasks import FAMILIES, make_manifest

pytestmark = pytest.mark.browser


def target_action(env, component, selector, text=None):
    coords = env.native.instance.driver.execute_script(
        """
      const frame = document.getElementById('component-' + arguments[0]);
      const elements = [...frame.contentDocument.querySelectorAll(arguments[1])];
      const e = elements.find(e => arguments[2] === null || e.textContent === arguments[2]);
      if (!e) throw Error('Test oracle target missing');
      const f = frame.getBoundingClientRect(), r = e.getBoundingClientRect();
      // Native sequence buttons can overlap. The test oracle must click an
      // actually exposed point, just as a screenshot policy would have to.
      for (const dy of [0.5, 0.1, 0.9]) for (const dx of [0.5, 0.1, 0.9]) {
        const x = r.x + r.width*dx, y = r.y + r.height*dy;
        const hit = frame.contentDocument.elementFromPoint(x, y);
        if (hit === e || e.contains(hit)) return [f.x+x, f.y+y];
      }
      throw Error('Test oracle target is occluded');
    """,
        component,
        selector,
        text,
    )
    return MouseAction(x=coords[0] / (WIDTH - 1), y=coords[1] / (HEIGHT - 1), click=1)


def click_element(env, component, selector, text=None):
    return env.step(target_action(env, component, selector, text))


def solve_component(env, component, task):
    query = env.native.instance.driver.execute_script(
        "const w=document.getElementById('component-'+arguments[0]).contentWindow;"
        "return w.core.getUtterance()",
        component,
    )
    if task == "click-button":
        click_element(env, component, "#area button", re.search(r'"(.*?)"', query)[1])
    elif task == "click-link":
        click_element(env, component, ".alink", re.search(r'"(.*?)"', query)[1])
    elif task == "click-dialog":
        click_element(env, component, "button.ui-button")
    elif task == "click-widget":
        widget = re.search(r'"(.*?)"', query)[1]
        click_element(env, component, f'[data-type="{widget}"]')
    elif task == "click-button-sequence":
        click_element(env, component, "#subbtn")
        click_element(env, component, "#subbtn2")
    elif task == "click-option":
        target = query.split("Select ")[1].split(" and click Submit")[0]
        click_element(env, component, "label", target)
        click_element(env, component, "#subbtn")
    elif task == "click-checkboxes":
        targets = query.split("Select ")[1].split(" and click Submit")[0].split(", ")
        for target in targets:
            if target != "nothing":
                click_element(env, component, "label", target)
        click_element(env, component, "#subbtn")
    else:
        raise AssertionError(task)


def test_real_browser_paired_orders_and_irreversible_violation():
    manifest = make_manifest(1, 0, 0, tasks=["click-button_click-link"])
    with BrowserEnvironment(manifest) as env:
        image = None
        for instance in manifest.instances:
            observation = env.reset(instance)
            if image is None:
                image = observation.image.tobytes()
            else:
                assert observation.image.tobytes() == image
            for component in instance.required_order:
                solve_component(env, component, instance.components[component])
            assert env.private_result()["verdict"]["success"]
            assert len(env.history) == 2
            assert env.history[0].action.click == env.history[1].action.click == 1
        env.reset(manifest.instances[0])
        solve_component(env, 1, "click-link")
        solve_component(env, 0, "click-button")
        result = env.private_result()
        assert result["native_reward"] == 1
        assert result["verdict"]["goals_valid"]
        assert not result["verdict"]["success"]
        assert result["verdict"]["first_violation"]["step"] == 1


def test_every_pilot_family_and_component_runs():
    manifest = make_manifest(1, 0, 0)
    with BrowserEnvironment(manifest) as env:
        for task in FAMILIES:
            instance = next(i for i in manifest.instances if i.task == task)
            observation = env.reset(instance)
            assert observation.image.size == (WIDTH, HEIGHT)
            for component in instance.required_order:
                solve_component(env, component, instance.components[component])
            assert env.private_result()["verdict"]["success"], (task, env.private_result())


def test_positive_partial_reward_is_not_success():
    manifest = make_manifest(1, 0, 0, tasks=["click-checkboxes"])
    with BrowserEnvironment(manifest) as env:
        env.reset(manifest.instances[0])
        # Inject a native partial-credit callback to test adapter reward semantics.
        env.native.instance.driver.execute_script(
            "document.getElementById('component-0').contentWindow.core.endEpisode(0.5, true)"
        )
        env.step(MouseAction(x=0, y=0, click=0))
        result = env.private_result()
        assert result["native_component_rewards"] == [0.5]
        assert not result["verdict"]["success"]
        assert not result["verdict"]["goals_valid"]


def test_browser_collection_export_and_numerical_replay(tmp_path):
    manifest = make_manifest(1, 0, 0, tasks=["click-button_click-link"])
    manifest.instances = manifest.instances[:1]
    with BrowserEnvironment(manifest) as env:
        env.reset(manifest.instances[0])
        queries = env.native.instance.driver.execute_script(
            "return [0,1].map(i=>document.getElementById('component-'+i)"
            ".contentWindow.core.getUtterance())"
        )
        commands = [
            target_action(env, i, selector, re.search(r'"(.*?)"', queries[i])[1]).vector()
            for i, selector in enumerate(["#area button", ".alink"])
        ]

    class ScriptedFixture:
        def reset(self, seed):
            self.commands = iter(commands)

        def predict(self, observation):
            return np.asarray([next(self.commands)])

        def provenance(self):
            return {"policy": "privileged_test_fixture_NOT_a_teacher", "prediction_horizon": 1}

    root = tmp_path / "fixture_run"
    run(
        manifest, root, ScriptedFixture(), mode="collect", split="train", max_episodes=1, attempts=1
    )
    directory = next((root / "episodes").iterdir())
    result = json.loads((directory / "private.json").read_text())
    assert result["verdict"]["success"]
    assert result["primitive_transitions"] == 2
    export_hdf5(root, tmp_path / "export")
    dataset = ChunkDataset(tmp_path / "export")
    assert len(dataset) == 2
    assert dataset[1]["is_pad"].sum() == 15
    with BrowserEnvironment(manifest) as env:
        env.reset(manifest.instances[0])
        for line in (directory / "transitions.jsonl").read_text().splitlines():
            row = json.loads(line)
            env.step(MouseAction.model_validate(row["executed"]["action"]))
        assert env.private_result()["verdict"]["success"]
