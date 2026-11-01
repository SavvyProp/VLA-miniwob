import json

import numpy as np
import pytest
from PIL import Image

from vla_miniwob.collection import run
from vla_miniwob.contracts import HistoryEntry, PolicyObservation, decode
from vla_miniwob.policies import RandomPolicy
from vla_miniwob.reporting import report
from vla_miniwob.tasks import make_manifest
from vla_miniwob.validation import Event, OrderValidator


class FixtureEnvironment:
    browser_version = "test-double"

    def __init__(self, manifest, **_):
        self.manifest = manifest

    def reset(self, instance):
        self.instance, self.steps, self.terminated = instance, 0, False
        self.validator = OrderValidator(instance.required_order)
        self.history = []
        return self.observe()

    def observe(self):
        return PolicyObservation(
            "fixture instruction", Image.new("RGB", (20, 10)), (0.0, 0.0, 0.0), tuple(self.history)
        )

    def step(self, action):
        self.steps += 1
        executed = decode(action, 20, 10)
        self.history.append(
            HistoryEntry(step=self.steps, action=executed.action, feedback="executed")
        )
        self.validator.ingest(Event(component=0, kind="complete", step=self.steps))
        self.terminated = True
        return self.observe(), executed, "executed"

    def private_result(self, reason=""):
        return {
            "verdict": self.validator.verdict(self.steps, 3, self.terminated, reason).model_dump()
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_resume_keeps_attempts_and_configuration_frozen(tmp_path):
    manifest = make_manifest(1, 0, 0, tasks=["click-button"])
    kwargs = dict(mode="collect", split="train", max_episodes=1, env_factory=FixtureEnvironment)
    assert run(manifest, tmp_path, RandomPolicy(), **kwargs)["new_attempts"] == 1
    assert run(manifest, tmp_path, RandomPolicy(), **kwargs)["new_attempts"] == 1
    assert run(manifest, tmp_path, RandomPolicy(), **kwargs)["new_attempts"] == 0
    summary = report(tmp_path)
    assert summary["accepted_episodes"] == 2
    assert "benchmark_macro_success" not in summary
    with pytest.raises(ValueError, match="configuration differs"):
        run(manifest, tmp_path, RandomPolicy(8), **kwargs)


class BrokenPolicy(RandomPolicy):
    def predict(self, observation):
        return np.array([[np.nan, 0, 1]])


def test_invalid_action_is_retained_and_counts_as_model_failure(tmp_path):
    manifest = make_manifest(0, 0, 1, tasks=["click-button"])
    run(
        manifest,
        tmp_path,
        BrokenPolicy(),
        mode="evaluate",
        split="eval",
        max_episodes=1,
        attempts=1,
        env_factory=FixtureEnvironment,
    )
    result = json.loads(next((tmp_path / "episodes").glob("*/private.json")).read_text())
    assert result["verdict"]["reason"] == "invalid_policy_output"
    assert not result["infrastructure_failure"]
    assert result["primitive_transitions"] == 0
    summary = report(tmp_path)
    assert summary["observed_macro_success"] == 0
    assert summary["benchmark_macro_success"] is None
    assert not summary["complete_evaluation"]


def test_collect_and_eval_protocols_cannot_leak(tmp_path):
    manifest = make_manifest(1, 1, 1)
    with pytest.raises(ValueError, match="training instances"):
        run(manifest, tmp_path, RandomPolicy(), mode="collect", split="eval", max_episodes=1)
    with pytest.raises(ValueError, match="exactly one attempt"):
        run(
            manifest,
            tmp_path,
            RandomPolicy(),
            mode="evaluate",
            split="eval",
            max_episodes=1,
            attempts=2,
        )
