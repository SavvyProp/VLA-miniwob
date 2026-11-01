import json

import numpy as np
import pytest
from PIL import Image

from vla_miniwob import collection
from vla_miniwob.collection import ResumeConfigurationError, implementation_digest, run
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


class MixedOutcomeEnvironment(FixtureEnvironment):
    """Accept direct wording, but exhaust all retries of the second instance."""

    def private_result(self, reason=""):
        if self.instance.wording == "reverse_clauses":
            validator = OrderValidator(self.instance.required_order)
            return {"verdict": validator.verdict(self.steps, 3, True, reason).model_dump()}
        return super().private_result(reason)


def file_snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_collection_retry_extension_preserves_episodes_and_is_idempotent(tmp_path):
    manifest = make_manifest(1, 0, 0, tasks=["click-button"])
    kwargs = dict(
        mode="collect", split="train", max_episodes=10, env_factory=MixedOutcomeEnvironment
    )
    assert run(manifest, tmp_path, RandomPolicy(), attempts=2, **kwargs) == {"new_attempts": 3}
    existing_episodes = file_snapshot(tmp_path / "episodes")

    assert run(manifest, tmp_path, RandomPolicy(), attempts=3, **kwargs) == {"new_attempts": 1}
    for relative, contents in existing_episodes.items():
        assert (tmp_path / "episodes" / relative).read_bytes() == contents
    results = [
        json.loads(path.read_text()) for path in (tmp_path / "episodes").glob("*/private.json")
    ]
    accepted = [row for row in results if row["verdict"]["success"]]
    rejected = [row for row in results if not row["verdict"]["success"]]
    assert len(accepted) == 1
    assert sorted(row["attempt"] for row in rejected) == [0, 1, 2]
    config = json.loads((tmp_path / "run.json").read_text())
    assert config["attempts_per_instance"] == 3
    assert len(config["resume_history"]) == 1
    extension = config["resume_history"][0]
    assert extension["changes"] == {"attempts_per_instance": {"old": 2, "new": 3}}
    assert extension["recorded_attempts"] == 3
    assert extension["changed_utc"]

    before = file_snapshot(tmp_path)
    assert run(manifest, tmp_path, RandomPolicy(), attempts=3, **kwargs) == {"new_attempts": 0}
    assert file_snapshot(tmp_path) == before


def test_collection_retry_limit_cannot_decrease(tmp_path):
    manifest = make_manifest(1, 0, 0, tasks=["click-button"])
    kwargs = dict(mode="collect", split="train", max_episodes=10, env_factory=FixtureEnvironment)
    run(manifest, tmp_path, RandomPolicy(), attempts=3, **kwargs)
    before = file_snapshot(tmp_path)
    with pytest.raises(ResumeConfigurationError, match="attempts_per_instance"):
        run(manifest, tmp_path, RandomPolicy(), attempts=2, **kwargs)
    assert file_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "field", ["policy.prediction_horizon", "runtime.packages.openai", "manifest_digest"]
)
def test_retry_extension_rejects_unrelated_changes_without_writing(tmp_path, monkeypatch, field):
    manifest = make_manifest(1, 0, 0, tasks=["click-button"])
    policy = RandomPolicy()
    kwargs = dict(mode="collect", split="train", max_episodes=1, env_factory=FixtureEnvironment)
    run(manifest, tmp_path, policy, attempts=2, **kwargs)
    before = file_snapshot(tmp_path)
    if field == "policy.prediction_horizon":
        policy = RandomPolicy(8)
    elif field == "runtime.packages.openai":
        original_version = collection.version
        monkeypatch.setattr(
            collection,
            "version",
            lambda name: "changed" if name == "openai" else original_version(name),
        )
    else:
        manifest = manifest.model_copy(update={"max_steps": manifest.max_steps + 1})
    with pytest.raises(ResumeConfigurationError) as error:
        run(manifest, tmp_path, policy, attempts=3, **kwargs)
    assert field in str(error.value)
    assert file_snapshot(tmp_path) == before


LEGACY_IMPLEMENTATION_DIGEST = "631409a5dfc7b723b3f01c28f1546f73a5e6f82ea0c35fe9a331a0891d154946"


def test_known_legacy_collection_resumes_and_records_source_migration(tmp_path):
    manifest = make_manifest(1, 0, 0, tasks=["click-button"])
    kwargs = dict(mode="collect", split="train", max_episodes=1, env_factory=FixtureEnvironment)
    run(manifest, tmp_path, RandomPolicy(), attempts=2, **kwargs)
    existing_episodes = file_snapshot(tmp_path / "episodes")
    run_path = tmp_path / "run.json"
    original = json.loads(run_path.read_text())
    original["implementation_digest"] = LEGACY_IMPLEMENTATION_DIGEST
    run_path.write_text(json.dumps(original))

    assert run(manifest, tmp_path, RandomPolicy(), attempts=3, **kwargs) == {"new_attempts": 1}
    for relative, contents in existing_episodes.items():
        assert (tmp_path / "episodes" / relative).read_bytes() == contents
    updated = json.loads(run_path.read_text())
    assert updated["implementation_digest"] == implementation_digest()
    assert len(updated["resume_history"]) == 1
    migration = updated["resume_history"][0]
    assert migration["recorded_attempts"] == 1
    assert migration["changes"] == {
        "attempts_per_instance": {"old": 2, "new": 3},
        "implementation_digest": {
            "old": LEGACY_IMPLEMENTATION_DIGEST,
            "new": implementation_digest(),
        },
    }


@pytest.mark.parametrize("unknown_target", [False, True])
def test_unrecognized_implementation_cannot_resume(tmp_path, monkeypatch, unknown_target):
    manifest = make_manifest(1, 0, 0, tasks=["click-button"])
    kwargs = dict(mode="collect", split="train", max_episodes=1, env_factory=FixtureEnvironment)
    run(manifest, tmp_path, RandomPolicy(), attempts=2, **kwargs)
    run_path = tmp_path / "run.json"
    original = json.loads(run_path.read_text())
    original["implementation_digest"] = LEGACY_IMPLEMENTATION_DIGEST if unknown_target else "f" * 64
    run_path.write_text(json.dumps(original))
    if unknown_target:
        monkeypatch.setattr(collection, "implementation_digest", lambda: "e" * 64)
    before = file_snapshot(tmp_path)
    with pytest.raises(ResumeConfigurationError, match="implementation_digest"):
        run(manifest, tmp_path, RandomPolicy(), attempts=3, **kwargs)
    assert file_snapshot(tmp_path) == before


def test_legacy_compatibility_does_not_change_evaluation_protocol(tmp_path):
    manifest = make_manifest(0, 0, 1, tasks=["click-button"])
    kwargs = dict(
        mode="evaluate", split="eval", max_episodes=1, attempts=1, env_factory=FixtureEnvironment
    )
    run(manifest, tmp_path, RandomPolicy(), **kwargs)
    run_path = tmp_path / "run.json"
    original = json.loads(run_path.read_text())
    original["implementation_digest"] = LEGACY_IMPLEMENTATION_DIGEST
    run_path.write_text(json.dumps(original))
    before = file_snapshot(tmp_path)
    with pytest.raises(ResumeConfigurationError, match="implementation_digest"):
        run(manifest, tmp_path, RandomPolicy(), **kwargs)
    assert file_snapshot(tmp_path) == before
