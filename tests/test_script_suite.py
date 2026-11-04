"""A complete collection job keeps held-out data separate and shares one call budget."""

import fcntl
import json
from types import SimpleNamespace

import pytest
from openai import APIConnectionError
from test_script_collection import FakeEnvironment, Policy

from scriptwob.collection import ResumeConfigurationError, RunLockedError, _all_results, run
from scriptwob.export import export_dataset
from scriptwob.generation import generate_manifest
from scriptwob.suite import collect_suite


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeEnvironment.created = []
    FakeEnvironment.fail = False
    monkeypatch.setattr("scriptwob.collection.implementation_digest", lambda: "suite-code-v1")


def manifest(train=1, dev=1, test=1):
    return generate_manifest(train=train, dev=dev, test=test, seed=23, levels=[0])


def collect(data, root, policy=None, **kwargs):
    return collect_suite(data, root, policy or Policy(), env_factory=FakeEnvironment, **kwargs)


def test_all_splits_are_isolated_and_only_train_can_be_exported(tmp_path):
    output = tmp_path / "suite"
    report = collect(manifest(), output, attempts=4)
    assert report["stop_reason"] == "complete"
    assert report["new_attempts"] == report["successful_episodes"] == 3
    assert report["new_predict_calls"] == 6
    assert report["accepted_episodes"] == 1
    assert report["unattempted_episodes"] == 0
    assert json.loads((output / "report.json").read_text()) == report
    assert (output / "suite.json").exists()
    assert not (output / "run.json").exists()
    for split in ("train", "dev", "test"):
        entry = report["splits"][split]
        assert entry["status"] == "complete" and entry["ran_this_invocation"]
        saved = json.loads((output / split / "run.json").read_text())
        assert saved["config"]["split"] == split
        assert saved["config"]["mode"] == ("collect" if split == "train" else "evaluate")
        assert saved["limits"]["attempts"] == (4 if split == "train" else 1)
        assert _all_results(output / split)[0]["accepted"] is (split == "train")
    exported = export_dataset(output / "train", tmp_path / "training")
    assert exported["episodes"] == 1
    for split in ("dev", "test"):
        with pytest.raises(ValueError, match="Only a train collection"):
            export_dataset(output / split, tmp_path / f"bad-{split}")
    assert len(FakeEnvironment.created) == 3
    assert all(environment.closed for environment in FakeEnvironment.created)


@pytest.mark.parametrize("budget,attempts,dev_status", [(2, 1, "pending"), (3, 2, "stopped")])
def test_prediction_budget_is_shared_across_splits(tmp_path, budget, attempts, dev_status):
    policy = Policy()
    report = collect(manifest(), tmp_path, policy, max_total_steps=budget)
    assert policy.total_calls == report["new_predict_calls"] == budget
    assert report["new_attempts"] == attempts
    assert report["stop_reason"] == "call_budget"
    assert report["splits"]["train"]["status"] == "complete"
    assert report["splits"]["dev"]["status"] == dev_status
    assert report["splits"]["test"]["status"] == "pending"
    assert report["splits"]["test"]["report"]["attempted_episodes"] == 0
    assert report["selected_episodes"] == 3


def test_exact_total_budget_finishes_and_completed_resume_needs_no_browser(tmp_path):
    data = manifest()
    first = collect(data, tmp_path, max_total_steps=6)
    assert first["stop_reason"] == "complete"
    first_environments = len(FakeEnvironment.created)
    resumed = collect(data, tmp_path, max_total_steps=1)
    assert resumed["stop_reason"] == "complete"
    assert resumed["new_attempts"] == resumed["new_predict_calls"] == 0
    assert len(FakeEnvironment.created) == first_environments
    assert all(not entry["ran_this_invocation"] for entry in resumed["splits"].values())


def test_resume_after_budget_moves_to_unrun_splits_without_repeating_successes(tmp_path):
    data = manifest()
    first = collect(data, tmp_path, max_total_steps=2)
    assert first["accepted_episodes"] == 1
    resumed = collect(data, tmp_path, max_total_steps=4)
    assert resumed["new_predict_calls"] == 4
    assert resumed["new_attempts"] == 2
    assert resumed["successful_episodes"] == 3
    assert resumed["stop_reason"] == "complete"
    assert not resumed["splits"]["train"]["ran_this_invocation"]
    assert len(_all_results(tmp_path / "train")) == 1


def test_provider_failure_in_dev_stops_test_and_keeps_train_data(tmp_path):
    class FailingInDev(Policy):
        def predict(self, observation):
            if self.total_calls == 2:
                self.error = APIConnectionError(request=SimpleNamespace(url="https://example.test"))
            return super().predict(observation)

    policy = FailingInDev()
    report = collect(manifest(), tmp_path, policy)
    assert report["stop_reason"] == "provider_error"
    assert report["accepted_episodes"] == 1
    assert report["new_predict_calls"] == 3
    assert report["new_attempts"] == 2
    assert report["splits"]["test"]["status"] == "pending"
    assert report["splits"]["test"]["report"]["attempted_episodes"] == 0
    assert json.loads((tmp_path / "report.json").read_text()) == report


@pytest.mark.parametrize("error,reason", [
    (KeyboardInterrupt(), "interrupted"),
    (RuntimeError("inference bug"), "runner_error"),
])
def test_interruptions_and_unexpected_failures_persist_suite_counts(tmp_path, error, reason):
    with pytest.raises(type(error)):
        collect(manifest(), tmp_path, Policy(error=error))
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["stop_reason"] == reason
    assert report["new_attempts"] == report["new_predict_calls"] == 1
    assert report["splits"]["train"]["status"] == reason
    assert report["splits"]["dev"]["status"] == "pending"
    assert report["splits"]["test"]["status"] == "pending"
    assert FakeEnvironment.created[0].closed


def test_empty_splits_are_explicit_and_need_no_child_directory(tmp_path):
    report = collect(manifest(dev=0, test=0), tmp_path)
    assert report["stop_reason"] == "complete"
    assert report["selected_episodes"] == report["accepted_episodes"] == 1
    assert report["splits"]["dev"]["status"] == "empty"
    assert report["splits"]["dev"]["report"] is None
    assert not (tmp_path / "dev").exists()


def test_attempt_cap_completion_is_distinct_from_success_and_coverage(tmp_path):
    FakeEnvironment.fail = True
    report = collect(manifest(), tmp_path, attempts=3)
    assert report["stop_reason"] == "complete"
    assert report["new_attempts"] == 5
    assert report["successful_episodes"] == report["accepted_episodes"] == 0
    assert report["attempted_episodes"] == 3
    assert report["unattempted_episodes"] == 0
    assert len(_all_results(tmp_path / "dev")) == 1
    assert len(_all_results(tmp_path / "test")) == 1


def test_episode_limit_applies_per_split_and_can_be_increased(tmp_path):
    data = manifest(train=2, dev=2, test=2)
    first = collect(data, tmp_path, max_episodes=1)
    assert first["selected_episodes"] == 3
    second = collect(data, tmp_path, max_episodes=2)
    assert second["selected_episodes"] == 6
    assert second["new_attempts"] == 3
    with pytest.raises(ResumeConfigurationError, match="max_episodes"):
        collect(data, tmp_path, max_episodes=1)


def test_reject_changed_suite_settings_without_requests(tmp_path, monkeypatch):
    data = manifest()
    collect(data, tmp_path)
    for kwargs in ({"headless": False}, {"attempts": 2}):
        policy = Policy()
        with pytest.raises(ResumeConfigurationError):
            collect(data, tmp_path, policy, **kwargs)
        assert policy.total_calls == 0
    with pytest.raises(ResumeConfigurationError, match="policy"):
        collect(data, tmp_path, Policy(name="different"))
    with pytest.raises(ResumeConfigurationError, match="manifest_digest"):
        collect(manifest(train=2), tmp_path)
    monkeypatch.setattr("scriptwob.collection.implementation_digest", lambda: "new-code")
    with pytest.raises(ResumeConfigurationError, match="implementation_digest"):
        collect(data, tmp_path)


def test_tampered_later_child_rejected_before_new_train_calls(tmp_path):
    data = manifest(train=2, dev=2, test=2)
    collect(data, tmp_path, max_episodes=1)
    path = tmp_path / "test" / "run.json"
    value = json.loads(path.read_text())
    value["config"]["policy"]["policy"] = "different"
    path.write_text(json.dumps(value))
    policy = Policy()
    previous_browsers = len(FakeEnvironment.created)
    with pytest.raises(ResumeConfigurationError, match="policy"):
        collect(data, tmp_path, policy, max_episodes=2)
    assert policy.total_calls == 0
    assert len(FakeEnvironment.created) == previous_browsers


def test_existing_single_split_directory_is_preserved_and_rejected(tmp_path):
    existing = tmp_path / "run.json"
    existing.write_text('{"old": "run metadata"}')
    policy = Policy()
    with pytest.raises(ResumeConfigurationError, match="single-split run"):
        collect(manifest(), tmp_path, policy)
    assert existing.read_text() == '{"old": "run metadata"}'
    assert policy.total_calls == 0
    assert not (tmp_path / "suite.json").exists()


@pytest.mark.parametrize("mode,split", [("collect", "train"), ("evaluate", "dev")])
def test_single_split_run_cannot_overwrite_suite_root(tmp_path, mode, split):
    data = manifest()
    collect(data, tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    previous_browsers = len(FakeEnvironment.created)
    policy = Policy()
    with pytest.raises(ResumeConfigurationError, match="collect --all-splits"):
        run(data, tmp_path, policy, mode=mode, split=split, attempts=1,
            env_factory=FakeEnvironment)
    assert policy.total_calls == 0
    assert len(FakeEnvironment.created) == previous_browsers
    assert not (tmp_path / "run.json").exists()
    assert not (tmp_path / "attempts").exists()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_writer_lock_blocks_concurrent_suite(tmp_path):
    with (tmp_path / ".writer.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RunLockedError):
            collect(manifest(), tmp_path)
    assert not FakeEnvironment.created
