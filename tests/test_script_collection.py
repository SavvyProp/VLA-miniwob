"""Collection correctness and crash recovery without a browser or paid inference."""

import fcntl
import hashlib
import json
from types import SimpleNamespace

import pytest
from openai import APIConnectionError
from PIL import Image

from scriptwob.collection import (
    ResumeConfigurationError,
    RunLockedError,
    _all_results,
    attempt_path,
    run,
)
from scriptwob.contracts import ClickAction, HistoryEntry, Observation
from scriptwob.policies import PolicyOutputError
from scriptwob.schema import Episode, Manifest, SemanticAction, Statement, Widget, World


def make_manifest(count=2, split="train"):
    return Manifest(episodes=[Episode(
        id=f"ep-{index}", split=split, level=index % 2, family="sequence", seed=index,
        layout_seed=index, structural_id=f"structure-{index}", max_steps=5,
        instruction="Click Increase, then Submit.",
        program=Statement(kind="action", action=SemanticAction(op="increment", target="count")),
        world=World(widgets=[Widget(id="count", label="Count", kind="counter")]),
    ) for index in range(count)])


class FakeEnvironment:
    browser_version = "mock-browser"
    created = []
    fail = False

    def __init__(self, *, headless=True):
        self.closed = False
        self.created.append(self)

    def reset(self, episode):
        self.episode = episode
        self.steps, self.terminated = 0, False
        self.history = ()
        return self.observation()

    def observation(self):
        return Observation(self.episode.instruction, Image.new("RGB", (17, 17), self.steps),
                           self.history)

    def step(self, action):
        self.steps += 1
        # Match actual browser behavior: report the rounded, executed coordinate.
        px, py = action.pixels(17, 17)
        actual = ClickAction(x=px / 16, y=py / 16)
        self.history += (HistoryEntry(step=self.steps, action=actual, feedback="clicked"),)
        self.terminated = self.steps >= 2
        return self.observation()

    def private_result(self):
        return {"success": self.terminated and not self.fail,
                "private_program": "DO NOT EXPORT", "steps": self.steps}

    def close(self):
        self.closed = True


class Policy:
    def __init__(self, *, fail=False, error=None, name="mock_frontier"):
        self.fail = fail
        self.error = error
        self.name = name
        self.total_calls = 0
        self.calls = []

    def reset(self, seed):
        self.calls = []

    def predict(self, observation):
        self.total_calls += 1
        if self.error is not None:
            raise self.error
        if self.fail:
            raise PolicyOutputError("Invalid mocked action")
        return ClickAction(x=0.333, y=0.667)

    def provenance(self):
        return {"policy": self.name, "supervision": "frontier"}


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeEnvironment.created = []
    FakeEnvironment.fail = False
    # Unit runs span multiple resume calls while other worktree files may be
    # edited by developers. Version-change rejection has its own explicit test.
    monkeypatch.setattr("scriptwob.collection.implementation_digest", lambda: "fixture-code-v1")


def result_files(root):
    return sorted((root / "attempts").glob("*/private.json"))


def test_collection_stores_only_public_transitions_and_matching_image_hashes(tmp_path):
    report = run(make_manifest(), tmp_path, Policy(), env_factory=FakeEnvironment)
    assert report["accepted_episodes"] == 2
    assert report["new_predict_calls"] == 4
    assert FakeEnvironment.created[0].closed
    for path in result_files(tmp_path):
        private = json.loads(path.read_text())
        assert private["accepted"] and private["success"]
        assert private["environment"]["private_program"] == "DO NOT EXPORT"
        public = (path.parent / "episode.json").read_text()
        transitions = (path.parent / "transitions.jsonl").read_text()
        assert "DO NOT EXPORT" not in public + transitions
        rows = [json.loads(line) for line in transitions.splitlines()]
        assert rows[0]["step"] == 1
        assert rows[0]["requested_action"]["x"] == 0.333
        assert rows[0]["action"]["x"] == 0.3125
        for row in rows:
            for key in ("image", "next_image"):
                assert row[key + "_sha256"] == hashlib.sha256(
                    (path.parent / row[key]).read_bytes()
                ).hexdigest()
    assert not list((tmp_path / "attempts").glob(".*.partial"))


def test_resume_extends_episode_limit_without_repeating_accepted_examples(tmp_path):
    manifest = make_manifest(3)
    policy = Policy()
    first = run(manifest, tmp_path, policy, max_episodes=1, env_factory=FakeEnvironment)
    assert first["accepted_episodes"] == 1
    second = run(manifest, tmp_path, policy, max_episodes=3, env_factory=FakeEnvironment)
    assert second["new_attempts"] == 2
    assert second["accepted_episodes"] == 3
    third = run(manifest, tmp_path, policy, max_episodes=3, env_factory=FakeEnvironment)
    assert third["new_attempts"] == 0
    assert len(FakeEnvironment.created) == 2


def test_resume_increases_retry_limit_and_does_not_accept_failed_attempts(tmp_path):
    manifest = make_manifest(1)
    first = run(manifest, tmp_path, Policy(fail=True), attempts=1, env_factory=FakeEnvironment)
    assert first["accepted_episodes"] == 0
    second = run(manifest, tmp_path, Policy(), attempts=3, env_factory=FakeEnvironment)
    assert second["accepted_episodes"] == 1
    assert second["total_attempts"] == 2
    outcomes = [json.loads(path.read_text()) for path in result_files(tmp_path)]
    assert [row["accepted"] for row in outcomes] == [False, True]


def test_changed_policy_manifest_or_reduced_limits_are_rejected(tmp_path):
    manifest = make_manifest(2)
    run(manifest, tmp_path, Policy(), attempts=3, env_factory=FakeEnvironment)
    for options, policy, data in [
        ({"attempts": 2}, Policy(), manifest),
        ({"max_episodes": 1}, Policy(), manifest),
        ({}, Policy(name="different"), manifest),
        ({}, Policy(), make_manifest(3)),
    ]:
        with pytest.raises(ResumeConfigurationError):
            run(data, tmp_path, policy, env_factory=FakeEnvironment, **options)


def test_changed_implementation_fingerprint_rejects_resume(tmp_path, monkeypatch):
    manifest = make_manifest(1)
    run(manifest, tmp_path, Policy(), env_factory=FakeEnvironment)
    monkeypatch.setattr("scriptwob.collection.implementation_digest", lambda: "fixture-code-v2")
    with pytest.raises(ResumeConfigurationError, match="implementation_digest"):
        run(manifest, tmp_path, Policy(), env_factory=FakeEnvironment)


def test_call_budget_bounds_new_model_requests_and_does_not_accept_partial_episode(tmp_path):
    policy = Policy()
    report = run(make_manifest(5), tmp_path, policy, max_total_steps=3,
                 env_factory=FakeEnvironment)
    assert policy.total_calls == 3
    assert report["stop_reason"] == "call_budget"
    assert report["accepted_episodes"] == 1
    assert report["new_attempts"] == 2
    assert json.loads(result_files(tmp_path)[-1].read_text())["reason"] in {
        "success", "call_budget",
    }


def test_provider_error_stops_entire_job_instead_of_retrying_every_episode(tmp_path):
    error = APIConnectionError(request=SimpleNamespace(url="https://example.test"))
    policy = Policy(error=error)
    report = run(make_manifest(5), tmp_path, policy, attempts=4, env_factory=FakeEnvironment)
    assert policy.total_calls == 1
    assert report["new_attempts"] == 1
    assert report["stop_reason"] == "provider_error"
    assert report["accepted_episodes"] == 0


def test_keyboard_interrupt_records_failed_attempt_and_resume_can_retry(tmp_path):
    manifest = make_manifest(1)
    with pytest.raises(KeyboardInterrupt):
        run(manifest, tmp_path, Policy(error=KeyboardInterrupt()), attempts=1,
            env_factory=FakeEnvironment)
    assert FakeEnvironment.created[0].closed
    failed = json.loads(result_files(tmp_path)[0].read_text())
    assert failed["reason"] == "interrupted" and not failed["accepted"]
    interrupted_report = json.loads((tmp_path / "report.json").read_text())
    assert interrupted_report["stop_reason"] == "interrupted"
    assert interrupted_report["total_attempts"] == 1
    assert interrupted_report["new_attempts"] == 1
    assert interrupted_report["new_predict_calls"] == 1
    assert interrupted_report["failures"]["by_reason"]["interrupted"] == 1
    report = run(manifest, tmp_path, Policy(), attempts=2, env_factory=FakeEnvironment)
    assert report["accepted_episodes"] == 1
    assert report["total_attempts"] == 2


def test_browser_startup_failure_leaves_zero_attempt_coverage_report(tmp_path):
    def cannot_start(**kwargs):
        raise OSError("Browser startup unavailable")

    with pytest.raises(OSError, match="Browser startup unavailable"):
        run(make_manifest(2), tmp_path, Policy(), env_factory=cannot_start)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["stop_reason"] == "infrastructure_error"
    assert report["total_attempts"] == 0
    assert report["new_predict_calls"] == 0
    assert report["metrics"]["unattempted_episodes"] == 2


def test_original_exception_survives_cleanup_failure_and_report_is_current(tmp_path):
    class BrokenCleanup(FakeEnvironment):
        def close(self):
            super().close()
            raise OSError("Cleanup failed")

    with pytest.raises(RuntimeError, match="Original runner error") as captured:
        run(make_manifest(1), tmp_path, Policy(error=RuntimeError("Original runner error")),
            env_factory=BrokenCleanup)
    assert "Browser cleanup also failed: OSError" in captured.value.__notes__
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["stop_reason"] == "runner_error"
    assert report["total_attempts"] == 1
    assert report["failures"]["by_reason"]["runner_error"] == 1


def test_hard_killed_partial_attempt_is_counted_and_never_accepted(tmp_path):
    manifest = make_manifest(2)
    run(manifest, tmp_path, Policy(), max_episodes=1, attempts=1,
        env_factory=FakeEnvironment)
    target = attempt_path(tmp_path, manifest.episodes[1], 1)
    staging = target.with_name("." + target.name + ".partial")
    staging.mkdir()
    (staging / "attempt.json").write_text(json.dumps({
        "episode_id": manifest.episodes[1].id, "attempt": 1, "split": "train",
        "level": 1, "family": "sequence",
    }))
    (staging / "progress.json").write_text(json.dumps({
        "predict_calls": 1, "transitions": 0, "inference_pending": True,
    }))
    report = run(manifest, tmp_path, Policy(), max_episodes=2, attempts=2,
                 env_factory=FakeEnvironment)
    assert report["accepted_episodes"] == 2
    assert report["total_attempts"] == 3
    recovered = json.loads((target / "private.json").read_text())
    assert recovered["reason"] == "interrupted"
    assert recovered["predict_calls"] == 1
    assert not recovered["accepted"] and not recovered["provider_usage_complete"]


def test_unpublished_staged_verdict_is_not_a_committed_collection_result(tmp_path):
    manifest = make_manifest(1)
    run(manifest, tmp_path, Policy(), env_factory=FakeEnvironment)
    committed = attempt_path(tmp_path, manifest.episodes[0], 1)
    staging = committed.with_name("." + committed.name + ".partial")
    committed.rename(staging)
    # Models may finish immediately before interruption of the atomic publish.
    # Readers must wait for publication or explicit recovery to count acceptance.
    assert json.loads((staging / "private.json").read_text())["accepted"] is True
    assert _all_results(tmp_path) == []
    recovered = run(manifest, tmp_path, Policy(), env_factory=FakeEnvironment)
    assert recovered["accepted_episodes"] == 1
    assert recovered["new_attempts"] == 0
    assert len(_all_results(tmp_path)) == 1


def test_two_writers_cannot_collect_into_the_same_directory(tmp_path):
    with (tmp_path / ".writer.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RunLockedError):
            run(make_manifest(), tmp_path, Policy(), env_factory=FakeEnvironment)
    assert not FakeEnvironment.created


def test_evaluation_is_single_attempt_and_never_training_accepted(tmp_path):
    report = run(make_manifest(2, "test"), tmp_path, Policy(), mode="evaluate", split="test",
                 attempts=1, env_factory=FakeEnvironment)
    assert report["successful_episodes"] == 2
    assert report["accepted_episodes"] == 0
    with pytest.raises(ValueError, match="restricted"):
        run(make_manifest(1, "test"), tmp_path, Policy(), split="test")
    with pytest.raises(ValueError, match="exactly one"):
        run(make_manifest(1, "dev"), tmp_path, Policy(), mode="evaluate", split="dev", attempts=2)


def test_failed_terminal_environment_is_not_accepted(tmp_path):
    FakeEnvironment.fail = True
    report = run(make_manifest(1), tmp_path, Policy(), attempts=2, env_factory=FakeEnvironment)
    assert report["total_attempts"] == 2
    assert report["accepted_episodes"] == 0


def test_report_separates_retry_yield_first_attempts_and_infrastructure(tmp_path):
    manifest = make_manifest(4)
    run(manifest, tmp_path, Policy(fail=True), max_episodes=1, attempts=1,
        env_factory=FakeEnvironment)

    class LaterProviderFailure(Policy):
        resets = 0

        def reset(self, seed):
            super().reset(seed)
            self.resets += 1

        def predict(self, observation):
            if self.resets == 3:
                raise APIConnectionError(request=SimpleNamespace(url="https://example.test"))
            return super().predict(observation)

    report = run(manifest, tmp_path, LaterProviderFailure(), max_episodes=4, attempts=2,
                 env_factory=FakeEnvironment)
    metrics = report["metrics"]
    # ep-0 succeeds only on its second attempt; ep-1 succeeds immediately;
    # ep-2 has an infrastructure failure, and ep-3 remains unattempted.
    assert metrics["attempted_episodes"] == 3
    assert metrics["unattempted_episodes"] == 1
    assert metrics["accepted_episodes"] == 2
    first = metrics["first_attempt"]
    assert first["successes"] == 1
    assert first["infrastructure_failures"] == 1
    assert first["denominator_all_attempted"] == 3
    assert first["success_rate_including_infrastructure_and_budget"] == pytest.approx(1 / 3)
    assert first["denominator_excluding_infrastructure_and_budget"] == 2
    assert first["success_rate_excluding_infrastructure_and_budget"] == 0.5
    retry = metrics["any_success_with_retries"]
    assert retry["successes"] == 2
    assert retry["rate_including_infrastructure_and_budget"] == pytest.approx(2 / 3)
    assert report["failures"]["provider_error_attempts"] == 1
    assert report["failures"]["infrastructure_attempts"] == 1
    assert report["failures"]["by_reason"] == {
        "invalid_policy_output": 1, "provider_error": 1,
    }
    assert report["levels"]["0"]["first_attempt"]["successes"] == 0
    assert report["levels"]["1"]["first_attempt"]["successes"] == 1


def test_report_distinguishes_collector_budget_from_model_failures(tmp_path):
    report = run(make_manifest(2), tmp_path, Policy(), max_total_steps=3,
                 env_factory=FakeEnvironment)
    first = report["metrics"]["first_attempt"]
    assert first["successes"] == 1
    assert first["call_budget_interruptions"] == 1
    assert first["infrastructure_failures"] == 0
    assert first["denominator_all_attempted"] == 2
    assert first["denominator_excluding_infrastructure_and_budget"] == 1
    assert first["success_rate_including_infrastructure_and_budget"] == 0.5
    assert first["success_rate_excluding_infrastructure_and_budget"] == 1


def test_report_sums_only_reported_usage_and_marks_unknown_provider_cost(tmp_path):
    class MeteredPolicy(Policy):
        def predict(self, observation):
            self.total_calls += 1
            if self.total_calls == 3:
                self.calls.append({"status": "provider_error"})
                raise APIConnectionError(request=SimpleNamespace(url="https://example.test"))
            self.calls.append({
                "status": "completed", "usage": {
                    "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 40},
                    "output_tokens_details": {"reasoning_tokens": 10},
                },
            })
            return ClickAction(x=0.2, y=0.8)

    report = run(make_manifest(2), tmp_path, MeteredPolicy(), env_factory=FakeEnvironment)
    usage = report["provider_usage"]
    assert usage["logged_provider_calls"] == 3
    assert usage["calls_with_usage"] == 2
    assert usage["calls_without_usage"] == 1
    assert usage["attempts_with_incomplete_usage"] == 1
    assert usage["complete"] is False
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 40
    assert usage["total_tokens"] == 240
    assert usage["cached_input_tokens"] == 80
    assert usage["reasoning_output_tokens"] == 20
