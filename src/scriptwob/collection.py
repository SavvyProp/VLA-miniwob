"""Bounded, locked, resumable collection with public trajectories and private verdicts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import OpenAIError
from pydantic import ValidationError
from selenium.common.exceptions import WebDriverException

from .contracts import ACTION_VERSION, ClickAction, Policy
from .policies import OraclePolicy, PolicyOutputError

if TYPE_CHECKING:
    from .environment import BrowserEnvironment
    from .schema import Episode, Manifest


class ResumeConfigurationError(ValueError):
    """Existing data were produced with a different immutable run configuration."""


class RunLockedError(RuntimeError):
    """Another process holds this run's writer lock."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Replace metadata atomically; never leave a partially written result file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def implementation_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".html", ".js", ".css"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt_path(root: Path, episode: Episode, attempt: int) -> Path:
    key = hashlib.sha256(episode.id.encode()).hexdigest()[:24]
    return root / "attempts" / f"{key}-{attempt:03d}"


@contextmanager
def _writer_lock(root: Path):
    # flock releases automatically after crashes. Never unlink this inode: doing
    # so could allow two writers to lock different files at the same pathname.
    with (root / ".writer.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLockedError(f"Another process is writing {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _prepare_run(output: Path, config: dict, manifest: Manifest, limits: dict) -> None:
    if (output / "suite.json").exists():
        raise ResumeConfigurationError(
            "This output directory contains an all-split collection suite. "
            "Use collect --all-splits, choose the matching split subdirectory, "
            "or start in a new output directory."
        )
    path = output / "run.json"
    if path.exists():
        saved = json.loads(path.read_text())
        old_config = saved.get("config", {})
        changed = sorted(key for key in old_config.keys() | config.keys()
                         if old_config.get(key) != config.get(key))
        if changed:
            raise ResumeConfigurationError(
                "Resume configuration differs: " + ", ".join(changed)
                + ". Keep the original settings or use a new output directory."
            )
        previous = saved["limits"]
        for name in ("attempts", "max_episodes"):
            if limits[name] < previous[name]:
                raise ResumeConfigurationError(
                    f"Cannot reduce {name} from {previous[name]} to {limits[name]} on resume"
                )
        snapshot = output / "manifest.json"
        if not snapshot.exists() or json.loads(snapshot.read_text()) != manifest.model_dump(
            mode="json"
        ):
            raise ResumeConfigurationError("Saved manifest snapshot is missing or changed")
        history = saved.get("limit_history", [])
        if previous != limits:
            history.append({"changed_utc": _now(), "old": previous, "new": limits})
        write_json(path, {**saved, "limits": limits, "limit_history": history})
    else:
        if (output / "attempts").exists() and any((output / "attempts").iterdir()):
            raise ResumeConfigurationError("Unidentified attempts exist without run.json")
        write_json(output / "manifest.json", manifest.model_dump(mode="json"))
        write_json(path, {
            "schema_version": 1, "created_utc": _now(), "config": config,
            "limits": limits, "limit_history": [],
        })
    (output / "attempts").mkdir(exist_ok=True)


def _recover_interrupted(output: Path) -> None:
    """Hard-killed attempts count toward the retry cap and are never accepted."""
    for staging in sorted((output / "attempts").glob(".*.partial")):
        metadata_path = staging / "attempt.json"
        if not metadata_path.exists():
            # Before the first atomic metadata write no API request can occur.
            shutil.rmtree(staging)
            continue
        metadata = json.loads(metadata_path.read_text())
        target = staging.with_name(staging.name[1:-len(".partial")])
        if target.exists():
            raise RuntimeError(f"Both staged and committed attempt exist: {target.name}")
        if not (staging / "private.json").exists():
            calls_path = staging / "provider_calls.json"
            progress_path = staging / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            supervision = json.loads((output / "run.json").read_text())["config"]["policy"].get(
                "supervision", "model"
            )
            write_json(staging / "private.json", {
                **metadata, "accepted": False, "success": False,
                "reason": "interrupted", "infrastructure_failure": True,
                "error_type": None, "finished_utc": _now(),
                "primitive_transitions": progress.get("transitions", 0),
                "predict_calls": progress.get("predict_calls", 0),
                "supervision": supervision, "environment": {},
                "provider_calls": json.loads(calls_path.read_text()) if calls_path.exists() else [],
                "provider_usage_complete": False,
                "note": "Process stopped before commit; in-flight provider usage may be unknown.",
            })
        staging.replace(target)


def _feedback(observation) -> str:
    return observation.history[-1].feedback if observation.history else "clicked"


def _record_calls(directory: Path, policy: Policy) -> None:
    write_json(directory / "provider_calls.json", getattr(policy, "calls", []))


def run_episode(
    env: BrowserEnvironment,
    policy: Policy,
    episode: Episode,
    directory: Path,
    *,
    attempt: int,
    mode: str,
    call_budget: int,
) -> dict:
    """Stage an attempt, then publish its verdict and files as one directory rename."""
    staging = directory.with_name("." + directory.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    seed = int.from_bytes(hashlib.sha256(f"{episode.id}:{attempt}".encode()).digest()[:8], "big")
    metadata = {
        "schema_version": 1, "episode_id": episode.id, "split": episode.split,
        "level": episode.level, "family": episode.family, "attempt": attempt,
        "rollout_seed": seed, "started_utc": _now(),
    }
    write_json(staging / "attempt.json", metadata)
    reason, error_type, infrastructure = "", None, False
    calls, transitions, reset_ok = 0, 0, False
    private: dict = {}
    try:
        policy.reset(seed)
        observation = env.reset(episode)
        reset_ok = True
        observation.image.save(staging / "000000.png")
        write_json(staging / "episode.json", {
            **metadata, "action_contract": ACTION_VERSION,
            "instruction": observation.instruction,
            "image_size": list(observation.image.size), "max_steps": episode.max_steps,
            "supervision": policy.provenance().get("supervision", "model"),
        })
        with (staging / "transitions.jsonl").open("w") as stream:
            while not env.terminated and env.steps < episode.max_steps:
                if calls >= call_budget:
                    reason = "call_budget"
                    break
                calls += 1
                # Persist intent before inference; a crash during the network call
                # remains distinguishable from an episode with no provider requests.
                write_json(staging / "progress.json", {
                    "predict_calls": calls, "transitions": transitions,
                    "inference_pending": True,
                })
                try:
                    action = ClickAction.model_validate(policy.predict(observation))
                finally:
                    _record_calls(staging, policy)
                following = env.step(action)
                image_name, next_image = f"{transitions:06d}.png", f"{transitions + 1:06d}.png"
                following.image.save(staging / next_image)
                row = {
                    "step": transitions + 1,
                    "image": image_name, "image_sha256": _image_digest(staging / image_name),
                    "requested_action": action.model_dump(mode="json"),
                    "action": following.history[-1].action.model_dump(mode="json"),
                    "feedback": _feedback(following),
                    "next_image": next_image,
                    "next_image_sha256": _image_digest(staging / next_image),
                    "terminal": env.terminated,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                transitions += 1
                observation = following
                write_json(staging / "progress.json", {
                    "predict_calls": calls, "transitions": transitions,
                    "inference_pending": False,
                })
        if not reason and not env.terminated:
            reason = "step_limit"
    except (PolicyOutputError, ValidationError) as exc:
        reason, error_type = "invalid_policy_output", type(exc).__name__
    except OpenAIError as exc:
        reason, error_type, infrastructure = "provider_error", type(exc).__name__, True
    except KeyboardInterrupt:
        reason, error_type, infrastructure = "interrupted", "KeyboardInterrupt", True
        raise
    except (OSError, TimeoutError, WebDriverException) as exc:
        reason, error_type, infrastructure = "infrastructure_error", type(exc).__name__, True
    except Exception as exc:
        # Unexpected code failures are persisted but propagated, never silently
        # counted as a policy's semantic failure.
        reason, error_type, infrastructure = "runner_error", type(exc).__name__, True
        raise
    finally:
        if reset_ok:
            try:
                private = env.private_result()
            except Exception as exc:
                reason, error_type, infrastructure = (
                    "infrastructure_error", type(exc).__name__, True
                )
        success = bool(private.get("success", False)) and not reason
        result = {
            **metadata, "finished_utc": _now(),
            "success": success,
            "accepted": success and mode == "collect" and episode.split == "train",
            "reason": reason or ("success" if success else "task_failure"),
            "infrastructure_failure": infrastructure, "error_type": error_type,
            "primitive_transitions": transitions, "predict_calls": calls,
            "browser_version": getattr(env, "browser_version", None),
            "supervision": policy.provenance().get("supervision", "model"),
            "provider_calls": getattr(policy, "calls", []),
            "provider_usage_complete": reason != "interrupted" and all(
                isinstance(call.get("usage"), dict) for call in getattr(policy, "calls", [])
            ),
            "environment": private,
        }
        _record_calls(staging, policy)
        write_json(staging / "private.json", result)
        staging.replace(directory)
    return result


def _all_results(output: Path) -> list[dict]:
    return [json.loads(path.read_text())
            for path in sorted((output / "attempts").glob("*/private.json"))
            if not path.parent.name.startswith(".")]


def _report(output: Path, selected: list, new_attempts: int, new_calls: int, stop: str) -> dict:
    results = _all_results(output)
    by_episode: dict[str, list[dict]] = {}
    for row in results:
        by_episode.setdefault(row["episode_id"], []).append(row)
    succeeded = {key for key, rows in by_episode.items() if any(row["success"] for row in rows)}
    accepted = {row["episode_id"] for row in results if row["accepted"]}
    def metrics(episodes: list) -> dict:
        attempted = [episode for episode in episodes if episode.id in by_episode]
        first = [min(by_episode[episode.id], key=lambda row: row["attempt"])
                 for episode in attempted]
        first_successes = sum(row["success"] for row in first)
        first_infrastructure = sum(row["infrastructure_failure"] for row in first)
        first_budget = sum(row["reason"] == "call_budget" for row in first)
        valid = [row for row in first if not row["infrastructure_failure"]
                 and row["reason"] != "call_budget"]
        any_successes = sum(episode.id in succeeded for episode in attempted)
        return {
            "selected_episodes": len(episodes), "attempted_episodes": len(attempted),
            "unattempted_episodes": len(episodes) - len(attempted),
            "accepted_episodes": sum(episode.id in accepted for episode in attempted),
            "first_attempt": {
                "successes": first_successes,
                "infrastructure_failures": first_infrastructure,
                "call_budget_interruptions": first_budget,
                "denominator_all_attempted": len(first),
                "success_rate_including_infrastructure_and_budget": (
                    first_successes / len(first) if first else None
                ),
                "denominator_excluding_infrastructure_and_budget": len(valid),
                "success_rate_excluding_infrastructure_and_budget": (
                    sum(row["success"] for row in valid) / len(valid) if valid else None
                ),
            },
            "any_success_with_retries": {
                "successes": any_successes,
                "denominator_attempted_episodes": len(attempted),
                "rate_including_infrastructure_and_budget": (
                    any_successes / len(attempted) if attempted else None
                ),
            },
        }

    levels = {str(level): metrics([ep for ep in selected if ep.level == level])
              for level in sorted({episode.level for episode in selected})}
    usage = {
        "logged_provider_calls": 0, "calls_with_usage": 0, "calls_without_usage": 0,
        "attempts_with_incomplete_usage": 0,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0,
        "note": "Token totals sum reported usage only; missing usage is not zero cost.",
    }
    for row in results:
        calls = row.get("provider_calls", [])
        usage["logged_provider_calls"] += len(calls)
        if not row.get("provider_usage_complete", False):
            usage["attempts_with_incomplete_usage"] += 1
        for call in calls:
            tokens = call.get("usage")
            if not isinstance(tokens, dict):
                usage["calls_without_usage"] += 1
                continue
            usage["calls_with_usage"] += 1
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                usage[name] += tokens.get(name, 0) or 0
            usage["cached_input_tokens"] += (
                tokens.get("input_tokens_details") or {}
            ).get("cached_tokens", 0) or 0
            usage["reasoning_output_tokens"] += (
                tokens.get("output_tokens_details") or {}
            ).get("reasoning_tokens", 0) or 0
    usage["complete"] = usage["attempts_with_incomplete_usage"] == 0
    report = {
        "schema_version": 1, "updated_utc": _now(),
        "output_dir": str(output.resolve()),
        "new_attempts": new_attempts, "new_predict_calls": new_calls,
        "total_attempts": len(results), "attempted_episodes": len(by_episode),
        "successful_episodes": len(succeeded), "accepted_episodes": len(accepted),
        "selected_episodes": len(selected), "stop_reason": stop,
        "metrics": metrics(selected), "levels": levels,
        "failures": {
            "infrastructure_attempts": sum(row["infrastructure_failure"] for row in results),
            "provider_error_attempts": sum(row["reason"] == "provider_error" for row in results),
            "by_reason": dict(sorted(Counter(
                row["reason"] for row in results if not row["success"]
            ).items())),
        },
        "provider_usage": usage,
        "protocol": (
            "First-attempt rates use one earliest attempt per attempted episode; unattempted "
            "episodes are reported separately. Any-success rates include retries and measure "
            "collection yield, not single-attempt policy performance. Infrastructure failures "
            "and collection call-budget interruptions have explicit alternative denominators."
        ),
    }
    write_json(output / "report.json", report)
    return report


def _report_exception(output: Path, selected: list, before: list[dict], error: BaseException):
    """Include attempts committed before an exception without replacing that exception."""
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        reason = "interrupted"
    elif isinstance(error, (OSError, TimeoutError, WebDriverException)):
        reason = "infrastructure_error"
    else:
        reason = "runner_error"
    try:
        previous = {(row["episode_id"], row["attempt"]) for row in before}
        added = [row for row in _all_results(output)
                 if (row["episode_id"], row["attempt"]) not in previous]
        _report(output, selected, len(added), sum(row.get("predict_calls", 0) for row in added),
                reason)
    except Exception as report_error:
        error.add_note(f"Could not refresh report.json: {type(report_error).__name__}")


def run(
    manifest: Manifest,
    output: Path,
    policy: Policy,
    *,
    mode: str = "collect",
    split: str = "train",
    max_episodes: int | None = None,
    attempts: int = 3,
    max_total_steps: int = 10000,
    headless: bool = True,
    env_factory=None,
) -> dict:
    """Resume a prefix of a split; per-invocation predict budget includes failed calls.

    Increasing ``max_episodes`` extends the selected manifest prefix. Increasing
    ``attempts`` allows additional retries of unsuccessful episodes. Both changes
    preserve accepted demonstrations; neither changes the immutable fingerprint.
    ``max_total_steps`` limits new predict calls, including oracle/model failures,
    in this invocation. SDK transport retries are separately declared by the policy.
    """
    if mode not in {"collect", "evaluate"} or split not in {"train", "dev", "test"}:
        raise ValueError("Invalid mode or split")
    if attempts < 1 or max_total_steps < 1 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("Run limits must be positive")
    if mode == "collect" and split != "train":
        raise ValueError("Expert demonstration collection is restricted to the train split")
    if mode == "evaluate" and (split == "train" or attempts != 1):
        raise ValueError("Evaluation requires dev/test and exactly one attempt per episode")
    selected = [episode for episode in manifest.episodes if episode.split == split]
    if not selected:
        raise ValueError(f"No {split} episodes in manifest")
    selected = selected[:max_episodes]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "manifest_digest": manifest.digest, "implementation_digest": implementation_digest(),
        "mode": mode, "split": split, "policy": policy.provenance(),
        "action_contract": ACTION_VERSION, "headless": headless,
    }
    limits = {
        "attempts": attempts, "max_episodes": len(selected),
        "max_total_steps_per_invocation": max_total_steps,
    }
    with _writer_lock(output):
        _prepare_run(output, config, manifest, limits)
        _recover_interrupted(output)
        recorded = _all_results(output)
        by_episode: dict[str, list[dict]] = {}
        for row in recorded:
            by_episode.setdefault(row["episode_id"], []).append(row)
        work = [episode for episode in selected if not any(
            row["success"] for row in by_episode.get(episode.id, [])
        ) and len(by_episode.get(episode.id, [])) < attempts]
        if not work:
            return _report(output, selected, 0, 0, "complete")
        # Even browser startup failures leave an honest coverage report. Creating
        # a browser has not yet consumed an episode attempt or a model request.
        _report(output, selected, 0, 0, "starting")
        environment = None
        run_error = None
        new_attempts, new_calls, stop = 0, 0, "complete"
        try:
            if env_factory is None:
                from .environment import BrowserEnvironment
                env_factory = BrowserEnvironment
            environment = env_factory(headless=headless)
            if isinstance(policy, OraclePolicy):
                policy.bind_environment(environment)
            for episode in work:
                previous = by_episode.get(episode.id, [])
                next_attempt = max((row["attempt"] for row in previous), default=0) + 1
                for attempt in range(next_attempt, attempts + 1):
                    if new_calls >= max_total_steps:
                        stop = "call_budget"
                        break
                    result = run_episode(
                        environment, policy, episode, attempt_path(output, episode, attempt),
                        attempt=attempt, mode=mode, call_budget=max_total_steps - new_calls,
                    )
                    new_attempts += 1
                    new_calls += result["predict_calls"]
                    _report(output, selected, new_attempts, new_calls, result["reason"])
                    if result["infrastructure_failure"]:
                        # Stop on an API/network/environment failure, rather than
                        # consuming every remaining episode or repeatedly billing.
                        stop = result["reason"]
                        break
                    if result["reason"] == "call_budget":
                        stop = "call_budget"
                        break
                    if result["success"]:
                        break
                if stop != "complete":
                    break
        except BaseException as error:
            run_error = error
            _report_exception(output, selected, recorded, error)
            raise
        finally:
            if environment is not None:
                try:
                    environment.close()
                except BaseException as close_error:
                    if run_error is None:
                        _report_exception(output, selected, recorded, close_error)
                        raise
                    run_error.add_note(
                        f"Browser cleanup also failed: {type(close_error).__name__}"
                    )
        return _report(output, selected, new_attempts, new_calls, stop)
