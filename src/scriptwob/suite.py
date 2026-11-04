"""Collect train demonstrations and held-out expert rollouts with one shared budget."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from selenium.common.exceptions import WebDriverException

from . import collection
from .contracts import ACTION_VERSION, Policy

if TYPE_CHECKING:
    from .schema import Manifest

SPLITS = ("train", "dev", "test")


def _prepare_suite(output: Path, manifest: Manifest, config: dict, limits: dict) -> None:
    metadata = output / "suite.json"
    if (output / "run.json").exists():
        raise collection.ResumeConfigurationError(
            "This output directory contains a single-split run. "
            "Use --fresh or choose a new directory for all-split collection."
        )
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved.get("schema_version") != 1 or not isinstance(saved.get("config"), dict):
            raise collection.ResumeConfigurationError("Unsupported collection suite metadata")
        old_config = saved["config"]
        changed = sorted(key for key in old_config.keys() | config.keys()
                         if old_config.get(key) != config.get(key))
        if changed:
            raise collection.ResumeConfigurationError(
                "Suite resume configuration differs: " + ", ".join(changed)
                + ". Use --fresh or choose a new output directory."
            )
        previous = saved["limits"]
        if limits["attempts"] < previous["attempts"]:
            raise collection.ResumeConfigurationError("Cannot reduce attempts on suite resume")
        for split in SPLITS:
            if limits["selected_episodes"][split] < previous["selected_episodes"][split]:
                raise collection.ResumeConfigurationError(
                    f"Cannot reduce max_episodes for {split} on suite resume"
                )
        snapshot = output / "manifest.json"
        if not snapshot.exists() or json.loads(snapshot.read_text()) != manifest.model_dump(
            mode="json"
        ):
            raise collection.ResumeConfigurationError("Saved suite manifest is missing or changed")
        history = saved.get("limit_history", [])
        if previous != limits:
            history.append({"changed_utc": collection._now(), "old": previous, "new": limits})
        collection.write_json(metadata, {**saved, "limits": limits, "limit_history": history})
    else:
        if any(path.name != ".writer.lock" for path in output.iterdir()):
            raise collection.ResumeConfigurationError(
                "Output directory contains unidentified data. "
                "Use --fresh or choose a new directory for all-split collection."
            )
        collection.write_json(output / "manifest.json", manifest.model_dump(mode="json"))
        collection.write_json(metadata, {
            "schema_version": 1, "created_utc": collection._now(), "config": config,
            "limits": limits, "limit_history": [],
        })


def _has_work(output: Path, selected: list, attempts: int) -> bool:
    by_episode: dict[str, list[dict]] = {}
    for row in collection._all_results(output):
        by_episode.setdefault(row["episode_id"], []).append(row)
    return any(
        not any(row["success"] for row in by_episode.get(episode.id, []))
        and len(by_episode.get(episode.id, [])) < attempts
        for episode in selected
    )


def _report(output: Path, splits: dict, stop: str) -> dict:
    reports = [entry["report"] for entry in splits.values() if entry["report"] is not None]
    report = {
        "schema_version": 1, "kind": "collection_suite", "updated_utc": collection._now(),
        "output_dir": str(output.resolve()), "stop_reason": stop, "splits": splits,
        "protocol": (
            "Train permits retries and exports successful demonstrations. Dev/test each use "
            "one attempt per episode and are never accepted for training. Complete means the "
            "selected attempt schedule finished, not that every task succeeded. The prediction "
            "budget is shared across all splits in this invocation."
        ),
    }
    for key in (
        "new_attempts", "new_predict_calls", "total_attempts", "attempted_episodes",
        "successful_episodes", "accepted_episodes", "selected_episodes",
    ):
        report[key] = sum(child[key] for child in reports)
    report["unattempted_episodes"] = report["selected_episodes"] - report["attempted_episodes"]
    collection.write_json(output / "report.json", report)
    return report


def collect_suite(
    manifest: Manifest,
    output: Path,
    policy: Policy,
    *,
    attempts: int = 3,
    max_episodes: int | None = None,
    max_total_steps: int = 10000,
    headless: bool = True,
    env_factory=None,
) -> dict:
    """Run all manifest splits, preserving split isolation and a single request budget.

    ``max_episodes`` selects a prefix independently in each split. ``attempts``
    applies only to training; held-out evaluation always uses one attempt. Existing
    successes and exhausted attempts are skipped using the single-split protocol.
    """
    if attempts < 1 or max_total_steps < 1 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("Run limits must be positive")
    selected = {
        split: [episode for episode in manifest.episodes if episode.split == split][:max_episodes]
        for split in SPLITS
    }
    config = {
        "manifest_digest": manifest.digest,
        "implementation_digest": collection.implementation_digest(),
        "policy": policy.provenance(), "action_contract": ACTION_VERSION,
        "headless": headless, "mode": "collect_suite", "splits": list(SPLITS),
    }
    limits = {
        "attempts": attempts,
        "selected_episodes": {split: len(episodes) for split, episodes in selected.items()},
        "max_total_steps_per_invocation": max_total_steps,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with collection._writer_lock(output):
        _prepare_suite(output, manifest, config, limits)
        # Validate every child before any browser creation or paid prediction.
        # The runner also takes each child's lock, guarding against direct child writes.
        for split, episodes in selected.items():
            if not episodes:
                continue
            child = output / split
            child.mkdir(exist_ok=True)
            child_config = {key: value for key, value in config.items() if key != "splits"}
            child_config.update(mode="collect" if split == "train" else "evaluate", split=split)
            child_limits = {
                "attempts": attempts if split == "train" else 1,
                "max_episodes": len(episodes),
                "max_total_steps_per_invocation": max_total_steps,
            }
            with collection._writer_lock(child):
                collection._prepare_run(child, child_config, manifest, child_limits)
                collection._recover_interrupted(child)
        splits = {}
        for split, episodes in selected.items():
            child = output / split
            status, report = "empty", None
            if episodes:
                with collection._writer_lock(child):
                    status = "pending" if _has_work(
                        child, episodes, attempts if split == "train" else 1
                    ) else "complete"
                    report = collection._report(child, episodes, 0, 0, status)
            splits[split] = {
                "status": status, "output_dir": str(child.resolve()),
                "ran_this_invocation": False, "report": report,
            }
        _report(output, splits, "starting")
        remaining = max_total_steps
        for split, entry in splits.items():
            if entry["status"] in {"empty", "complete"}:
                continue
            if remaining <= 0:
                return _report(output, splits, "call_budget")
            entry.update(status="running", ran_this_invocation=True)
            _report(output, splits, "running")
            child = output / split
            try:
                report = collection.run(
                    manifest, child, policy,
                    mode="collect" if split == "train" else "evaluate", split=split,
                    attempts=attempts if split == "train" else 1, max_episodes=max_episodes,
                    max_total_steps=remaining, headless=headless, env_factory=env_factory,
                )
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    reason = "interrupted"
                elif isinstance(error, (OSError, TimeoutError, WebDriverException)):
                    reason = "infrastructure_error"
                else:
                    reason = "runner_error"
                entry["status"] = reason
                try:
                    entry["report"] = json.loads((child / "report.json").read_text())
                    _report(output, splits, reason)
                except Exception as report_error:
                    error.add_note(f"Could not refresh suite report: {type(report_error).__name__}")
                raise
            entry["report"] = report
            reason = report["stop_reason"]
            entry["status"] = "complete" if reason == "complete" else "stopped"
            remaining -= report["new_predict_calls"]
            _report(output, splits, reason if reason != "complete" else "running")
            if reason != "complete":
                return _report(output, splits, reason)
        return _report(output, splits, "complete")
