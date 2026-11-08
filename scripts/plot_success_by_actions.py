"""Pool evaluation results by verified reference action count and plot with Matplotlib.

Run with the SeeClick environment (which includes Matplotlib):
    integrations/seeclick/.venv/bin/python scripts/plot_success_by_actions.py \
        runs/seeclick-800-devicefix-dev runs/seeclick-800-devicefix-test

No model inference, browser, or API calls are needed. Each input must be a
completed evaluation split with one attempt per task and the same run settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from scriptwob.engine import oracle_trace
from scriptwob.schema import Manifest


def read_evaluations(run_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    rows = []
    sources = []
    seen = set()
    common_config = None
    for run_path in run_paths:
        config = json.loads((run_path / "run.json").read_text())["config"]
        report = json.loads((run_path / "report.json").read_text())
        split = config["split"]
        if config["mode"] != "evaluate" or split not in {"dev", "test"}:
            raise ValueError(f"Expected a dev/test evaluation: {run_path}")
        if report["stop_reason"] != "complete":
            raise ValueError(f"Evaluation is incomplete: {run_path}")
        comparable = {key: value for key, value in config.items() if key != "split"}
        if common_config is not None and comparable != common_config:
            raise ValueError(f"Evaluation settings/checkpoint differ: {run_path}")
        common_config = comparable

        manifest = Manifest.model_validate_json((run_path / "manifest.json").read_text())
        if manifest.digest != config["manifest_digest"]:
            raise ValueError(f"Saved manifest digest mismatch: {run_path}")
        episodes = {episode.id: episode for episode in manifest.episodes if episode.split == split}
        run_rows = []
        for path in sorted((run_path / "attempts").glob("*/private.json")):
            if path.parent.name.startswith("."):
                continue
            attempt = json.loads(path.read_text())
            episode_id = attempt["episode_id"]
            if episode_id in seen:
                raise ValueError(f"Duplicate task/attempt; expected one attempt: {episode_id}")
            seen.add(episode_id)
            episode = episodes[episode_id]
            if attempt["split"] != split or attempt["level"] != episode.level:
                raise ValueError(f"Attempt metadata does not match manifest: {path}")
            if attempt["infrastructure_failure"]:
                raise ValueError(f"Infrastructure failure requires separate analysis: {path}")

            trace, _ = oracle_trace(episode)
            required_actions = len(trace)
            if trace[-1].op != "done" or required_actions != episode.complexity.reference_actions:
                raise ValueError(f"Reference trace disagrees with saved action count: {episode_id}")
            run_rows.append({
                "episode_id": episode_id,
                "split": split,
                "level": episode.level,
                "family": episode.family,
                "required_actions": required_actions,
                "required_actions_excluding_done": required_actions - 1,
                "actual_actions": attempt["primitive_transitions"],
                "success": int(attempt["success"]),
                "reason": attempt["reason"],
                "attempt_path": str(path.parent),
            })

        successes = sum(row["success"] for row in run_rows)
        if (
            len(run_rows) != report["attempted_episodes"]
            or len(run_rows) != report["selected_episodes"]
            or len(run_rows) != report["total_attempts"]
            or successes != report["successful_episodes"]
        ):
            raise ValueError(f"Saved report disagrees with attempt totals: {run_path}")
        rows.extend(run_rows)
        sources.append({
            "run": str(run_path),
            "split": split,
            "tasks": len(run_rows),
            "successes": successes,
            "manifest_digest": manifest.digest,
            "run_json_sha256": hashlib.sha256((run_path / "run.json").read_bytes()).hexdigest(),
        })
    if not rows:
        raise ValueError("No evaluated tasks found")
    return sorted(rows, key=lambda row: (row["required_actions"], row["episode_id"])), sources


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["required_actions"]].append(row)
    return [{
        "required_actions": length,
        "successes": sum(row["success"] for row in group),
        "tasks": len(group),
        "success_rate_percent": 100 * sum(row["success"] for row in group) / len(group),
        **{f"L{level}_tasks": sum(row["level"] == level for row in group) for level in range(6)},
    } for length, group in sorted(groups.items())]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(groups: list[dict], rows: list[dict], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    lengths = [group["required_actions"] for group in groups]
    rates = [group["success_rate_percent"] for group in groups]
    successes = sum(row["success"] for row in rows)
    splits = " + ".join(sorted({row["split"] for row in rows}))
    levels = sorted({row["level"] for row in rows})
    level_label = "L0–L5" if levels == list(range(6)) else ", ".join(f"L{n}" for n in levels)

    fig, ax = plt.subplots(figsize=(12, 5.7))
    bars = ax.bar(lengths, rates, color="tab:blue", width=0.72)
    ax.bar_label(
        bars,
        labels=[f"{group['successes']}/{group['tasks']}" for group in groups],
        padding=5,
        fontsize=9,
    )
    ax.set_xticks(range(min(lengths), max(lengths) + 1))
    ax.set_xlim(min(lengths) - 0.7, max(lengths) + 0.7)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Required actions to complete the task (including Done)")
    ax.set_ylabel("Task success rate")
    ax.set_title(
        f"SeeClick: success by required action count ({splits}, {level_label} pooled)\n"
        f"{len(rows)} tasks · {successes} successes · {100 * successes / len(rows):.1f}% overall"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5, 0.025,
        "Labels: successful / evaluated tasks. Unlabelled action counts have no tasks.\n"
        "One action = Done only. Sample sizes and level composition vary across action counts.",
        ha="center", fontsize=9, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output / "success_by_actions.png", dpi=200)
    fig.savefig(output / "success_by_actions.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Completed evaluation run directories")
    parser.add_argument("--output", type=Path, default=Path("reports/seeclick-800-combined"))
    args = parser.parse_args()
    rows, sources = read_evaluations(args.runs)
    groups = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "actions_per_task.csv", rows)
    write_csv(args.output / "success_by_actions.csv", groups)
    successes = sum(row["success"] for row in rows)
    result = {
        "tasks": len(rows),
        "successes": successes,
        "success_rate_percent": 100 * successes / len(rows),
        "action_count_definition": "Length of replayed oracle trace, including final Done click.",
        "oracle_verified_tasks": len(rows),
        "protocol": "One saved evaluation attempt per task; all levels and input splits pooled.",
        "caveat": "Sample sizes and level composition vary; this does not isolate length effects.",
        "sources": sources,
        "groups": groups,
    }
    (args.output / "success_by_actions.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(groups, rows, args.output)
    print(f"Verified {len(rows)} tasks: {successes}/{len(rows)} succeeded.")
    print(f"Plot and data saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
