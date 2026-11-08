"""Plot success against task-conditioned path entropy plus button-choice cost.

Run: integrations/seeclick/.venv/bin/python scripts/plot_combined_entropy.py
Uses saved analysis only: no new model evaluations or API calls.
The unweighted sum is a descriptive difficulty score, not joint Shannon entropy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from evaluate_path_entropy import rate_summary
from plot_success_by_actions import write_csv


def summarize(rows: list[dict], width: float = 5, minimum: int = 20) -> list[dict]:
    """Merge adjacent right-closed bins using counts, never success labels."""
    if not math.isfinite(width) or width <= 1e-10 or minimum < 1:
        raise ValueError("Bin width must exceed 1e-10 bits and minimum count must be positive")
    fixed = defaultdict(list)
    for row in rows:
        value = row["combined_score_bits"]
        if not math.isfinite(value) or value < 0:
            raise ValueError("Combined scores must be finite and nonnegative")
        index = max(1, math.ceil((value - 1e-10) / width))
        fixed[index].append(row)
    blocks, pending, lower, upper = [], [], None, None
    for index, members in sorted(fixed.items()):
        if lower is None:
            lower = (index - 1) * width
        upper = index * width
        pending.extend(members)
        if len(pending) >= minimum:
            blocks.append((lower, upper, pending))
            pending, lower = [], None
    if pending:
        if blocks:
            old_lower, _, old_members = blocks.pop()
            blocks.append((old_lower, upper, old_members + pending))
        else:
            blocks.append((lower, upper, pending))
    return [{
        "score_bin": f"{'[' if lo == 0 else '('}{lo:g}, {hi:g}]",
        "lower_bits": lo, "upper_bits": hi,
        "minimum_observed_score_bits": min(row["combined_score_bits"] for row in members),
        "maximum_observed_score_bits": max(row["combined_score_bits"] for row in members),
        **rate_summary(members),
        "meets_minimum": len(members) >= minimum,
        "mean_path_entropy_bits": sum(row["entropy_bits"] for row in members) / len(members),
        "mean_button_choice_bits": (
            sum(row["original_choice_bits"] for row in members) / len(members)
        ),
        **{f"L{level}_tasks": sum(row["level"] == level for row in members) for level in range(6)},
    } for lo, hi, members in blocks]


def plot(groups: list[dict], rows: list[dict], output: Path, width: float, minimum: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for index, group in enumerate(groups):
        rate = group["success_rate_percent"]
        lower, upper = group["wilson_95_lower_percent"], group["wilson_95_upper_percent"]
        ax.bar(index, rate, width=0.65, color="tab:blue")
        ax.errorbar(index, rate, yerr=[[max(0, rate - lower)], [max(0, upper - rate)]],
                    fmt="none", ecolor="0.15", capsize=5, linewidth=1.5)
        ax.text(index, upper + 3, f"{rate:.1f}%\n{group['successes']}/{group['tasks']}",
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(groups)), [
        group["score_bin"] + ("*" if not group["meets_minimum"] else "") for group in groups
    ])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Combined score: task-conditioned path entropy + button-choice cost (bits)")
    ax.set_ylabel("Task success rate")
    splits = " + ".join(sorted({row["split"] for row in rows}))
    levels = sorted({row["level"] for row in rows})
    level_label = "L0–L5" if levels == list(range(6)) else ", ".join(f"L{n}" for n in levels)
    successes = sum(row["success"] for row in rows)
    ax.set_title(
        "SeeClick: path entropy + button-choice cost\n"
        f"{splits}, {level_label} · {len(rows)} tasks · {successes}/{len(rows)} successful "
        f"({100 * successes / len(rows):.1f}%)"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    note = (
        "Button cost = Σ log₂(enabled buttons) along the correct original-task trace, "
        "including Done.\n"
        f"Adjacent {width:g}-bit bins merged toward at least {minimum} trials. "
        "Labels: successful / evaluated tasks.\n"
        "Whiskers: approximate 95% Wilson intervals; related templates limit precision.\n"
        "The sum is a descriptive difficulty score; the components can overlap."
    )
    if any(not group["meets_minimum"] for group in groups):
        note += f"\n* Fewer than {minimum} tasks available."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output / "success_by_combined_entropy.png", dpi=200)
    fig.savefig(output / "success_by_combined_entropy.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=Path("reports/seeclick-800-workload-entropy/analysis.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/seeclick-800-combined-entropy"))
    parser.add_argument("--bin-width", type=float, default=5)
    parser.add_argument("--min-tasks", type=int, default=20)
    args = parser.parse_args()
    source_bytes = args.analysis.read_bytes()
    source = json.loads(source_bytes)
    if source["prior"] != "observed-workload":
        raise ValueError("Expected path entropy conditioned on observed queue workload")
    fields = ("episode_id", "split", "level", "success", "entropy_bits", "original_choice_bits")
    rows = []
    for original in source["tasks_detail"]:
        row = {key: original[key] for key in fields}
        if row["success"] not in (0, 1):
            raise ValueError("Expected binary success labels")
        for key in ("entropy_bits", "original_choice_bits"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid {key} for {row['episode_id']}")
        counts = original["original_enabled_action_counts"]
        if not counts or any(not isinstance(count, int) or count < 1 for count in counts):
            raise ValueError(f"Invalid enabled-button counts for {row['episode_id']}")
        recomputed = math.fsum(math.log2(count) for count in counts)
        if not math.isclose(row["original_choice_bits"], recomputed, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Button-choice cost disagrees with trace for {row['episode_id']}")
        row["combined_score_bits"] = row["entropy_bits"] + row["original_choice_bits"]
        rows.append(row)
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("Expected one row per task, with at least one task")
    successes = sum(row["success"] for row in rows)
    if len(rows) != source["tasks"] or successes != source["successes"]:
        raise ValueError("Source totals disagree with outcomes")
    rows.sort(key=lambda row: (row["combined_score_bits"], row["episode_id"]))
    groups = summarize(rows, args.bin_width, args.min_tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "combined_score_per_task.csv", rows)
    write_csv(args.output / "success_by_combined_entropy.csv", groups)
    result = {
        "tasks": len(rows), "successes": successes,
        "minimum_combined_score_bits": min(row["combined_score_bits"] for row in rows),
        "maximum_combined_score_bits": max(row["combined_score_bits"] for row in rows),
        "formula": "combined_score_bits = entropy_bits + original_choice_bits",
        "weights": {"path_entropy": 1, "button_choice_cost": 1},
        "path_entropy_formula": source["formula"],
        "conditioning_scope": source["conditioning_scope"],
        "button_choice_definition": source["choice_cost_definition"],
        "interpretation": (
            "Unweighted additive difficulty score. The components may overlap; this is not "
            "joint Shannon entropy or a model success probability. No independence assumed."
        ),
        "protocol": "Same saved model outcomes; button counts from complete correct oracle traces.",
        "binning": {
            "initial_width_bits": args.bin_width, "minimum_tasks": args.min_tasks,
            "right_closed": True, "first_bin_includes_zero": True,
            "roundoff_tolerance_bits": 1e-10,
            "method": (
                "Accumulate adjacent fixed-width bins until minimum count, then merge an "
                "undersized final group backward. Binning ignores success labels."
            ),
        },
        "intervals": "Approximate 95% Wilson intervals; related templates limit independence.",
        "source_analysis": str(args.analysis),
        "source_analysis_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": groups, "tasks_detail": rows,
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(groups, rows, args.output, args.bin_width, args.min_tasks)
    print(f"Saved {len(rows)} tasks, {successes} successes to {args.output.resolve()}")
    for group in groups:
        print(f"{group['score_bin']}: {group['successes']}/{group['tasks']} "
              f"({group['success_rate_percent']:.1f}%)")


if __name__ == "__main__":
    main()
