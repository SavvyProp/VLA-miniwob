"""Plot task-conditioned path entropy alone from an existing entropy analysis.

Run with integrations/seeclick/.venv/bin/python scripts/plot_task_conditioned_entropy.py.
Uses saved outcomes and entropy_bits; button-choice cost is never read or subtracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from evaluate_path_entropy import entropy_bin, rate_summary
from plot_success_by_actions import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path,
        default=Path("reports/seeclick-800-workload-entropy/analysis.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/seeclick-800-task-conditioned-entropy"),
    )
    args = parser.parse_args()
    source_bytes = args.analysis.read_bytes()
    source = json.loads(source_bytes)
    if source["prior"] != "observed-workload":
        raise ValueError("Expected entropy conditioned on the observed queue workload")
    fields = ("episode_id", "split", "level", "entropy_bits", "success")
    rows = [{key: row[key] for key in fields} for row in source["tasks_detail"]]
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty outcomes with one row per task")
    if any(row["success"] not in (0, 1) for row in rows):
        raise ValueError("Success labels must be binary")
    successes = sum(row["success"] for row in rows)
    if len(rows) != source["tasks"] or successes != source["successes"]:
        raise ValueError("Saved totals disagree with per-task outcomes")

    members = defaultdict(list)
    for row in rows:
        members[entropy_bin(row["entropy_bits"], 0.5)].append(row)
    groups = []
    for index in range(max(members) + 1):
        lower = max(0, index - 1) * 0.5 if index <= 4 else index - 3
        upper = index * 0.5 if index <= 4 else index - 2
        groups.append({
            "entropy_bin": "0" if index == 0 else f"({lower:g}, {upper:g}]",
            "lower_bits": lower, "upper_bits": upper,
            **rate_summary(members[index]),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "entropy_per_task.csv", rows)
    write_csv(args.output / "success_by_entropy.csv", groups)
    report = {
        "tasks": len(rows), "successes": successes,
        "metric": "entropy_bits", "button_choice_cost_included": False,
        "formula": source["formula"], "conditioning_scope": source["conditioning_scope"],
        "interpretation": (
            "Script and observed queue sizes/cursors are fixed; relevant state values vary. "
            "This is not conditioning on the full deterministic initial page."
        ),
        "note": (
            "Button-choice cost was already separate in the source calculation. "
            "Use path entropy directly, without subtraction. No new model evaluations."
        ),
        "binning": {
            "zero_separate": True, "width_below_two_bits": 0.5,
            "width_above_two_bits": 1, "right_closed": True,
            "empty_bins": "Retained with null rates in tables; omitted from plot",
        },
        "intervals": (
            "Approximate 95% Wilson intervals; shared templates limit independence assumptions."
        ),
        "source_analysis": str(args.analysis),
        "source_analysis_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": groups, "tasks_detail": rows,
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    populated = [group for group in groups if group["tasks"]]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    for index, group in enumerate(populated):
        rate = group["success_rate_percent"]
        lower, upper = group["wilson_95_lower_percent"], group["wilson_95_upper_percent"]
        ax.bar(index, rate, width=0.65, color="tab:blue")
        ax.errorbar(index, rate, yerr=[[max(0, rate - lower)], [max(0, upper - rate)]],
                    fmt="none", ecolor="0.15", capsize=5, linewidth=1.5)
        ax.text(index, upper + 3, f"{rate:.1f}%\n{group['successes']}/{group['tasks']}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(populated)), [group["entropy_bin"] for group in populated])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Task-conditioned execution-path entropy (bits)")
    ax.set_ylabel("Task success rate")
    splits = " + ".join(sorted({row["split"] for row in rows}))
    levels = sorted({row["level"] for row in rows})
    level_label = "L0–L5" if levels == list(range(6)) else ", ".join(f"L{n}" for n in levels)
    ax.set_title(
        "SeeClick: entropy conditioned on actual task\n"
        f"{splits}, {level_label} · {len(rows)} tasks · {successes}/{len(rows)} successful "
        f"({100 * successes / len(rows):.1f}%)"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5, 0.025,
        "Fixed script and actual queue sizes/cursors; branch-relevant values still vary. "
        "Button-choice cost excluded.\n"
        "Bins: zero separately; 0.5 bit up to 2; 1 bit above 2. Empty bins omitted.\n"
        "Labels: successful / evaluated tasks. Whiskers: approximate 95% Wilson intervals; "
        "related templates limit precision.",
        ha="center", fontsize=9, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(args.output / "success_by_entropy.png", dpi=200)
    fig.savefig(args.output / "success_by_entropy.pdf")
    plt.close(fig)
    print(f"Saved entropy-only plot and data: {args.output.resolve()}")
    print(f"{successes}/{len(rows)} successes; button-choice cost excluded.")


if __name__ == "__main__":
    main()
