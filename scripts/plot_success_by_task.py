"""Plot verified, pooled evaluation success by script level and task family.

Example (no model inference or API calls):
    integrations/seeclick/.venv/bin/python scripts/plot_success_by_task.py \
        runs/seeclick-800-code-dev runs/seeclick-800-code-test \
        --output reports/seeclick-800-code-combined --title "SeeClick: code-like instructions"
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from plot_success_by_actions import read_evaluations, write_csv

from scriptwob.generation import FAMILIES

LEVELS = (
    "Atomic action",
    "Sequence",
    "Conditional",
    "Repetition and\nstopping",
    "Branches\nwith loops",
    "Nested\ncontrol flow",
)


def aggregate(rows: list[dict], *, by_family: bool = False) -> list[dict]:
    groups = []
    for level in range(6):
        families = FAMILIES[level] if by_family else {None: LEVELS[level].replace("\n", " ")}
        for family, description in families.items():
            selected = [
                row for row in rows
                if row["level"] == level and (family is None or row["family"] == family)
            ]
            if not selected:
                continue
            successes = sum(row["success"] for row in selected)
            group = {
                "level": level,
                **({"family": family} if by_family else {}),
                "description": description,
                "successes": successes,
                "tasks": len(selected),
                "success_rate_percent": 100 * successes / len(selected),
            }
            for split in ("dev", "test"):
                group[f"{split}_tasks"] = sum(row["split"] == split for row in selected)
                group[f"{split}_successes"] = sum(
                    row["success"] for row in selected if row["split"] == split
                )
            groups.append(group)
    if sum(group["tasks"] for group in groups) != len(rows):
        raise ValueError("Some tasks are missing from the level/family catalog")
    return groups


def plot(levels: list[dict], families: list[dict], result: dict, output: Path, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    subtitle = (
        f"{' + '.join(result['splits'])} combined · {result['successes']}/{result['tasks']} tasks"
        f" succeeded ({result['success_rate_percent']:.1f}%)"
    )
    fig, ax = plt.subplots(figsize=(10.5, 6))
    labels = [f"L{row['level']}\n{LEVELS[row['level']]}" for row in levels]
    bars = ax.bar(labels, [row["success_rate_percent"] for row in levels], color="tab:blue",
                  width=0.65)
    ax.bar_label(bars, labels=[
        f"{row['success_rate_percent']:.1f}%\n({row['successes']}/{row['tasks']})" for row in levels
    ], padding=5, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("Whole-task success rate")
    ax.set_xlabel("Script complexity level", labelpad=10)
    ax.set_title(f"{title}\n{subtitle}", pad=14)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.025, "One attempt per task. Labels show successes / evaluated tasks.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"success_by_level.{suffix}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    labels = [f"L{row['level']}  {row['family'].replace('_', ' ')}" for row in families]
    rates = [row["success_rate_percent"] for row in families]
    bars = ax.barh(labels, rates, color="tab:blue", height=0.65)
    ax.bar_label(bars, labels=[
        f"{row['success_rate_percent']:.1f}% ({row['successes']}/{row['tasks']})"
        for row in families
    ], padding=5, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Whole-task success rate")
    ax.set_ylabel("Task family")
    ax.set_title(f"{title}: task families\n{subtitle}", pad=14)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.025,
             "One attempt per task. Labels show successes / evaluated tasks; family sizes vary.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"success_by_family.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="SeeClick")
    args = parser.parse_args()
    rows, sources = read_evaluations(args.runs)
    levels = aggregate(rows)
    families = aggregate(rows, by_family=True)
    successes = sum(row["success"] for row in rows)
    result = {
        "tasks": len(rows), "successes": successes,
        "success_rate_percent": 100 * successes / len(rows),
        "splits": sorted({row["split"] for row in rows}),
        "protocol": "First-attempt successes divided by evaluated tasks, pooling input splits. "
                    "Success requires completing the entire script and clicking Done. "
                    "Input runs must be complete with matching checkpoint/task/prompt settings.",
        "sources": sources, "levels": levels, "families": families,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tasks.csv", rows)
    write_csv(args.output / "success_by_level.csv", levels)
    write_csv(args.output / "success_by_family.csv", families)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(levels, families, result, args.output, args.title)
    print(f"Verified {successes}/{len(rows)} successes ({result['success_rate_percent']:.1f}%).")
    print(f"Plots and data: {args.output.resolve()}")


if __name__ == "__main__":
    main()
