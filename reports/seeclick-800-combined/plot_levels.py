"""Reproduce the pooled dev/test figure using the existing SeeClick runtime.

From the repository root:
    integrations/seeclick/.venv/bin/python reports/seeclick-800-combined/plot_levels.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

# Keep Matplotlib's cache out of the user's potentially read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
SOURCES = {
    split: ROOT / "runs" / f"seeclick-800-devicefix-{split}" / "report.json"
    for split in ("dev", "test")
}
LEVELS = (
    ("Atomic action", "Perform one basic operation, such as pressing a button."),
    ("Sequence", "Perform several actions in a specified order."),
    ("Conditional", "Choose actions using a visible if/else condition."),
    ("Repetition and stopping", "Repeat a fixed count, while / until, or for each item."),
    ("Branches with loops", "Combine conditional decisions with repetition."),
    ("Nested control flow", "Execute loops within loops, sometimes with branches."),
)


def main() -> None:
    reports = {split: json.loads(path.read_text()) for split, path in SOURCES.items()}
    configs = [
        json.loads(path.with_name("run.json").read_text())["config"]
        for path in SOURCES.values()
    ]
    identities = [{k: v for k, v in config.items() if k != "split"} for config in configs]
    if identities[0] != identities[1]:
        raise ValueError("The evaluation runs do not have matching model/task settings")
    for split, report in reports.items():
        if report["stop_reason"] != "complete":
            raise ValueError(f"The {split} evaluation is incomplete")
        if report["attempted_episodes"] != report["selected_episodes"]:
            raise ValueError(f"The {split} evaluation has unattempted tasks")

    rows = []
    for level, (title, description) in enumerate(LEVELS):
        row = {"level": level, "title": title, "description": description}
        for split, report in reports.items():
            metrics = report["levels"][str(level)]["first_attempt"]
            row[f"{split}_successes"] = metrics["successes"]
            row[f"{split}_tasks"] = metrics["denominator_all_attempted"]
        row["successes"] = row["dev_successes"] + row["test_successes"]
        row["tasks"] = row["dev_tasks"] + row["test_tasks"]
        row["success_rate_percent"] = 100 * row["successes"] / row["tasks"]
        rows.append(row)

    successes = sum(row["successes"] for row in rows)
    tasks = sum(row["tasks"] for row in rows)
    overall = 100 * successes / tasks
    with (OUTPUT / "combined_levels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "aggregation": "First-attempt successes summed across dev/test, divided by tasks summed",
        "sources": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for split, path in SOURCES.items()
        },
        "overall": {"successes": successes, "tasks": tasks, "success_rate_percent": overall},
        "levels": rows,
    }
    (OUTPUT / "combined_levels.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#172538",
        "axes.labelcolor": "#425268",
        "xtick.color": "#425268",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.4, 8.4), facecolor="white")
    label_ax = fig.add_axes((0.045, 0.20, 0.365, 0.58))
    ax = fig.add_axes((0.43, 0.20, 0.43, 0.58))
    for axes in (label_ax, ax):
        axes.set_ylim(5.6, -0.6)
    label_ax.set_xlim(0, 1)
    label_ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Whole-task success rate", labelpad=12)
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E1E7EF", linewidth=0.8)
    ax.tick_params(axis="x", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in rows:
        y = row["level"]
        label_ax.text(0, y - 0.13, f"L{y}   {row['title']}", fontsize=13.2,
                      weight="bold", va="center")
        label_ax.text(0, y + 0.19, row["description"], fontsize=10.5,
                      color="#566579", va="center")
        ax.barh(y, 100, height=0.47, color="#F1F4F8", zorder=1)
        ax.barh(y, row["success_rate_percent"], height=0.47, color="#2868A5", zorder=2)
        ax.text(1.035, y - 0.10, f"{row['success_rate_percent']:.1f}%",
                transform=ax.get_yaxis_transform(), va="center", weight="bold", fontsize=14)
        ax.text(1.035, y + 0.18, f"{row['successes']} / {row['tasks']} passed",
                transform=ax.get_yaxis_transform(), va="center", color="#566579", fontsize=10.5)

    fig.text(0.045, 0.927, "SeeClick success by script complexity", fontsize=24, weight="bold")
    fig.text(0.045, 0.88,
             f"Combined dev + test  |  {tasks} tasks  |  {successes} passed  |  "
             f"{overall:.1f}% overall success", fontsize=13, color="#425268")
    fig.text(0.045, 0.817, "SCRIPT LEVEL", fontsize=10, weight="bold", color="#65748A")
    fig.text(0.43, 0.817, "FIRST-ATTEMPT SUCCESS", fontsize=10, weight="bold", color="#65748A")
    fig.text(0.875, 0.817, "COMBINED RESULT", fontsize=10, weight="bold", color="#65748A")
    fig.text(0.045, 0.087,
             "Pooled counts across splits; one attempt per task. "
             "Success requires the complete correct action sequence and Done.",
             fontsize=10.3, color="#566579")
    fig.text(0.045, 0.052,
             "Levels describe control structure. Higher levels can require fewer actions "
             "when a loop executes zero times.", fontsize=10.3, color="#566579")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"combined_levels.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(OUTPUT), **summary["overall"]}))


if __name__ == "__main__":
    main()
