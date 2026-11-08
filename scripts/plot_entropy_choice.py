"""Plot saved episode outcomes against path entropy and button-choice cost."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def summarize_joint(rows: list[dict]) -> list[dict]:
    """Aggregate metric pairs rounded to six decimals without outcome-based bins."""
    groups: dict[tuple[float, float], dict] = {}
    for row in rows:
        pair = (round(row["entropy_bits"], 6), round(row["original_choice_bits"], 6))
        group = groups.setdefault(pair, {
            "entropy_bits": pair[0],
            "choice_bits": pair[1],
            "tasks": 0,
            "successes": 0,
        })
        group["tasks"] += 1
        group["successes"] += int(row["success"])
    return [
        {**groups[pair], "success_rate_percent": (
            100 * groups[pair]["successes"] / groups[pair]["tasks"]
        )}
        for pair in sorted(groups)
    ]


def plot_joint(groups: list[dict], output: Path, conditioned: bool) -> None:
    """Show both difficulty metrics while retaining repeated-pair sample counts."""
    if not groups:
        raise ValueError("The joint plot requires at least one metric pair")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    tasks = sum(group["tasks"] for group in groups)
    successes = sum(group["successes"] for group in groups)
    area_per_task = 18
    fig, ax = plt.subplots(figsize=(10.8, 7.3))
    bubbles = ax.scatter(
        [group["entropy_bits"] for group in groups],
        [group["choice_bits"] for group in groups],
        s=[area_per_task * group["tasks"] for group in groups],
        c=[group["success_rate_percent"] for group in groups],
        cmap="viridis", vmin=0, vmax=100, edgecolors="0.3", linewidths=0.5,
        zorder=3,
    )
    colorbar = fig.colorbar(bubbles, ax=ax, pad=0.025, fraction=0.045)
    colorbar.set_label("Observed success rate at each metric pair")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    size_handles = [
        ax.scatter(
            [], [], s=area_per_task * count, facecolor="0.75", edgecolor="0.3",
            linewidth=0.5, label=str(count),
        )
        for count in (1, 5, 20)
    ]
    ax.legend(
        handles=size_handles, title="Tasks per bubble", loc="upper right",
        labelspacing=1.0, borderpad=0.8, framealpha=0.95,
    )
    ax.set_xlabel(
        "Queue-length-conditioned path entropy (bits)" if conditioned else
        "Path entropy under the reference state distribution (bits)"
    )
    ax.set_ylabel("Button-choice cost on the original task (bits)")
    ax.set_title(
        "SeeClick: script branching and button-choice difficulty\n"
        f"{tasks} evaluated tasks · {successes}/{tasks} successful "
        f"({100 * successes / tasks:.1f}%)"
    )
    ax.margins(x=0.06, y=0.08)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)
    fig.text(
        0.5, 0.026,
        "Each bubble groups identical metric pairs rounded to 6 decimals; no jitter. "
        "Area is proportional to task count.\n"
        "Color summarizes original saved outcomes. Small bubbles have uncertain success rates.\n"
        "Button-choice cost = sum of log₂(enabled buttons) along the correct original-task trace.",
        ha="center", fontsize=9, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "entropy_vs_button_choice.png", dpi=200)
    fig.savefig(output / "entropy_vs_button_choice.pdf")
    plt.close(fig)
