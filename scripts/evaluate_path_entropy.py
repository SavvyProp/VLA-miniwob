"""Compute exact path entropy with observed queue sizes and saved evaluation outcomes.

Example (the SeeClick environment already includes Matplotlib):
    integrations/seeclick/.venv/bin/python scripts/evaluate_path_entropy.py \
        runs/seeclick-800-devicefix-dev runs/seeclick-800-devicefix-test

This is local oracle analysis, without model inference, browsers, or API calls.
The default fixes each queue's original size; --prior reference reproduces the
earlier metric that varies queue size from 1 to 5. Button-choice cost is exported
separately and is never added to the plotted entropy.
Plots use 0.5-bit intervals between 0 and 2 bits and one-bit intervals above 2.
Additional plots pool sparse neighboring entropy values and show entropy
alongside button-choice cost. Rates include approximate 95% Wilson intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from plot_entropy_choice import plot_joint, summarize_joint
from plot_success_by_actions import read_evaluations, write_csv
from script_entropy import PRIOR_VERSIONS, evaluate_entropy

from scriptwob.schema import Manifest


def entropy_bin(entropy: float, low_width: float = 0.5) -> int:
    """Separate zero, refine 0..2 bits, then use right-closed one-bit intervals."""
    if low_width not in {0.1, 0.2, 0.25, 0.5, 1.0}:
        raise ValueError("Unsupported lower-entropy bin width")
    if not math.isfinite(entropy) or entropy < -1e-10:
        raise ValueError("Entropy must be finite and nonnegative")
    if entropy <= 1e-10:
        return 0
    # The tolerance assigns roundoff just above a boundary to its intended
    # right-closed interval. It is in bits, independent of interval width.
    if entropy <= 2 + 1e-10:
        return math.ceil((entropy - 1e-10) / low_width)
    return round(2 / low_width) + math.ceil(entropy - 2 - 1e-10)


def rate_summary(rows: list[dict]) -> dict:
    """Descriptive binomial intervals; related task templates need not be iid."""
    n = len(rows)
    k = sum(row["success"] for row in rows)
    lower = upper = rate = None
    if n:
        rate = k / n
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        lower, upper = max(0, center - radius), min(1, center + radius)
    return {
        "tasks": n,
        "successes": k,
        "success_rate_percent": 100 * rate if n else None,
        "wilson_95_lower_percent": 100 * lower if n else None,
        "wilson_95_upper_percent": 100 * upper if n else None,
    }


def summarize(rows: list[dict], low_width: float = 0.5) -> list[dict]:
    groups = []
    fine_bins = round(2 / low_width)
    for index in range(max(entropy_bin(row["entropy_bits"], low_width) for row in rows) + 1):
        members = [row for row in rows if entropy_bin(row["entropy_bits"], low_width) == index]
        if index <= fine_bins:
            lower = round(max(0, index - 1) * low_width, 10)
            upper = round(index * low_width, 10)
        else:
            lower, upper = 1 + index - fine_bins, 2 + index - fine_bins
        groups.append({
            "entropy_bin": "0" if index == 0 else f"({lower:g}, {upper:g}]",
            "lower_bits": lower,
            "upper_bits": upper,
            **rate_summary(members),
            "mean_entropy_bits": (
                sum(row["entropy_bits"] for row in members) / len(members) if members else None
            ),
            "mean_required_actions": (
                sum(row["required_actions"] for row in members) / len(members)
                if members else None
            ),
            "mean_choice_bits": (
                sum(row["original_choice_bits"] for row in members) / len(members)
                if members else None
            ),
            **{
                f"L{level}_tasks": sum(row["level"] == level for row in members)
                for level in range(6)
            },
        })
    return groups


def summarize_supported(rows: list[dict], minimum: int = 20) -> list[dict]:
    """Pool adjacent entropy values without using outcomes or splitting ties.

    Zero remains its own category. The final undersized positive group joins
    its predecessor. If a category has too few total trials, report it honestly.
    """
    if minimum < 1:
        raise ValueError("Minimum group size must be positive")
    zero = [row for row in rows if row["entropy_bits"] <= 1e-10]
    tied = {}
    for row in rows:
        if row["entropy_bits"] > 1e-10:
            tied.setdefault(round(row["entropy_bits"], 10), []).append(row)
    positive, pending = [], []
    for _, members in sorted(tied.items()):
        pending.extend(members)
        if len(pending) >= minimum:
            positive.append(pending)
            pending = []
    if pending:
        if positive:
            positive[-1].extend(pending)
        else:
            positive.append(pending)
    groups = ([zero] if zero else []) + positive
    return [{
        "lower_bits": min(row["entropy_bits"] for row in members),
        "upper_bits": max(row["entropy_bits"] for row in members),
        **rate_summary(members),
        "meets_minimum": len(members) >= minimum,
    } for members in groups]


def plot_supported(groups: list[dict], output: Path, minimum: int = 20) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, group in enumerate(groups):
        rate = group["success_rate_percent"]
        ax.errorbar(
            rate, index,
            xerr=[[max(0, rate - group["wilson_95_lower_percent"])],
                  [max(0, group["wilson_95_upper_percent"] - rate)]],
            fmt="o", color="tab:blue", markersize=7, capsize=5,
        )
        ax.text(103, index, f"{rate:.1f}%  ({group['successes']}/{group['tasks']})",
                va="center", fontsize=10)
    labels = []
    for group in groups:
        lo, hi = group["lower_bits"], group["upper_bits"]
        label = f"{lo:.3g}" if abs(lo - hi) < 1e-10 else f"{lo:.3g}–{hi:.3g}"
        if not group["meets_minimum"]:
            label += f" (n < {minimum})"
        labels.append(label)
    ax.set_yticks(range(len(groups)), labels)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Task success rate (approximate 95% Wilson interval)")
    ax.set_ylabel("Observed entropy range (bits)")
    ax.set_title(f"SeeClick: success with more trials per entropy group\n"
                 f"Adjacent values pooled toward at least {minimum} tasks; zero kept separate")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.025,
             "Groups use entropy and sample counts only; identical entropy values stay together.\n"
             "Wide entropy ranges trade detail for more trials. "
             "Related task templates limit interval precision.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output / "success_by_entropy_pooled.png", dpi=200)
    fig.savefig(output / "success_by_entropy_pooled.pdf")
    plt.close(fig)


def plot(
    groups: list[dict], rows: list[dict], output: Path, prior: str, low_width: float = 0.5
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    successes = sum(row["success"] for row in rows)
    splits = " + ".join(sorted({row["split"] for row in rows}))
    levels = sorted({row["level"] for row in rows})
    level_label = "L0–L5" if levels == list(range(6)) else ", ".join(f"L{n}" for n in levels)
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    # Empty intervals remain in the exported table; avoid crowding the plot
    # with categories having no measured rate.
    populated = [group for group in groups if group["tasks"]]
    bars = ax.bar(
        range(len(populated)),
        [group["success_rate_percent"] for group in populated],
        width=0.65, color="tab:blue",
    )
    for bar, group in zip(bars, populated, strict=True):
        x, rate = bar.get_x() + bar.get_width() / 2, group["success_rate_percent"]
        lower, upper = group["wilson_95_lower_percent"], group["wilson_95_upper_percent"]
        ax.errorbar(x, rate, yerr=[[max(0, rate - lower)], [max(0, upper - rate)]],
                    fmt="none", ecolor="0.15", capsize=5, linewidth=1.5)
        ax.text(x, upper + 3, f"{rate:.1f}%\n{group['successes']}/{group['tasks']}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(populated)), [group["entropy_bin"] for group in populated])
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    conditioned = prior == "observed-workload"
    ax.set_xlabel(
        "Execution-path entropy (bits; each queue size fixed to its original value)"
        if conditioned else "Execution-path entropy (bits; reference distribution of states)"
    )
    ax.set_ylabel("Task success rate")
    ax.set_title(
        f"SeeClick: {'queue-length-conditioned ' if conditioned else ''}path entropy and success\n"
        f"{splits}, {level_label} · {len(rows)} tasks · {successes}/{len(rows)} successful "
        f"({100 * successes / len(rows):.1f}%)"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5, 0.025,
        f"Bins: zero separately; {low_width:g} bit from 0 to 2; 1 bit above 2. "
        "Empty bins omitted.\n"
        "Labels: rate and successful / evaluated tasks. "
        "Whiskers: approximate 95% Wilson intervals; "
        "related templates limit precision.",
        ha="center", fontsize=9, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output / "success_by_entropy.png", dpi=200)
    fig.savefig(output / "success_by_entropy.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--prior", choices=tuple(PRIOR_VERSIONS), default="observed-workload",
        help="Fix actual queue sizes (default), or vary sizes 1..5 with the earlier prior",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--low-entropy-bin-width", type=float, choices=(0.1, 0.2, 0.25, 0.5, 1.0), default=0.5,
        help="Bin width between 0 and 2 bits; zero is separate and bins above 2 are one bit wide",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(
            "reports/seeclick-800-workload-entropy" if args.prior == "observed-workload"
            else "reports/seeclick-800-entropy"
        )
    outcomes, sources = read_evaluations(args.runs)
    episodes = {}
    for run in args.runs:
        manifest = Manifest.model_validate_json((run / "manifest.json").read_text())
        episodes.update({episode.id: episode for episode in manifest.episodes})
    rows = []
    distributions = []
    for index, outcome in enumerate(outcomes, start=1):
        episode = episodes[outcome["episode_id"]]
        summary, paths = evaluate_entropy(episode, prior=args.prior)
        if summary["original_required_actions"] != outcome["required_actions"]:
            raise ValueError(f"Original reference length changed: {episode.id}")
        row = {
            **outcome,
            **summary,
            "program_size": episode.complexity.program_size,
            "control_depth": episode.complexity.control_depth,
            "loop_depth": episode.complexity.loop_depth,
            "predicate_complexity": episode.complexity.predicate_complexity,
        }
        rows.append(row)
        distributions.append({"episode_id": episode.id, "paths": paths})
        if index % 25 == 0 or index == len(outcomes):
            print(f"Computed exact path entropy for {index}/{len(outcomes)} tasks", flush=True)

    rows.sort(key=lambda row: (row["entropy_bits"], row["episode_id"]))
    groups = summarize(rows, args.low_entropy_bin_width)
    supported = summarize_supported(rows)
    joint = summarize_joint(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "metrics_per_task.csv", [
        {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
         for key, value in row.items()}
        for row in rows
    ])
    write_csv(args.output / "success_by_entropy.csv", groups)
    write_csv(args.output / "success_by_entropy_pooled.csv", supported)
    write_csv(args.output / "entropy_vs_button_choice.csv", joint)
    with (args.output / "path_distributions.jsonl").open("w") as stream:
        for distribution in distributions:
            stream.write(json.dumps(distribution, separators=(",", ":")) + "\n")

    repo = Path(__file__).resolve().parents[1]
    analysis = {
        "method": (
            f"Exact enumeration under {PRIOR_VERSIONS[args.prior]}, conditioned on solvability"
        ),
        "prior": args.prior,
        "formula": (
            "H(B | fixed script, observed queue lengths/cursors, solvable reference state)"
            if args.prior == "observed-workload" else
            "H(B | fixed script, solvable reference state)"
        ) + " = -sum_b p(b) log2 p(b)",
        "conditioning_scope": (
            "Queue sizes, cursors and existing record metadata fixed; remaining relevant item "
            "category/status vary. Counter prior remains 0..3; toggles and status widgets vary."
            if args.prior == "observed-workload" else
            "Queue sizes uniform 1..5; independent initial-state prior as in the earlier analysis."
        ),
        "choice_cost_definition": (
            "sum_t log2(number of enabled semantic buttons before correct action t), including "
            "Done; evaluated on the original oracle trace; reported separately from entropy"
        ),
        "analysis_action_cap": 1000,
        "tasks": len(rows),
        "successes": sum(row["success"] for row in rows),
        "minimum_entropy_bits": min(row["entropy_bits"] for row in rows),
        "maximum_entropy_bits": max(row["entropy_bits"] for row in rows),
        "sources": sources,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in (
                "scripts/evaluate_path_entropy.py", "scripts/script_entropy.py",
                "scripts/plot_entropy_choice.py",
                "scripts/plot_success_by_actions.py", "src/scriptwob/engine.py",
                "src/scriptwob/schema.py", "src/scriptwob/generation.py",
            )
        },
        "binning": {
            "zero_separate": True,
            "lower_range_bits": [0, 2],
            "lower_bin_width_bits": args.low_entropy_bin_width,
            "upper_bin_width_bits": 1,
            "right_closed": True,
            "roundoff_tolerance_bits": 1e-10,
            "empty_bins": "Retained in table with null rates; omitted from plot",
        },
        "outcome_definition": (
            "One original saved SeeClick attempt per task; no new policy rollouts"
        ),
        "intervals": {
            "method": "Wilson score", "confidence": 0.95,
            "caveat": "Approximate binomial intervals; related templates need not be independent",
        },
        "pooled_binning": {
            "minimum_tasks": 20, "zero_separate": True, "tie_rounding_decimals": 10,
            "method": "Pool adjacent positive entropy values; merge undersized tail backward",
            "uses_success_labels": False,
        },
        "joint_aggregation": {
            "coordinates": ["entropy_bits", "original_choice_bits"],
            "rounding_decimals": 6, "bubble_area": "tasks", "bubble_color": "success_rate_percent",
        },
        "groups": groups,
        "pooled_groups": supported,
        "joint_groups": joint,
        "tasks_detail": rows,
    }
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    plot(groups, rows, args.output, args.prior, args.low_entropy_bin_width)
    plot_supported(supported, args.output)
    plot_joint(joint, args.output, conditioned=args.prior == "observed-workload")
    print(f"Saved per-task metrics, exact path distributions, and plots to {args.output.resolve()}")


if __name__ == "__main__":
    main()
