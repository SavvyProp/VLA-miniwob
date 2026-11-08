"""Plot saved evaluation success against task instruction length in characters.

Example:
    integrations/seeclick/.venv/bin/python scripts/plot_success_by_prompt_length.py \
        runs/seeclick-800-devicefix-dev runs/seeclick-800-devicefix-test

Uses original saved outcomes only. No model inference, browser, or API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from evaluate_path_entropy import rate_summary
from plot_success_by_actions import read_evaluations, write_csv

from scriptwob.schema import Manifest


def read_prompts(runs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows, sources = read_evaluations(runs)
    instructions = {}
    for run in runs:
        manifest = Manifest.model_validate_json((run / "manifest.json").read_text())
        instructions.update({episode.id: episode.instruction for episode in manifest.episodes})
    for row in rows:
        saved_path = Path(row["attempt_path"]) / "episode.json"
        saved = json.loads(saved_path.read_text())
        instruction = saved["instruction"]
        if instruction != instructions[row["episode_id"]]:
            raise ValueError(f"Saved instruction differs from manifest: {saved_path}")
        row.update({
            "instruction": instruction,
            "prompt_chars": len(instruction),
            "episode_json_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        })
    return sorted(rows, key=lambda row: (row["prompt_chars"], row["episode_id"])), sources


def summarize(rows: list[dict], width: int = 50, minimum: int = 20) -> list[dict]:
    """Merge adjacent fixed-width bins using counts only, including the sparse tail."""
    if width < 1 or minimum < 1:
        raise ValueError("Bin width and minimum group size must be positive")
    fixed = defaultdict(list)
    for row in rows:
        fixed[row["prompt_chars"] // width].append(row)
    blocks, pending, lower, upper = [], [], None, None
    for index, members in sorted(fixed.items()):
        if lower is None:
            lower = index * width
        upper = (index + 1) * width
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
        "prompt_chars_bin": f"{lower}–{upper - 1}",
        "lower_chars_inclusive": lower,
        "upper_chars_exclusive": upper,
        "minimum_observed_chars": min(row["prompt_chars"] for row in members),
        "maximum_observed_chars": max(row["prompt_chars"] for row in members),
        **rate_summary(members),
        "meets_minimum": len(members) >= minimum,
        **{f"L{level}_tasks": sum(row["level"] == level for row in members) for level in range(6)},
    } for lower, upper, members in blocks]


def plot(groups: list[dict], rows: list[dict], output: Path, width: int, minimum: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(11, 6.2))
    rates = [group["success_rate_percent"] for group in groups]
    bars = ax.bar(range(len(groups)), rates, color="tab:blue", width=0.65)
    for bar, group in zip(bars, groups, strict=True):
        center = bar.get_x() + bar.get_width() / 2
        rate = group["success_rate_percent"]
        lower, upper = group["wilson_95_lower_percent"], group["wilson_95_upper_percent"]
        ax.errorbar(center, rate, yerr=[[max(0, rate - lower)], [max(0, upper - rate)]],
                    fmt="none", ecolor="0.15", capsize=5, linewidth=1.5)
        ax.text(center, upper + 3, f"{rate:.1f}%\n{group['successes']}/{group['tasks']}",
                ha="center", va="bottom", fontsize=10)
    labels = [group["prompt_chars_bin"] + ("*" if not group["meets_minimum"] else "")
              for group in groups]
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylim(0, 116)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Task prompt length (characters, including whitespace and punctuation)")
    ax.set_ylabel("Task success rate")
    splits = " + ".join(sorted({row["split"] for row in rows}))
    levels = sorted({row["level"] for row in rows})
    level_label = "L0–L5" if levels == list(range(6)) else ", ".join(f"L{n}" for n in levels)
    successes = sum(row["success"] for row in rows)
    ax.set_title(f"SeeClick: prompt length and success\n"
                 f"{splits}, {level_label} · {len(rows)} tasks · "
                 f"{successes}/{len(rows)} successful "
                 f"({100 * successes / len(rows):.1f}%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    footnote = (
        f"Adjacent {width}-character bins merged toward at least {minimum} tasks per group. "
        "Labels: successful / evaluated tasks.\n"
        "Whiskers: approximate 95% Wilson intervals. "
        "Length excludes model boilerplate and history.\n"
        "Related task templates limit interval precision; length and task complexity vary together."
    )
    if any(not group["meets_minimum"] for group in groups):
        footnote += f"\n* Fewer than {minimum} tasks available."
    fig.text(0.5, 0.025, footnote, ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output / "success_by_prompt_length.png", dpi=200)
    fig.savefig(output / "success_by_prompt_length.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/seeclick-800-prompt-length"))
    parser.add_argument("--bin-width", type=int, default=50)
    parser.add_argument("--min-tasks", type=int, default=20)
    args = parser.parse_args()
    if args.bin_width < 1 or args.min_tasks < 1:
        parser.error("--bin-width and --min-tasks must be positive")
    rows, sources = read_prompts(args.runs)
    groups = summarize(rows, args.bin_width, args.min_tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "prompt_length_per_task.csv", rows)
    write_csv(args.output / "success_by_prompt_length.csv", groups)
    repo = Path(__file__).resolve().parents[1]
    result = {
        "tasks": len(rows),
        "successes": sum(row["success"] for row in rows),
        "minimum_prompt_chars": min(row["prompt_chars"] for row in rows),
        "maximum_prompt_chars": max(row["prompt_chars"] for row in rows),
        "length_definition": (
            "Python len of the unmodified saved episode instruction: Unicode code points, "
            "including whitespace and punctuation. Verified against the saved manifest. "
            "Excludes system/response instructions, image markup and action history; "
            "not token count."
        ),
        "protocol": (
            "One original saved attempt per task; input evaluation splits and levels pooled."
        ),
        "binning": {
            "initial_width_chars": args.bin_width, "minimum_tasks": args.min_tasks,
            "left_closed_right_open": True,
            "method": (
                "From shortest to longest, accumulate complete fixed-width bins until minimum "
                "count; merge undersized final group backward. Empty bins have no rate. "
                "Never split equal lengths. Grouping ignores success labels."
            ),
        },
        "intervals": "Approximate 95% Wilson score intervals using a binomial reference model.",
        "caveat": (
            "Related templates and heterogeneous difficulties limit binomial assumptions. "
            "This descriptive association does not isolate a causal effect of prompt length."
        ),
        "sources": sources,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ("scripts/plot_success_by_prompt_length.py",
                         "scripts/plot_success_by_actions.py", "scripts/evaluate_path_entropy.py")
        },
        "groups": groups,
        "tasks_detail": rows,
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(groups, rows, args.output, args.bin_width, args.min_tasks)
    print(f"Verified {len(rows)} prompts: {result['successes']}/{len(rows)} tasks succeeded.")
    print(f"Plot and data saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
