"""Evaluation metrics and collection coverage are deliberately separate."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tasks import FAMILIES, read_manifest


def cell(instance: dict) -> str:
    return f"{instance['task']}|{instance['order']}|{instance['wording']}"


def report(root: Path, bootstrap_samples: int = 2000) -> dict:
    run = json.loads((root / "run.json").read_text())
    manifest = read_manifest(root / "manifest.json")
    results = [
        json.loads(p.read_text()) for p in sorted((root / "episodes").glob("*/private.json"))
    ]
    usable = [r for r in results if not r["infrastructure_failure"]]
    by_cell = defaultdict(list)
    planned = defaultdict(set)
    for instance in manifest.instances:
        if instance.split == run["split"]:
            planned[cell(instance.model_dump())].add(instance.id)
    for row in results:
        by_cell[cell(row["instance"])].append(row)
    coverage = {}
    for key, ids in planned.items():
        rows = by_cell[key]
        accepted = {r["instance"]["id"] for r in rows if r["verdict"]["success"]}
        coverage[key] = {
            "planned_instances": len(ids),
            "attempts": len(rows),
            "attempted_instances": len({r["instance"]["id"] for r in rows}),
            "accepted_instances": len(accepted),
            "shortfall": len(ids - accepted),
            "accepted_transitions": sum(
                r["primitive_transitions"] for r in rows if r["verdict"]["success"]
            ),
        }
    output = {
        "mode": run["mode"],
        "manifest_digest": manifest.digest,
        "attempts": len(results),
        "infrastructure_failures": len(results) - len(usable),
        "accepted_episodes": sum(r["verdict"]["success"] for r in usable),
        "primitive_transitions": sum(r["primitive_transitions"] for r in results),
        "prediction_horizon": run["prediction_horizon"],
        "execution_prefix": run["execution_prefix"],
        "coverage": coverage,
    }
    if run["mode"] == "collect":
        output["attempt_acceptance_rate"] = (
            output["accepted_episodes"] / len(usable) if usable else None
        )
        return output
    # Equal weights for declared order/wording cells, then equal task weights.
    # Paired order/wording rows with the same values/layout seed form one cluster.
    tasks = defaultdict(list)
    for row in usable:
        tasks[row["instance"]["task"]].append(row)

    def mean_success(rows):
        cells = defaultdict(list)
        for row in rows:
            cells[cell(row["instance"])].append(float(row["verdict"]["success"]))
        return float(np.mean([np.mean(v) for v in cells.values()]))

    per_task = {}
    for task, rows in tasks.items():
        per_task[task] = {
            "episodes": len(rows),
            "success": mean_success(rows),
            "final_state_success": float(np.mean([r["verdict"]["goals_valid"] for r in rows])),
            "order_violation_rate": float(np.mean([not r["verdict"]["order_valid"] for r in rows])),
            "mean_primitive_horizon": float(np.mean([r["primitive_transitions"] for r in rows])),
            "mean_first_violation_step": (
                float(
                    np.mean(
                        [
                            r["verdict"]["first_violation"]["step"]
                            for r in rows
                            if r["verdict"].get("first_violation")
                        ]
                    )
                )
                if any(r["verdict"].get("first_violation") for r in rows)
                else None
            ),
        }
    output["per_task"] = per_task
    output["complexity"] = {
        str(length): float(
            np.mean([v["success"] for task, v in per_task.items() if len(FAMILIES[task]) == length])
        )
        for length in sorted({len(FAMILIES[task]) for task in per_task})
    }
    complete = all(
        {r["instance"]["id"] for r in by_cell[key] if not r["infrastructure_failure"]} == ids
        for key, ids in planned.items()
    )
    output["complete_evaluation"] = complete
    output["observed_macro_success"] = (
        float(np.mean([r["success"] for r in per_task.values()])) if per_task else None
    )
    # Partial/missing cells must not be advertised as a complete benchmark score.
    output["benchmark_macro_success"] = output["observed_macro_success"] if complete else None
    output["bootstrap_95_interval"] = None
    if tasks and bootstrap_samples > 0:
        clusters = {}
        for task, rows in tasks.items():
            groups = defaultdict(list)
            for row in rows:
                i = row["instance"]
                groups[(i["values_seed"], i["layout_seed"])].append(row)
            clusters[task] = list(groups.values())
        # At least two independent UI clusters per task are needed for a useful interval.
        if all(len(groups) >= 2 for groups in clusters.values()):
            rng = np.random.default_rng(2026)
            estimates = []
            names = list(clusters)
            for _ in range(bootstrap_samples):
                scores = []
                for task in rng.choice(names, len(names), replace=True):
                    groups = clusters[task]
                    sampled = [
                        row
                        for idx in rng.integers(len(groups), size=len(groups))
                        for row in groups[idx]
                    ]
                    scores.append(mean_success(sampled))
                estimates.append(np.mean(scores))
            output["bootstrap_95_interval"] = np.quantile(estimates, [0.025, 0.975]).tolist()
    output["uncertainty_method"] = (
        "hierarchical task/UI-cluster bootstrap; paired variants stay together"
    )
    return output
