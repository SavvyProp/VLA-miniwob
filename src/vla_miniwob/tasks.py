"""Frozen task families, paired orders, wording, and disjoint instance splits."""

import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import StrictModel

MINIWOB_REVISION = "33c3b4ddef8c6eb67c57a29663d844b1eda7e614"
COMPWOB_REVISION = "83f1e7b59e762648c953de751d31f75720683622"
TASK_REVISION = "independent-panels-v1"
COMPONENTS = (
    "click-button",
    "click-link",
    "click-dialog",
    "click-checkboxes",
    "click-option",
    "click-widget",
    "click-button-sequence",
)
FAMILIES = {name: (name,) for name in COMPONENTS}
for parts in [
    ("click-button", "click-link"),
    ("click-link", "click-button", "click-dialog"),
    ("click-link", "click-button", "click-checkboxes", "click-dialog"),
    ("click-link", "click-button", "click-checkboxes", "click-option", "click-dialog"),
    (
        "click-widget",
        "click-link",
        "click-button",
        "click-checkboxes",
        "click-option",
        "click-dialog",
    ),
    (
        "click-button-sequence",
        "click-widget",
        "click-link",
        "click-button",
        "click-checkboxes",
        "click-option",
        "click-dialog",
    ),
]:
    FAMILIES["_".join(parts)] = parts


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "big")


class Instance(StrictModel):
    id: str
    task: str
    split: Literal["train", "dev", "eval"]
    values_seed: int
    layout_seed: int
    order_seed: int
    order: Literal["forward", "reverse"]
    wording: Literal["direct", "reverse_clauses"]

    @property
    def components(self) -> tuple[str, ...]:
        return FAMILIES[self.task]

    @property
    def required_order(self) -> list[int]:
        values = list(range(len(self.components)))
        return values if self.order == "forward" else values[::-1]

    def browser_config(self) -> dict:
        layout = list(range(len(self.components)))
        random.Random(self.layout_seed).shuffle(layout)
        return dict(
            components=self.components,
            layout=layout,
            seeds=[stable_seed(self.values_seed, c) for c in self.components],
            order=self.required_order,
            wording=self.wording,
        )


class Manifest(StrictModel):
    version: str = TASK_REVISION
    miniwob_revision: str = MINIWOB_REVISION
    compwob_revision: str = COMPWOB_REVISION
    max_steps: int = Field(default=64, ge=1)
    prediction_horizon: int = Field(default=16, ge=1)
    execution_prefix: int = Field(default=1, ge=1)
    settle_ms: int = Field(default=100, ge=0)
    instances: list[Instance]

    @model_validator(mode="after")
    def validate_manifest(self):
        if (
            self.version != TASK_REVISION
            or self.miniwob_revision != MINIWOB_REVISION
            or self.compwob_revision != COMPWOB_REVISION
        ):
            raise ValueError("Unsupported task/environment revision")
        if self.execution_prefix > self.prediction_horizon:
            raise ValueError("Execution prefix cannot exceed prediction horizon")
        ids, split_seeds = set(), {}
        for item in self.instances:
            if item.task not in FAMILIES:
                raise ValueError(f"Unsupported mouse task: {item.task}")
            if item.id in ids:
                raise ValueError(f"Duplicate instance: {item.id}")
            ids.add(item.id)
            # A paired UI may occur across orders/wordings, but never across splits.
            for key in [
                (item.task, "values", item.values_seed),
                (item.task, "layout", item.layout_seed),
            ]:
                if key in split_seeds and split_seeds[key] != item.split:
                    raise ValueError("Instance leakage across splits")
                split_seeds[key] = item.split
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def make_manifest(
    train: int, dev: int, evaluation: int, seed: int = 2026, tasks: list[str] | None = None
) -> Manifest:
    if min(train, dev, evaluation) < 0:
        raise ValueError("Split counts must be nonnegative")
    instances = []
    for task in tasks or list(FAMILIES):
        if task not in FAMILIES:
            raise ValueError(f"Unknown task: {task}")
        for split, count in [("train", train), ("dev", dev), ("eval", evaluation)]:
            for index in range(count):
                for order in ["forward", "reverse"] if len(FAMILIES[task]) > 1 else ["forward"]:
                    for wording in ["direct", "reverse_clauses"]:
                        instances.append(
                            Instance(
                                id=f"{task}/{split}/{index:06d}/{order}/{wording}",
                                task=task,
                                split=split,
                                order=order,
                                wording=wording,
                                values_seed=stable_seed(seed, task, split, index, "values"),
                                layout_seed=stable_seed(seed, task, split, index, "layout"),
                                order_seed=stable_seed(seed, task, split, index, order),
                            )
                        )
    return Manifest(instances=instances)


def read_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text())
