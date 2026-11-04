"""Serializable, versioned contracts for scriptWoB tasks.

Programs and their annotations are evaluator-private. Only ``World.public_dict``
and the natural-language instruction belong in a policy observation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SemanticAction(StrictModel):
    op: Literal["click", "increment", "decrement", "toggle", "advance", "approve", "reject", "done"]
    target: str = Field(min_length=1, max_length=80)

    @property
    def key(self) -> str:
        return f"{self.target}:{self.op}"


class Item(StrictModel):
    id: str
    label: str
    category: str
    status: str = "open"
    processed: bool = False
    result: str = ""


class Widget(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$", max_length=80)
    label: str = Field(min_length=1, max_length=80)
    kind: Literal["counter", "toggle", "button", "status", "queue"]
    value: int | bool | str = 0
    minimum: int = -100
    maximum: int = 100
    items: list[Item] = Field(default_factory=list, max_length=12)
    cursor: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_state(self) -> Widget:
        if self.id == "done":
            raise ValueError("done is reserved for the completion control")
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        if self.kind in {"counter", "button"}:
            if type(self.value) is not int:
                raise ValueError("Counters and buttons need integer values")
            if not self.minimum <= self.value <= self.maximum:
                raise ValueError("Widget value outside bounds")
        if self.kind == "toggle" and type(self.value) is not bool:
            raise ValueError("Toggle value must be Boolean")
        if self.cursor > len(self.items):
            raise ValueError("Queue cursor exceeds collection length")
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Queue item IDs must be unique")
        return self


class World(StrictModel):
    widgets: list[Widget] = Field(min_length=1, max_length=12)
    events: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_widgets(self) -> World:
        if len({widget.id for widget in self.widgets}) != len(self.widgets):
            raise ValueError("Widget IDs must be unique")
        if len({widget.label for widget in self.widgets}) != len(self.widgets):
            raise ValueError("Widget labels must be unique")
        return self

    def widget(self, target: str) -> Widget:
        for widget in self.widgets:
            if widget.id == target:
                return widget
        raise ValueError(f"Unknown widget: {target}")

    def public_dict(self) -> dict:
        return self.model_dump(mode="json")


class Predicate(StrictModel):
    kind: Literal["compare", "and", "or", "not"] = "compare"
    target: str | None = None
    field: Literal["value", "remaining", "cursor", "category", "status", "label", "processed"] = (
        "value"
    )
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"] = "eq"
    value: int | bool | str | None = None
    operands: list[Predicate] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def well_formed(self) -> Predicate:
        if self.kind == "compare":
            if self.target is None or self.value is None or self.operands:
                raise ValueError("Comparison requires target/value and no operands")
        else:
            if self.target is not None or self.value is not None:
                raise ValueError("Boolean predicates use operands, not target/value")
            expected = 1 if self.kind == "not" else 2
            if len(self.operands) < expected or (self.kind == "not" and len(self.operands) != 1):
                raise ValueError("Invalid Boolean predicate arity")
        return self


class Statement(StrictModel):
    """A validated recursive AST; irrelevant fields are rejected by its validator."""

    kind: Literal["action", "sequence", "if", "repeat", "while", "until", "foreach"]
    action: SemanticAction | None = None
    steps: list[Statement] = Field(default_factory=list, max_length=32)
    condition: Predicate | None = None
    body: Statement | None = None
    otherwise: Statement | None = None
    count: int | None = Field(default=None, ge=0, le=100)
    collection: str | None = None

    @model_validator(mode="after")
    def valid_shape(self) -> Statement:
        present = {
            name
            for name in ("action", "steps", "condition", "body", "otherwise", "count", "collection")
            if getattr(self, name) is not None and getattr(self, name) != []
        }
        required = {
            "action": {"action"},
            "sequence": {"steps"},
            "if": {"condition", "body", "otherwise"},
            "repeat": {"count", "body"},
            "while": {"condition", "body"},
            "until": {"condition", "body"},
            "foreach": {"collection", "body"},
        }[self.kind]
        if present != required:
            raise ValueError(
                f"{self.kind} requires exactly {sorted(required)}, got {sorted(present)}"
            )
        if self.action is not None and self.action.op == "done":
            raise ValueError("Done is implicit after the program, never a program statement")
        return self


class Complexity(StrictModel):
    program_size: int = 0
    control_depth: int = 0
    loop_depth: int = 0
    predicate_count: int = 0
    predicate_complexity: int = 0
    reference_actions: int = 0
    branch_evaluations: int = 0
    loop_iterations: list[int] = Field(default_factory=list)
    widgets: int = 0
    items: int = 0
    operators: list[str] = Field(default_factory=list)


class Episode(StrictModel):
    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    split: Literal["train", "dev", "test"]
    level: int = Field(ge=0, le=5)
    family: str
    seed: int = Field(ge=0)
    layout_seed: int = Field(ge=0)
    program: Statement
    world: World
    instruction: str = Field(min_length=1, max_length=16000)
    complexity: Complexity = Field(default_factory=Complexity)
    structural_id: str
    max_steps: int = Field(default=128, ge=1, le=1000)


class Manifest(StrictModel):
    version: Literal["scriptwob-v1"] = "scriptwob-v1"
    seed: int = Field(default=0, ge=0)
    split_strategy: Literal["seed", "structure"] = "seed"
    episodes: list[Episode] = Field(default_factory=list)

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @model_validator(mode="after")
    def unique_episodes(self) -> Manifest:
        if len({episode.id for episode in self.episodes}) != len(self.episodes):
            raise ValueError("Episode IDs must be unique")
        if self.split_strategy == "structure":
            owners: dict[str, str] = {}
            for episode in self.episodes:
                previous = owners.setdefault(episode.structural_id, episode.split)
                if previous != episode.split:
                    raise ValueError("Structural split overlap")
        return self


class StepResult(StrictModel):
    done: bool
    success: bool
    error: str | None = None
    action: SemanticAction | None = None
