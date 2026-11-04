"""Public screenshot-only interface, deliberately independent of private task state."""

from dataclasses import dataclass
from typing import Protocol

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

ACTION_VERSION = "scriptwob-click-v1"


class ClickAction(BaseModel):
    """One press-release click in normalized screenshot coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)

    def pixels(self, width: int, height: int) -> tuple[int, int]:
        if min(width, height) < 2:
            raise ValueError("Viewport must be at least two pixels in each dimension")
        return int(self.x * (width - 1) + 0.5), int(self.y * (height - 1) + 0.5)


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    step: int
    action: ClickAction
    feedback: str = "clicked"


@dataclass(frozen=True)
class Observation:
    instruction: str
    image: Image.Image
    history: tuple[HistoryEntry, ...] = ()


def format_context(instruction: str, history: tuple[HistoryEntry, ...]) -> str:
    """Shared by expert prompting and student export; contains no evaluator fields."""
    lines = ["PROCEDURE", instruction, "EXECUTED CLICK HISTORY"]
    for entry in history:
        lines.append(
            f"{entry.step}: ({entry.action.x:.6f}, {entry.action.y:.6f}) {entry.feedback}"
        )
    if not history:
        lines.append("(none)")
    return "\n".join(lines)


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...
    def predict(self, observation: Observation) -> ClickAction: ...
    def provenance(self) -> dict: ...
