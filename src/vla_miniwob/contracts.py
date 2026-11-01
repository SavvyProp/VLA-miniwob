"""Versioned, model-independent policy and execution contracts."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "mouse-absolute-click-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MouseAction(StrictModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    click: float = Field(allow_inf_nan=False)

    def vector(self) -> list[float]:
        return [self.x, self.y, self.click]


class ExecutedAction(StrictModel):
    action: MouseAction
    pixel_x: int
    pixel_y: int
    kind: Literal["move", "click"]
    clipped: bool


def decode(action: MouseAction, width: int, height: int) -> ExecutedAction:
    """Clip finite coordinates, round to nearest pixel, emit one complete click."""
    if min(width, height) < 2:
        raise ValueError("Viewport must have at least two pixels per dimension")
    x, y = np.clip([action.x, action.y], 0, 1).tolist()
    px, py = int(np.floor(x * (width - 1) + 0.5)), int(np.floor(y * (height - 1) + 0.5))
    return ExecutedAction(
        action=MouseAction(
            x=px / (width - 1), y=py / (height - 1), click=float(action.click >= 0.5)
        ),
        pixel_x=px,
        pixel_y=py,
        kind="click" if action.click >= 0.5 else "move",
        clipped=x != action.x or y != action.y,
    )


class HistoryEntry(StrictModel):
    step: int
    action: MouseAction
    feedback: str


@dataclass(frozen=True)
class PolicyObservation:
    """The only data exposed to a policy: no DOM, reward, goals, or semantic stage."""

    instruction: str
    image: Image.Image
    state: tuple[float, float, float]
    history: tuple[HistoryEntry, ...]


def format_context(instruction: str, history: tuple[HistoryEntry, ...]) -> str:
    lines = ["PROCEDURE", instruction, "EXECUTED ACTION HISTORY"]
    lines.extend(
        f"{h.step}: ({h.action.x:.6f}, {h.action.y:.6f}, {h.action.click:.0f}) {h.feedback}"
        for h in history
    )
    if not history:
        lines.append("(none)")
    return "\n".join(lines)


class Policy(Protocol):
    """Adapters return one sampled H x 3 chunk in physical [0,1] GUI coordinates."""

    def reset(self, seed: int) -> None: ...
    def predict(self, observation: PolicyObservation) -> np.ndarray: ...
    def provenance(self) -> dict: ...
