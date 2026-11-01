"""Private chronology ledger; never pass this module's state to policies."""

from typing import Literal

from .contracts import StrictModel


class Event(StrictModel):
    component: int
    kind: Literal["progress", "complete", "failure"]
    step: int


class Verdict(StrictModel):
    goals_valid: bool
    order_valid: bool
    budget_valid: bool
    success: bool
    first_violation: Event | None
    completed: list[int]
    reason: str


class OrderValidator:
    def __init__(self, required_order: list[int]):
        if not required_order or len(set(required_order)) != len(required_order):
            raise ValueError("Order must contain distinct components")
        self.order = required_order
        self.completed: set[int] = set()
        self.events: list[Event] = []
        self.first_violation: Event | None = None
        self.failed = False

    def ingest(self, event: Event) -> None:
        if event.component not in self.order:
            raise ValueError("Unknown component in evaluator event")
        self.events.append(event)
        earlier = self.order[: self.order.index(event.component)]
        if not set(earlier).issubset(self.completed) and self.first_violation is None:
            self.first_violation = event
        if event.kind == "complete":
            self.completed.add(event.component)
        if event.kind == "failure":
            self.failed = True

    def verdict(self, steps: int, budget: int, terminated: bool, reason: str = "") -> Verdict:
        goals = set(self.order) == self.completed and not self.failed
        ordered = self.first_violation is None
        within = terminated and 0 < steps <= budget
        success = goals and ordered and within and not reason
        return Verdict(
            goals_valid=goals,
            order_valid=ordered,
            budget_valid=within,
            success=success,
            first_violation=self.first_violation,
            completed=sorted(self.completed),
            reason=reason
            or (
                "success"
                if success
                else "order_violation"
                if not ordered
                else "component_failure"
                if self.failed
                else "budget_exhausted"
            ),
        )
