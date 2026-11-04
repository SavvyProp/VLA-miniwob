"""Authoritative state transitions and private, observation-dependent interpreter."""

from __future__ import annotations

import operator
from collections.abc import Generator

from .schema import Episode, Predicate, SemanticAction, Statement, StepResult, World


def evaluate(condition: Predicate, world: World) -> bool:
    if condition.kind == "and":
        return all(evaluate(value, world) for value in condition.operands)
    if condition.kind == "or":
        return any(evaluate(value, world) for value in condition.operands)
    if condition.kind == "not":
        return not evaluate(condition.operands[0], world)
    widget = world.widget(condition.target or "")
    if condition.field == "value":
        actual = widget.value
    elif condition.field == "remaining":
        actual = len(widget.items) - widget.cursor
    elif condition.field == "cursor":
        actual = widget.cursor
    else:
        # An exhausted queue has no current record; all its item comparisons
        # are false, including !=. Use remaining/cursor to test exhaustion.
        if widget.cursor >= len(widget.items):
            return False
        actual = getattr(widget.items[widget.cursor], condition.field)
    comparison = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }[condition.operator]
    return bool(comparison(actual, condition.value))


def available_actions(world: World) -> list[SemanticAction]:
    actions = [SemanticAction(op="done", target="done")]
    for widget in world.widgets:
        if widget.kind == "button" and widget.value < widget.maximum:
            operations = ["click"]
        elif widget.kind == "counter":
            operations = []
            if widget.value < widget.maximum:
                operations.append("increment")
            if widget.value > widget.minimum:
                operations.append("decrement")
        elif widget.kind == "toggle":
            operations = ["toggle"]
        elif widget.kind == "queue" and widget.cursor < len(widget.items):
            operations = ["advance"]
            if not widget.items[widget.cursor].processed:
                operations.extend(["approve", "reject"])
        else:
            operations = []
        actions.extend(SemanticAction(op=operation, target=widget.id) for operation in operations)
    return actions


def transition(world: World, action: SemanticAction) -> None:
    """Apply a physical control event in place; knows nothing about the program."""
    if action not in available_actions(world):
        raise ValueError(f"Unavailable action: {action.key}")
    if action.op == "done":
        world.events.append("Clicked Done.")
        return
    widget = world.widget(action.target)
    if action.op in {"click", "increment"}:
        widget.value += 1
    elif action.op == "decrement":
        widget.value -= 1
    elif action.op == "toggle":
        widget.value = not widget.value
    elif action.op == "advance":
        widget.cursor += 1
    elif action.op in {"approve", "reject"}:
        item = widget.items[widget.cursor]
        item.processed = True
        item.result = "approved" if action.op == "approve" else "rejected"
    verbs = {
        "click": "Pressed",
        "increment": "Increased",
        "decrement": "Decreased",
        "toggle": "Toggled",
        "advance": "Selected Next on",
        "approve": "Approved current item in",
        "reject": "Rejected current item in",
    }
    world.events.append(f"{verbs[action.op]} {widget.label}.")


def validate_program(program: Statement, world: World) -> None:
    """Validate all branches, even those not taken in a particular episode."""
    visited = 0

    def predicate(condition: Predicate, depth: int = 0) -> None:
        if depth > 16:
            raise ValueError("Predicate exceeds maximum depth 16")
        if condition.kind != "compare":
            for operand in condition.operands:
                predicate(operand, depth + 1)
            return
        widget = world.widget(condition.target or "")
        if condition.field != "value" and widget.kind != "queue":
            raise ValueError("Queue fields require a queue widget")
        field_types = {
            "value": type(widget.value),
            "remaining": int,
            "cursor": int,
            "category": str,
            "status": str,
            "label": str,
            "processed": bool,
        }
        if type(condition.value) is not field_types[condition.field]:
            raise ValueError("Predicate operands must have matching types")
        if condition.operator not in {"eq", "ne"} and type(condition.value) is not int:
            raise ValueError("Ordered comparisons are numeric only")

    def visit(node: Statement, depth: int = 0) -> None:
        nonlocal visited
        visited += 1
        if depth > 16 or visited > 256:
            raise ValueError("Program exceeds maximum depth 16 or size 256")
        if node.action:
            widget = world.widget(node.action.target)
            valid_ops = {
                "button": {"click"},
                "counter": {"increment", "decrement"},
                "toggle": {"toggle"},
                "status": set(),
                "queue": {"advance", "approve", "reject"},
            }[widget.kind]
            if node.action.op not in valid_ops:
                raise ValueError("Action is incompatible with widget kind")
        if node.condition:
            predicate(node.condition)
        if node.collection and world.widget(node.collection).kind != "queue":
            raise ValueError("For-each collection must be a queue")
        for child in node.steps:
            visit(child, depth + 1)
        if node.body:
            visit(node.body, depth + 1)
        if node.otherwise:
            visit(node.otherwise, depth + 1)

    visit(program)


class Engine:
    """Strict trace monitor. The policy never receives this object or its oracle.

    Every authored sequence is ordered; for-each traverses the displayed order.
    Consequently the next *semantic* action is unique. Pixel locations are not
    prescribed. Both while and until check before each iteration.
    """

    def __init__(self, episode: Episode):
        validate_program(episode.program, episode.world)
        self.episode = episode
        self.world = episode.world.model_copy(deep=True)
        self.done = False
        self.success = False
        self.error: str | None = None
        self.trace: list[SemanticAction] = []
        self.branch_evaluations = 0
        self.loop_iterations: list[int] = []
        self._visits = 0
        self._iterator = self._execute(episode.program)
        self._expected: SemanticAction | None = None
        self._advance()

    def _execute(self, node: Statement) -> Generator[SemanticAction, None, None]:
        self._visits += 1
        if self._visits > 10000:
            raise ValueError("Program exceeded interpreter work budget")
        if node.kind == "action":
            yield node.action
        elif node.kind == "sequence":
            for child in node.steps:
                yield from self._execute(child)
        elif node.kind == "if":
            self.branch_evaluations += 1
            branch = node.body if evaluate(node.condition, self.world) else node.otherwise
            yield from self._execute(branch)
        elif node.kind in {"repeat", "while", "until"}:
            slot = len(self.loop_iterations)
            self.loop_iterations.append(0)
            iteration = 0
            while True:
                if node.kind == "repeat":
                    continuing = iteration < node.count
                else:
                    condition = evaluate(node.condition, self.world)
                    continuing = condition if node.kind == "while" else not condition
                if not continuing:
                    break
                if iteration >= 1000:
                    raise ValueError("Loop exceeded 1000 iterations")
                yield from self._execute(node.body)
                iteration += 1
                self.loop_iterations[slot] = iteration
        elif node.kind == "foreach":
            widget = self.world.widget(node.collection)
            slot = len(self.loop_iterations)
            self.loop_iterations.append(0)
            for cursor in range(widget.cursor, len(widget.items)):
                if widget.cursor != cursor:
                    raise ValueError("For-each body must not move its collection cursor")
                yield from self._execute(node.body)
                if widget.cursor != cursor:
                    raise ValueError("For-each body must not move its collection cursor")
                yield SemanticAction(op="advance", target=widget.id)
                self.loop_iterations[slot] += 1

    def _advance(self) -> None:
        try:
            self._expected = next(self._iterator)
            if self._expected not in available_actions(self.world):
                raise ValueError(f"Program requires unavailable action {self._expected.key}")
        except StopIteration:
            self._expected = SemanticAction(op="done", target="done")
        except (ValueError, TypeError) as error:
            self.done = True
            self.error = f"invalid_program: {error}"
            self._expected = None

    def expected_action(self) -> SemanticAction | None:
        return None if self.done else self._expected.model_copy()

    def available_actions(self) -> list[SemanticAction]:
        return [] if self.done else available_actions(self.world)

    def invalid_action(self, reason: str = "No interactive control was selected") -> StepResult:
        if not self.done:
            self.done = True
            self.error = f"invalid_action: {reason}"
        return StepResult(done=self.done, success=self.success, error=self.error)

    def step(self, action: SemanticAction) -> StepResult:
        if self.done:
            raise ValueError("Episode is already terminal")
        self.trace.append(action.model_copy())
        if action != self._expected:
            if action in available_actions(self.world):
                transition(self.world, action)
            self.done = True
            if action.op == "done":
                self.error = "premature_done"
            elif self._expected.op == "done":
                self.error = "extra_action"
            else:
                self.error = "wrong_action"
            return StepResult(done=True, success=False, error=self.error, action=action)
        transition(self.world, action)
        if action.op == "done":
            self.done = self.success = True
        else:
            self._advance()
        if not self.done and len(self.trace) >= self.episode.max_steps:
            self.done = True
            self.error = "step_budget_exhausted"
        return StepResult(done=self.done, success=self.success, error=self.error, action=action)


def oracle_trace(episode: Episode) -> tuple[list[SemanticAction], Engine]:
    engine = Engine(episode)
    while not engine.done:
        engine.step(engine.expected_action())
    if not engine.success:
        raise ValueError(f"Episode is not solvable within its budget: {engine.error}")
    return engine.trace, engine
