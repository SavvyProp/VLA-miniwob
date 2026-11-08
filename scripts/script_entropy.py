"""Exact execution-path entropy under an explicit intervention prior.

This is an offline analysis utility, not part of the benchmark runtime. It holds
the program and widget inventory fixed, enumerates counterfactual initial worlds,
and delegates execution to the authoritative scriptWoB interpreter.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

from scriptwob.engine import Engine, evaluate, oracle_trace
from scriptwob.schema import Episode, Item, Predicate, SemanticAction, Statement, Widget, World

PRIOR_VERSIONS = {
    "observed-workload": "scriptwob-observed-queue-length-v2",
    "reference": "scriptwob-independent-initial-state-v1",
}
PRIOR_VERSION = PRIOR_VERSIONS["observed-workload"]
Path = tuple[tuple[str, str, str], ...]


class EntropyLimitError(RuntimeError):
    """Enumeration/execution hit a limit; no entropy estimate should be reported."""


@dataclass(frozen=True)
class TraceResult:
    path: Path
    actions: tuple[SemanticAction, ...]
    success: bool
    error: str | None
    enabled_action_counts: tuple[int, ...]
    branch_evaluations: int


def _sites(program: Statement) -> dict[int, str]:
    sites: dict[int, str] = {}

    def visit(node: Statement, site: str) -> None:
        if id(node) in sites:
            raise ValueError("Shared AST node identity")
        sites[id(node)] = site
        for index, child in enumerate(node.steps):
            visit(child, f"{site}.steps[{index}]")
        if node.body is not None:
            visit(node.body, f"{site}.body")
        if node.otherwise is not None:
            visit(node.otherwise, f"{site}.otherwise")

    visit(program, "root")
    return sites


class _TracingEngine(Engine):
    def __init__(self, episode: Episode):
        # ASTs loaded from JSON already have distinct node objects. Normalize
        # hand-built ASTs that reuse a Python object at multiple source sites.
        try:
            self._sites = _sites(episode.program)
        except ValueError:
            program = Statement.model_validate(episode.program.model_dump())
            episode = episode.model_copy(update={"program": program})
            self._sites = _sites(program)
        self.path_events: list[tuple[str, str, str]] = []
        super().__init__(episode)

    def _execute(self, node: Statement):
        site = self._sites[id(node)]
        self.path_events.append((site, node.kind, "enter"))
        if node.kind == "if":
            # evaluate() is pure. Logging the outcome before the real
            # interpreter runs preserves behavior and distinguishes identical
            # branch bodies. Boolean short-circuit internals are not recorded.
            outcome = "true" if evaluate(node.condition, self.world) else "false"
            self.path_events.append((site, node.kind, outcome))
        # Every loop body entry/exit is recorded at its static source site.
        # Together with loop entry/exit this identifies zero iterations, each
        # iteration, nesting, and the final stopping decision without copying
        # the interpreter's control-flow implementation.
        yield from super()._execute(node)
        self.path_events.append((site, node.kind, "exit"))


def trace_world(
    episode: Episode, world: World | None = None, *, max_steps: int = 1000
) -> TraceResult:
    """Execute one world, overriding the saved rollout's action budget.

    Invalid/unavailable-action worlds return success=False. Work/action limits
    raise rather than silently becoming rejected prior mass.
    """
    if not 1 <= max_steps <= 1000:
        raise ValueError("max_steps must be between 1 and 1000")
    candidate = episode.model_copy(
        update={"world": episode.world if world is None else world, "max_steps": max_steps}
    )
    engine = _TracingEngine(candidate)
    enabled_action_counts = []
    while not engine.done:
        action = engine.expected_action()
        available = engine.available_actions()
        if action not in available:
            raise AssertionError("The oracle must select an enabled action")
        enabled_action_counts.append(len(available))
        engine.step(action)
    error = engine.error or ""
    if any(
        marker in error
        for marker in ("step_budget_exhausted", "work budget", "exceeded 1000 iterations")
    ):
        raise EntropyLimitError(f"{episode.id}: {error}")
    return TraceResult(
        tuple(engine.path_events), tuple(engine.trace), engine.success, engine.error,
        tuple(enabled_action_counts), engine.branch_evaluations,
    )


def _predicate_fields(program: Statement) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = defaultdict(set)

    def condition(predicate: Predicate) -> None:
        if predicate.kind == "compare":
            fields[predicate.target].add(predicate.field)
        else:
            for operand in predicate.operands:
                condition(operand)

    def visit(node: Statement) -> None:
        if node.condition is not None:
            condition(node.condition)
        for child in [*node.steps, node.body, node.otherwise]:
            if child is not None:
                visit(child)

    visit(program)
    return fields


def _widget_options(
    widget: Widget, fields: set[str], *, prior: str
) -> list[tuple[Widget, Fraction]]:
    """Representative states with exactly marginalized probabilities."""
    if widget.kind == "queue":
        if prior == "observed-workload":
            # Keep the observed workload, including any progress through it.
            # Only unread records can affect subsequent execution: queue
            # cursors never move backwards. Unused fields and all metadata
            # retain their observed values instead of being regenerated.
            varied_fields = [name for name in ("category", "status") if name in fields]
            domains = {"category": ("red", "blue"), "status": ("open", "closed")}
            assignments = list(itertools.product(*(domains[name] for name in varied_fields)))
            remaining = len(widget.items) - widget.cursor
            mass = Fraction(1, len(assignments) ** remaining)
            options = []
            for record_values in itertools.product(assignments, repeat=remaining):
                value = widget.model_dump()
                for index, values in enumerate(record_values, start=widget.cursor):
                    value["items"][index].update(zip(varied_fields, values, strict=True))
                options.append((Widget.model_validate(value), mass))
            return options
        categories = ("red", "blue") if "category" in fields else ("red",)
        statuses = ("open", "closed") if "status" in fields else ("open",)
        options = []
        item_fields = tuple(itertools.product(categories, statuses))
        for length in range(1, 6):
            # Length is uniform FIRST; assignments within a length are uniform.
            # A flattened uniform enumeration would overweight long queues.
            mass = Fraction(1, 5 * len(item_fields) ** length)
            for assignment in itertools.product(item_fields, repeat=length):
                items = [
                    Item(
                        id=f"record{index}", label=f"Record {index + 1}",
                        category=category, status=status,
                    )
                    for index, (category, status) in enumerate(assignment)
                ]
                value = widget.model_dump()
                value.update(items=[item.model_dump() for item in items], cursor=0, value=0)
                options.append((Widget.model_validate(value), mass))
        return options
    if widget.kind == "counter":
        values = tuple(value for value in range(4) if widget.minimum <= value <= widget.maximum)
        if not values:
            raise ValueError(f"{widget.id}: counter bounds exclude the prior's values 0..3")
    elif widget.kind == "toggle":
        values = (False, True)
    elif widget.kind == "status":
        if type(widget.value) is not str:
            raise ValueError(f"{widget.id}: status prior requires a string-valued widget")
        values = ("ready", "waiting")
    elif widget.kind == "button":
        values = (0,)
    else:
        raise ValueError(f"Unsupported widget kind: {widget.kind}")
    result = []
    for value in values:
        candidate = widget.model_dump()
        candidate.update(value=value)
        result.append((Widget.model_validate(candidate), Fraction(1, len(values))))
    return result


def _option_count(widget: Widget, fields: set[str], *, prior: str) -> int:
    """Check the Cartesian size before allocating exponential queue options."""
    if widget.kind == "queue":
        assignments = 2 ** len({"category", "status"} & fields)
        if prior == "observed-workload":
            return assignments ** (len(widget.items) - widget.cursor)
        return sum(assignments ** length for length in range(1, 6))
    if widget.kind == "counter":
        return sum(widget.minimum <= value <= widget.maximum for value in range(4))
    return 2 if widget.kind in {"toggle", "status"} else 1


def _fraction(value: Fraction) -> dict[str, int | float | str]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": str(value),
        "value": float(value),
    }


def evaluate_entropy(
    episode: Episode, *, prior: str = "observed-workload", max_steps: int = 1000,
    max_states: int = 1_000_000,
) -> tuple[dict, list[dict]]:
    """Return exact path probabilities and Shannon entropy in bits.

    The default prior fixes observed queue lengths, cursors and record metadata,
    varying referenced category/status fields on remaining records uniformly.
    This conditions on queue workload, not the complete initial page: counters
    still vary uniformly over legal 0..3 values, and toggle/status widgets retain
    their fair priors. Program, widget inventory and constants remain fixed.
    ``prior="reference"`` reproduces the original uniform 1..5 queue-length prior.
    Neither prior is the generator's conditional distribution of worlds given
    that program. The original correct trace also supplies a separate uniform
    enabled-button choice cost; this is never added to execution-path entropy.
    Unsolvable worlds are excluded by globally conditioning on solvability.
    Enumeration, prior masses, and path probabilities use exact Fractions;
    only logarithms and reported entropy/effective counts use floating point.
    """
    if prior not in PRIOR_VERSIONS:
        raise ValueError(f"Unknown prior {prior!r}; choose one of {sorted(PRIOR_VERSIONS)}")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    original = trace_world(episode, max_steps=max_steps)
    if not original.success:
        raise ValueError(f"Original world is not solvable: {episode.id}: {original.error}")
    oracle, _ = oracle_trace(episode.model_copy(update={"max_steps": max_steps}))
    if list(original.actions) != oracle:
        raise AssertionError("Instrumentation changed the original oracle action trace")

    fields = _predicate_fields(episode.program)
    state_count = math.prod(
        _option_count(widget, fields[widget.id], prior=prior) for widget in episode.world.widgets
    )
    if state_count > max_states:
        raise EntropyLimitError(
            f"{episode.id}: {state_count} states exceed max_states={max_states}"
        )
    options = [
        _widget_options(widget, fields[widget.id], prior=prior) for widget in episode.world.widgets
    ]
    if state_count != math.prod(map(len, options)):
        raise AssertionError("Prior option count differs from enumeration")
    masses: dict[Path, Fraction] = defaultdict(Fraction)
    counts: dict[Path, int] = defaultdict(int)
    action_counts: dict[Path, set[int]] = defaultdict(set)
    rejections: dict[str, dict] = {}
    total_mass = Fraction()
    valid_mass = Fraction()
    valid_states = 0

    for assignment in itertools.product(*options):
        world = episode.world.model_copy(update={"widgets": [option[0] for option in assignment]})
        mass = math.prod((option[1] for option in assignment), start=Fraction(1))
        total_mass += mass
        trace = trace_world(episode, world, max_steps=max_steps)
        if not trace.success:
            entry = rejections.setdefault(trace.error, {"state_count": 0, "prior_mass": Fraction()})
            entry["state_count"] += 1
            entry["prior_mass"] += mass
            continue
        valid_states += 1
        valid_mass += mass
        masses[trace.path] += mass
        counts[trace.path] += 1
        action_counts[trace.path].add(len(trace.actions))

    if total_mass != 1:
        raise AssertionError(f"Prior mass must sum to 1, got {total_mass}")
    if not valid_mass:
        raise ValueError(f"{episode.id}: no solvable worlds under the specified prior")
    probabilities = {path: mass / valid_mass for path, mass in masses.items()}
    if sum(probabilities.values(), start=Fraction()) != 1:
        raise AssertionError("Conditional path probabilities must sum exactly to 1")
    entropy = math.fsum(-float(p) * math.log2(float(p)) for p in probabilities.values())
    original_probability = probabilities.get(original.path, Fraction())
    all_lengths = set().union(*action_counts.values())
    paths = []
    for path, probability in sorted(probabilities.items()):
        if len(action_counts[path]) != 1:
            raise AssertionError("One complete execution path must fix its action count")
        encoded = json.dumps(path, separators=(",", ":"))
        paths.append({
            "path_id": hashlib.sha256(encoded.encode()).hexdigest(),
            "events": [list(event) for event in path],
            "probability": _fraction(probability),
            "prior_mass": _fraction(masses[path]),
            "state_count": counts[path],
            "required_actions": next(iter(action_counts[path])),
            "is_original_path": path == original.path,
        })
    choice_probability = Fraction(1, math.prod(original.enabled_action_counts))
    summary = {
        "prior": prior,
        "prior_version": PRIOR_VERSIONS[prior],
        "episode_id": episode.id,
        "entropy_bits": entropy,
        "path_count": len(paths),
        "effective_path_count": 2 ** entropy,
        "original_path_probability": float(original_probability),
        "original_path_probability_exact": str(original_probability),
        "original_surprisal_bits": (
            -math.log2(float(original_probability)) if original_probability else None
        ),
        "original_path_in_prior_support": bool(original_probability),
        "valid_prior_mass": float(valid_mass),
        "valid_prior_mass_exact": str(valid_mass),
        "rejected_prior_mass": float(1 - valid_mass),
        "rejected_prior_mass_exact": str(1 - valid_mass),
        "rejection_reasons": {
            reason: {
                "state_count": value["state_count"],
                "prior_mass": _fraction(value["prior_mass"]),
            }
            for reason, value in sorted(rejections.items())
        },
        "enumerated_state_count": state_count,
        "valid_state_count": valid_states,
        "rejected_state_count": state_count - valid_states,
        "min_required_actions": min(all_lengths),
        "max_required_actions": max(all_lengths),
        "original_required_actions": len(original.actions),
        "original_branch_evaluations": original.branch_evaluations,
        "original_choice_bits": math.fsum(map(math.log2, original.enabled_action_counts)),
        "original_uniform_button_success_probability": float(choice_probability),
        "original_uniform_button_success_probability_exact": str(choice_probability),
        "original_enabled_action_counts": list(original.enabled_action_counts),
        "original_wrong_button_opportunities": sum(
            count - 1 for count in original.enabled_action_counts
        ),
        "original_queue_lengths": {
            widget.id: len(widget.items)
            for widget in episode.world.widgets if widget.kind == "queue"
        },
        "original_queue_remaining": {
            widget.id: len(widget.items) - widget.cursor
            for widget in episode.world.widgets if widget.kind == "queue"
        },
        "max_steps": max_steps,
        "marginalized_queue_fields": {
            widget.id: sorted({"category", "status"} - fields[widget.id])
            for widget in episode.world.widgets if widget.kind == "queue"
        },
    }
    return summary, paths
