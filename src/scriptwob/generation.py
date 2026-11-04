"""Seeded grammar instantiation, canonical language, and difficulty metadata.

The finite family catalog is deliberately explicit: this is controlled random
program generation, not a claim of sampling every possible natural-language
procedure. Literal values and layouts vary independently of structural splits.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

from .engine import oracle_trace, validate_program
from .schema import (
    Complexity,
    Episode,
    Item,
    Manifest,
    Predicate,
    SemanticAction,
    Statement,
    Widget,
    World,
)

FAMILIES: dict[int, dict[str, str]] = {
    0: {
        "single_button": "Press one named button",
        "single_toggle": "Toggle one checkbox",
        "single_counter": "Change a counter once",
    },
    1: {
        "ordered_buttons": "An ordered sequence of button presses",
        "mixed_controls": "An ordered sequence across widget types",
    },
    2: {
        "visible_branch": "Choose a branch from visible state",
        "branch_sequence": "Choose between short sequences",
        "compound_branch": "Branch on a Boolean combination",
    },
    3: {
        "counted_repeat": "Repeat a block a fixed number of times",
        "while_counter": "Increase a counter while a numeric condition holds",
        "until_counter": "Decrease a counter until a numeric condition holds",
        "queue_until": "Advance until a matching record",
        "foreach": "Process every record in displayed order",
    },
    4: {
        "foreach_if": "Classify each record using a conditional",
        "while_if": "Branch inside a state-dependent loop",
        "if_loop": "Choose between two loops",
        "repeat_if": "Repeated conditional execution",
    },
    5: {
        "nested_repeat": "Repeat a block containing repetition",
        "foreach_repeat_if": "Repeat a conditional for each record",
        "while_repeat": "Counted loop inside a while loop",
        "branch_nested_loop": "Branch into nested repetitions",
        "until_repeat_if": "Until loop containing repetition and branching",
    },
}


def action(target: str, op: str = "click") -> Statement:
    return Statement(kind="action", action=SemanticAction(target=target, op=op))


def sequence(*steps: Statement) -> Statement:
    return Statement(kind="sequence", steps=list(steps))


def compare(
    target: str, value: int | bool | str, operator: str = "eq", field: str = "value"
) -> Predicate:
    return Predicate(target=target, value=value, operator=operator, field=field)


def branch(condition: Predicate, yes: Statement, no: Statement) -> Statement:
    return Statement(kind="if", condition=condition, body=yes, otherwise=no)


def repeat(count: int, body: Statement) -> Statement:
    return Statement(kind="repeat", count=count, body=body)


def _shape(program: Statement) -> dict:
    """Canonical structure ignores literal values, IDs, labels, and repeat counts."""
    result: dict = {"kind": program.kind}
    if program.action:
        result["operation"] = program.action.op
    if program.condition:
        result["condition"] = _predicate_shape(program.condition)
    if program.steps:
        result["steps"] = [_shape(child) for child in program.steps]
    if program.body:
        result["body"] = _shape(program.body)
    if program.otherwise:
        result["otherwise"] = _shape(program.otherwise)
    return result


def _predicate_shape(predicate: Predicate) -> dict:
    if predicate.kind == "compare":
        return {
            "kind": "compare",
            "field": predicate.field,
            "operator": predicate.operator,
            "type": type(predicate.value).__name__,
        }
    children = [_predicate_shape(child) for child in predicate.operands]
    if predicate.kind in {"and", "or"}:
        children.sort(key=lambda child: json.dumps(child, sort_keys=True))
    return {"kind": predicate.kind, "operands": children}


def structural_id(program: Statement) -> str:
    encoded = json.dumps(_shape(program), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def structural_split(program: Statement) -> str:
    # Stable assignment independent of a manifest's seed and collection size.
    bucket = int(structural_id(program)[:12], 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "test")


def task_fingerprint(episode: Episode) -> str:
    """Task identity independent of seeds and internal IDs.

    Include effective card order, language, initial values, and complete queue
    contents, including records not yet revealed. Future records distinguish
    different interactive episodes with identical first screenshots. Literal UI
    labels intentionally remain part of this identity.
    """
    layout = [widget.id for widget in episode.world.widgets]
    random.Random(episode.layout_seed).shuffle(layout)
    widgets = []
    for target in layout:
        widget = episode.world.widget(target).model_dump(mode="json")
        widget.pop("id")
        for item in widget["items"]:
            item.pop("id")
        widgets.append(widget)
    data = {
        "instruction": episode.instruction,
        "widgets": widgets,
        "events": episode.world.events,
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def instance_split(episode: Episode) -> str:
    """Stable 60/20/20 assignment of exact tasks, not their random seed IDs."""
    bucket = int(task_fingerprint(episode)[:12], 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "test")


def render_predicate(predicate: Predicate, world: World) -> str:
    if predicate.kind == "not":
        return f"NOT ({render_predicate(predicate.operands[0], world)})"
    if predicate.kind in {"and", "or"}:
        separator = f" {predicate.kind.upper()} "
        return (
            "("
            + separator.join(render_predicate(child, world) for child in predicate.operands)
            + ")"
        )
    widget = world.widget(predicate.target)
    if predicate.field == "value":
        subject = f'the value of "{widget.label}"'
        if widget.kind == "toggle":
            subject = f'the checked state of "{widget.label}"'
    elif predicate.field == "remaining":
        subject = f'the number of remaining items in "{widget.label}"'
    elif predicate.field == "cursor":
        subject = f'the number of times Next has been clicked on "{widget.label}"'
    else:
        subject = f'the {predicate.field} of the current item in "{widget.label}"'
    relation = {
        "eq": "equals",
        "ne": "does not equal",
        "lt": "is less than",
        "le": "is at most",
        "gt": "is greater than",
        "ge": "is at least",
    }[predicate.operator]
    value = (
        ("checked" if predicate.value else "unchecked")
        if type(predicate.value) is bool and widget.kind == "toggle"
        else json.dumps(predicate.value)
    )
    return f"{subject} {relation} {value}"


def render_instruction(program: Statement, world: World) -> str:
    def render(node: Statement) -> str:
        if node.kind == "action":
            name = world.widget(node.action.target).label
            operation = {
                "click": f'Press "{name}"',
                "toggle": f'Toggle "{name}" once',
                "increment": f'Click + on "{name}" once',
                "decrement": f'Click − on "{name}" once',
                "advance": f'Click Next on "{name}" once',
                "approve": f'Click Approve for the current item in "{name}"',
                "reject": f'Click Reject for the current item in "{name}"',
            }[node.action.op]
            return operation
        if node.kind == "sequence":
            return "In order: " + "; then ".join(f"[{render(child)}]" for child in node.steps)
        if node.kind == "if":
            return (
                f"If {render_predicate(node.condition, world)}, execute [{render(node.body)}]; "
                f"otherwise execute [{render(node.otherwise)}]"
            )
        if node.kind == "repeat":
            return f"Repeat exactly {node.count} times: [{render(node.body)}]"
        if node.kind in {"while", "until"}:
            return (
                f"{node.kind.title()} {render_predicate(node.condition, world)}, "
                f"repeat [{render(node.body)}]. Check the condition before every iteration"
            )
        label = world.widget(node.collection).label
        return (
            f'For each remaining item in "{label}", in displayed order, '
            f'execute [{render(node.body)}], then click Next on "{label}". '
            "Stop when no items remain"
        )

    return (
        render(program)
        + ". When the whole script is complete, click Done. Do not perform any extra actions."
    )


def complexity_profile(program: Statement, world: World) -> Complexity:
    profile = Complexity(
        widgets=len(world.widgets), items=sum(len(widget.items) for widget in world.widgets)
    )
    operators: set[str] = set()

    def predicate_size(predicate: Predicate) -> int:
        return 1 + sum(predicate_size(child) for child in predicate.operands)

    def walk(node: Statement, control_depth: int = 0, loop_depth: int = 0) -> None:
        profile.program_size += 1
        operators.add(node.kind)
        control_depth += int(node.kind in {"if", "repeat", "while", "until", "foreach"})
        loop_depth += int(node.kind in {"repeat", "while", "until", "foreach"})
        profile.control_depth = max(profile.control_depth, control_depth)
        profile.loop_depth = max(profile.loop_depth, loop_depth)
        if node.condition:
            profile.predicate_count += 1
            profile.predicate_complexity = max(
                profile.predicate_complexity, predicate_size(node.condition)
            )
        for child in [*node.steps, node.body, node.otherwise]:
            if child:
                walk(child, control_depth, loop_depth)

    walk(program)
    profile.operators = sorted(operators)
    return profile


def _instantiate(rng: random.Random, level: int, family: str) -> tuple[Statement, World]:
    labels = rng.sample(
        ["Amber", "Cedar", "Maple", "Violet", "Coral", "Silver", "Indigo", "Olive"], 6
    )
    widgets = [
        Widget(id="a", label=labels[0], kind="button"),
        Widget(id="b", label=labels[1], kind="button"),
    ]
    used: set[str] = set()

    def add(widget: Widget) -> None:
        if not any(existing.id == widget.id for existing in widgets):
            widgets.append(widget)

    def counter() -> str:
        add(
            Widget(
                id="count",
                label=labels[2],
                kind="counter",
                value=rng.randrange(0, 4),
                minimum=0,
                maximum=100,
            )
        )
        return "count"

    def toggle() -> str:
        add(Widget(id="flag", label=labels[3], kind="toggle", value=bool(rng.randrange(2))))
        return "flag"

    def status() -> str:
        add(
            Widget(
                id="state", label=labels[4], kind="status", value=rng.choice(["ready", "waiting"])
            )
        )
        return "state"

    def queue() -> str:
        count = rng.randint(1, 5)
        items = [
            Item(
                id=f"record{i}",
                label=f"Record {i + 1}",
                category=rng.choice(["red", "blue"]),
                status=rng.choice(["open", "closed"]),
            )
            for i in range(count)
        ]
        add(Widget(id="queue", label=labels[5], kind="queue", items=items))
        return "queue"

    def numeric_value() -> int:
        return next(widget.value for widget in widgets if widget.id == "count")

    def simple_action(exclude_b: bool = False) -> Statement:
        selection = rng.randrange(4)
        if selection == 0:
            return action("a" if exclude_b else rng.choice(["a", "b"]))
        if selection == 1:
            return action(toggle(), "toggle")
        return action(counter(), "increment")

    def condition() -> Predicate:
        selection = rng.randrange(3)
        if selection == 0:
            return compare(status(), rng.choice(["ready", "waiting"]), rng.choice(["eq", "ne"]))
        if selection == 1:
            return compare(toggle(), bool(rng.randrange(2)), rng.choice(["eq", "ne"]))
        operation = rng.choice(["lt", "le", "eq", "gt", "ge"])
        # Both outcomes must be attainable under the generator's initial
        # counter distribution (0..3); e.g. "counter < 0" would be dead code.
        threshold = rng.randint(0, 3) if operation == "eq" else rng.randint(1, 2)
        return compare(counter(), threshold, operation)

    if level == 0:
        if family == "single_button":
            program = action(rng.choice(["a", "b"]))
        elif family == "single_toggle":
            program = action(toggle(), "toggle")
        else:
            target = counter()
            program = action(
                target, rng.choice(["increment", "decrement"]) if numeric_value() else "increment"
            )
    elif level == 1:
        length = rng.randint(2, 5)
        program = sequence(
            *(
                action(rng.choice(["a", "b"])) if family == "ordered_buttons" else simple_action()
                for _ in range(length)
            )
        )
    elif level == 2:
        predicate = condition()
        if family == "compound_branch":
            # Independent visible fields avoid tautologies and contradictions
            # produced by comparing one field to two incompatible values.
            other = (
                compare(toggle(), bool(rng.randrange(2)))
                if predicate.target != "flag"
                else compare(status(), rng.choice(["ready", "waiting"]))
            )
            predicate = Predicate(kind=rng.choice(["and", "or"]), operands=[predicate, other])
        yes, no = action("a"), action("b")
        if family == "branch_sequence":
            yes, no = sequence(action("a"), simple_action()), sequence(action("b"), simple_action())
        program = branch(predicate, yes, no)
    elif family == "counted_repeat":
        body = simple_action()
        if rng.randrange(2):
            body = sequence(body, action("a"))
        program = repeat(rng.randint(0, 5), body)
    elif family in {"while_counter", "until_counter"}:
        target = counter()
        start = numeric_value()
        if family == "while_counter":
            program = Statement(
                kind="while",
                condition=compare(target, start + rng.randint(0, 5), "lt"),
                body=action(target, "increment"),
            )
        else:
            program = Statement(
                kind="until",
                condition=compare(target, rng.randint(0, start), "le"),
                body=action(target, "decrement"),
            )
    elif family == "queue_until":
        target = queue()
        widget = next(widget for widget in widgets if widget.id == target)
        stop = rng.randrange(len(widget.items))
        for index, item in enumerate(widget.items):
            item.category = "blue" if index < stop else "red"
        program = Statement(
            kind="until",
            condition=compare(target, "red", field="category"),
            body=action(target, "advance"),
        )
    elif family in {"foreach", "foreach_if", "foreach_repeat_if"}:
        target = queue()
        body = action(target, rng.choice(["approve", "reject"]))
        if family == "foreach_if":
            predicate = compare(
                target, rng.choice(["red", "blue"]), rng.choice(["eq", "ne"]), "category"
            )
            if rng.randrange(2):
                predicate = Predicate(
                    kind=rng.choice(["and", "or"]),
                    operands=[predicate, compare(target, "open", field="status")],
                )
            body = branch(predicate, action(target, "approve"), action(target, "reject"))
        elif family == "foreach_repeat_if":
            predicate = compare(
                target, rng.choice(["red", "blue"]), rng.choice(["eq", "ne"]), "category"
            )
            repeated = branch(predicate, simple_action(exclude_b=True), action("b"))
            body = sequence(repeat(rng.randint(1, 3), repeated), body)
        program = Statement(kind="foreach", collection=target, body=body)
    elif family == "while_if":
        target = counter()
        flag = toggle()
        body = sequence(
            branch(compare(flag, bool(rng.randrange(2))), action("a"), action("b")),
            action(flag, "toggle"),
            action(target, "increment"),
        )
        program = Statement(
            kind="while",
            condition=compare(target, numeric_value() + rng.randint(0, 4), "lt"),
            body=body,
        )
    elif family == "if_loop":
        target = counter()
        left = repeat(rng.randint(0, 4), simple_action())
        right = Statement(
            kind=rng.choice(["until", "while"]),
            condition=compare(target, numeric_value() + rng.randint(0, 4), "ge"),
            body=action(target, "increment"),
        )
        if right.kind == "while":
            right.condition.operator = "lt"
        program = branch(condition(), left, right)
    elif family == "repeat_if":
        flag = toggle()
        body = sequence(
            branch(
                compare(flag, bool(rng.randrange(2))), simple_action(exclude_b=True), action("b")
            ),
            action(flag, "toggle"),
        )
        program = repeat(rng.randint(0, 5), body)
    elif family == "nested_repeat":
        inner = repeat(rng.randint(1, 3), simple_action())
        program = repeat(rng.randint(1, 3), sequence(inner, action("b")))
    elif family == "while_repeat":
        target = counter()
        body = sequence(
            repeat(rng.randint(1, 3), action(rng.choice(["a", "b"]))), action(target, "increment")
        )
        program = Statement(
            kind="while",
            condition=compare(target, numeric_value() + rng.randint(0, 3), "lt"),
            body=body,
        )
    elif family == "branch_nested_loop":
        left = repeat(rng.randint(1, 3), repeat(rng.randint(1, 3), simple_action(exclude_b=True)))
        right = repeat(rng.randint(1, 3), repeat(rng.randint(1, 3), action("b")))
        program = branch(condition(), left, right)
    elif family == "until_repeat_if":
        target = counter()
        flag = toggle()
        inner = sequence(
            branch(compare(flag, True), action("a"), action("b")), action(flag, "toggle")
        )
        body = sequence(repeat(rng.randint(1, 3), inner), action(target, "increment"))
        program = Statement(
            kind="until",
            condition=compare(target, numeric_value() + rng.randint(0, 3), "ge"),
            body=body,
        )
    else:
        raise ValueError(f"Unknown task family: {family}")

    def references(node: Statement) -> None:
        if node.action:
            used.add(node.action.target)
        if node.collection:
            used.add(node.collection)

        def predicates(predicate: Predicate) -> None:
            if predicate.target:
                used.add(predicate.target)
            for child in predicate.operands:
                predicates(child)

        if node.condition:
            predicates(node.condition)
        for child in [*node.steps, node.body, node.otherwise]:
            if child:
                references(child)

    references(program)
    # Retain one distractor button when possible, without changing semantics.
    used.add(rng.choice(["a", "b"]))
    selected = [widget for widget in widgets if widget.id in used]
    rng.shuffle(selected)
    return program, World(widgets=selected)


def generate_episode(
    seed: int, level: int, split: str = "train", family: str | None = None
) -> Episode:
    if level not in FAMILIES:
        raise ValueError("Level must be in 0..5")
    rng = random.Random(seed)
    family = family or rng.choice(list(FAMILIES[level]))
    if family not in FAMILIES[level]:
        raise ValueError(f"Family {family!r} is not available at L{level}")
    program, world = _instantiate(rng, level, family)
    profile = complexity_profile(program, world)
    identity = hashlib.sha256(f"scriptwob-v1:{seed}:{level}:{family}".encode()).hexdigest()[:16]
    episode = Episode(
        id=f"L{level}-{identity}",
        split=split,
        level=level,
        family=family,
        seed=seed,
        layout_seed=rng.randrange(2**32),
        program=program,
        world=world,
        instruction=render_instruction(program, world),
        complexity=profile,
        structural_id=structural_id(program),
        max_steps=256,
    )
    trace, engine = oracle_trace(episode)
    profile.reference_actions = len(trace)
    profile.branch_evaluations = engine.branch_evaluations
    profile.loop_iterations = engine.loop_iterations
    episode.max_steps = max(8, len(trace) + 4)
    return episode


def generate_manifest(
    train: int = 120,
    dev: int = 24,
    test: int = 24,
    seed: int = 0,
    levels: Sequence[int] | None = None,
    split_strategy: str = "seed",
) -> Manifest:
    """Counts are totals per split, balanced across the requested levels.

    Seed splits hold out exact tasks grouped by their language, layout and state.
    Holding out seed IDs alone would leak the finite atomic task space between
    splits. Exact repeats can occur within a split: more seeds do not promise
    more unique tasks.
    """
    levels = list(
        levels
        if levels is not None
        else (range(2, 6) if split_strategy == "structure" else range(6))
    )
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(level not in FAMILIES for level in levels)
    ):
        raise ValueError("Specify unique levels in 0..5")
    if split_strategy not in {"seed", "structure"}:
        raise ValueError("split_strategy must be seed or structure")
    if split_strategy == "structure" and any(level < 2 for level in levels):
        raise ValueError(
            "Structural splits require levels 2..5; use seed splits for atomic/sequence tasks"
        )
    if any(count < 0 for count in (train, dev, test)) or seed < 0:
        raise ValueError("Counts and seed must be nonnegative")
    rng = random.Random(seed)
    episodes: list[Episode] = []
    seen: set[str] = set()
    for split, count in (("train", train), ("dev", dev), ("test", test)):
        for index in range(count):
            level = levels[index % len(levels)]
            for _ in range(10000):
                episode = generate_episode(rng.randrange(2**63), level, split)
                if episode.id in seen:
                    continue
                assignment = (
                    structural_split(episode.program)
                    if split_strategy == "structure"
                    else instance_split(episode)
                )
                if assignment != split:
                    continue
                episodes.append(episode)
                seen.add(episode.id)
                break
            else:
                raise ValueError(f"Could not sample a valid L{level} {split} structure")
    return Manifest(seed=seed, split_strategy=split_strategy, episodes=episodes)


def validate_manifest(manifest: Manifest) -> Manifest:
    """Check external manifests semantically, including annotations and solvability."""
    instance_owners: dict[str, str] = {}
    for episode in manifest.episodes:
        validate_program(episode.program, episode.world)
        if episode.structural_id != structural_id(episode.program):
            raise ValueError(f"Structural fingerprint mismatch: {episode.id}")
        if (
            manifest.split_strategy == "structure"
            and structural_split(episode.program) != episode.split
        ):
            raise ValueError(f"Invalid structural split assignment: {episode.id}")
        trace, engine = oracle_trace(episode)
        profile = complexity_profile(episode.program, episode.world)
        profile.reference_actions = len(trace)
        profile.branch_evaluations = engine.branch_evaluations
        profile.loop_iterations = engine.loop_iterations
        if profile != episode.complexity:
            raise ValueError(f"Complexity annotations do not match program: {episode.id}")
        if episode.instruction != render_instruction(episode.program, episode.world):
            raise ValueError(f"Instruction differs from canonical semantics: {episode.id}")
        fingerprint = task_fingerprint(episode)
        owner = instance_owners.setdefault(fingerprint, episode.split)
        if owner != episode.split:
            raise ValueError(f"Identical task occurs in different splits: {episode.id}")
        if manifest.split_strategy == "seed" and instance_split(episode) != episode.split:
            raise ValueError(f"Invalid exact-instance split assignment: {episode.id}")
    return manifest


def read_manifest(path: str | Path) -> Manifest:
    return validate_manifest(Manifest.model_validate_json(Path(path).read_text()))


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n")
