"""Static, equivalent presentations of an episode's authored procedure.

The canonical instruction remains part of the task identity. Alternative styles
are presentation only: they use the program and widget names/kinds, never widget
values, queue contents, the reference trace, or a branch's evaluated truth.
"""

from __future__ import annotations

import json

from .schema import Episode, Manifest, Predicate, Statement, World

DEFAULT_PROMPT_STYLE = "natural"
PROMPT_STYLES = ("natural", "code")
PROMPT_STYLE_VERSIONS = {
    "natural": "scriptwob-natural-v1",
    "code": "scriptwob-code-v1",
}


def validate_prompt_style(style: str) -> str:
    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style {style!r}; choose from {', '.join(PROMPT_STYLES)}")
    return style


def instruction_version(style: str) -> str:
    return PROMPT_STYLE_VERSIONS[validate_prompt_style(style)]


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _predicate(condition: Predicate, labels: dict[str, tuple[str, str]]) -> str:
    if condition.kind == "not":
        return f"NOT ({_predicate(condition.operands[0], labels)})"
    if condition.kind in {"and", "or"}:
        joiner = f" {condition.kind.upper()} "
        return "(" + joiner.join(_predicate(child, labels) for child in condition.operands) + ")"

    label, kind = labels[condition.target]
    name = _quoted(label)
    if condition.field == "value":
        if (
            kind == "toggle" and type(condition.value) is bool
            and condition.operator in {"eq", "ne"}
        ):
            checked = condition.value if condition.operator == "eq" else not condition.value
            return f"checkbox {name} is {'checked' if checked else 'unchecked'}"
        subject = {
            "counter": f"counter {name}",
            "button": f"press count of button {name}",
            "status": f"status {name}",
            "toggle": f"checked state of checkbox {name}",
            "queue": f"value of queue {name}",
        }[kind]
    elif condition.field == "remaining":
        subject = f"remaining items in queue {name}"
    elif condition.field == "cursor":
        subject = f"position of queue {name}"
    else:
        subject = f"{condition.field} of current item in queue {name}"
    relation = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[
        condition.operator
    ]
    value = (
        str(condition.value).upper()
        if type(condition.value) is bool
        else json.dumps(condition.value, ensure_ascii=False)
    )
    return f"{subject} {relation} {value}"


def _code_instruction(program: Statement, world: World) -> str:
    # Copy only immutable presentation metadata into the renderer. In particular,
    # rendering predicates cannot inspect their truth in the initial/current page.
    labels = {widget.id: (widget.label, widget.kind) for widget in world.widgets}
    kinds: set[str] = set()
    item_predicates = False

    def inspect_predicate(condition: Predicate) -> None:
        nonlocal item_predicates
        if condition.kind == "compare" and condition.field in {
            "category", "status", "label", "processed"
        }:
            item_predicates = True
        for child in condition.operands:
            inspect_predicate(child)

    def render(node: Statement, depth: int = 0) -> list[str]:
        kinds.add(node.kind)
        prefix = "    " * depth
        if node.condition:
            inspect_predicate(node.condition)
        if node.kind == "action":
            name = _quoted(labels[node.action.target][0])
            operation = {
                "click": f"Click button {name} once.",
                "toggle": f"Toggle checkbox {name} once.",
                "increment": f"Click + Increase on counter {name} once.",
                "decrement": f"Click − Decrease on counter {name} once.",
                "advance": f"Click Next on queue {name} once.",
                "approve": f"Click Approve for the current item in queue {name} once.",
                "reject": f"Click Reject for the current item in queue {name} once.",
            }[node.action.op]
            return [prefix + operation]
        if node.kind == "sequence":
            return [line for child in node.steps for line in render(child, depth)]
        if node.kind == "if":
            return [
                prefix + f"IF {_predicate(node.condition, labels)}:",
                *render(node.body, depth + 1),
                prefix + "ELSE:",
                *render(node.otherwise, depth + 1),
                prefix + "END IF",
            ]
        if node.kind == "repeat":
            header, ending = f"REPEAT EXACTLY {node.count} TIMES:", "END REPEAT"
        elif node.kind in {"while", "until"}:
            header = f"{node.kind.upper()} {_predicate(node.condition, labels)}:"
            ending = f"END {node.kind.upper()}"
        elif node.kind == "foreach":
            name = _quoted(labels[node.collection][0])
            return [
                prefix + f"FOR EACH REMAINING ITEM IN queue {name}, IN DISPLAYED ORDER:",
                *render(node.body, depth + 1),
                "    " * (depth + 1) + f"Click Next on queue {name} once.",
                prefix + "END FOR EACH",
            ]
        else:
            raise ValueError(f"Unsupported statement kind: {node.kind}")
        return [prefix + header, *render(node.body, depth + 1), prefix + ending]

    lines = render(program)
    rules = ["Follow the script in order. Perform only the listed clicks."]
    if "if" in kinds:
        rules.append(
            "Read each IF condition when entering that IF; finish only the selected branch."
        )
    if kinds & {"if", "while", "until"}:
        rules.append("Testing a condition does not click or change any control.")
    if "while" in kinds:
        rules.append("WHILE: check before the first and every later iteration; stop when false.")
    if "until" in kinds:
        rules.append("UNTIL: check before the first and every later iteration; stop when true.")
    if "foreach" in kinds:
        rules.append(
            "FOR EACH: use the current item, finish its body, then click Next; stop at no items."
        )
    if item_predicates:
        rules.append(
            "If a queue has no current item, each comparison of an item field is FALSE, "
            "including !=."
        )
    return "\n".join([*rules, "", *lines, "Click Done."])


def render_episode_instruction(episode: Episode, style: str = DEFAULT_PROMPT_STYLE) -> str:
    """Render authored rules without changing task identity or solving any step."""
    validate_prompt_style(style)
    if style == "natural":
        return episode.instruction
    return _code_instruction(episode.program, episode.world)


def build_instruction_map(manifest: Manifest, style: str) -> dict[str, str]:
    """Map canonical observations to a presentation, rejecting ambiguous rewrites."""
    validate_prompt_style(style)
    result: dict[str, str] = {}
    for episode in manifest.episodes:
        rendered = render_episode_instruction(episode, style)
        previous = result.setdefault(episode.instruction, rendered)
        if previous != rendered:
            raise ValueError(
                f"Canonical instruction maps to conflicting {style!r} presentations "
                f"(episode {episode.id}); cannot rewrite observations unambiguously"
            )
    return result
