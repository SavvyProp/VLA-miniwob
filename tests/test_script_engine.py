import pytest
from pydantic import ValidationError

from scriptwob.engine import Engine, evaluate, oracle_trace
from scriptwob.generation import action, branch, compare, generate_episode, repeat, sequence
from scriptwob.schema import Episode, Item, Predicate, SemanticAction, Statement, Widget, World


def episode(program, widgets, max_steps=64):
    return Episode(
        id="test-episode",
        split="train",
        level=0,
        family="unit",
        seed=0,
        layout_seed=0,
        program=program,
        world=World(widgets=widgets),
        instruction="Unit-test procedure",
        structural_id="unit",
        max_steps=max_steps,
    )


def button(target="a"):
    return Widget(id=target, label=target.title(), kind="button")


def test_sequence_repetition_and_explicit_completion():
    spec = episode(sequence(repeat(3, action("a")), action("b")), [button(), button("b")])
    engine = Engine(spec)
    for _ in range(3):
        assert not engine.step(SemanticAction(target="a", op="click")).done
    assert engine.expected_action().target == "b"
    engine.step(SemanticAction(target="b", op="click"))
    assert not engine.done
    assert engine.expected_action() == SemanticAction(target="done", op="done")
    assert engine.step(engine.expected_action()).success
    assert spec.world.widget("a").value == 0  # episode initial world is immutable to execution


def test_wrong_branch_is_terminal_even_when_controls_would_allow_repair():
    spec = episode(
        branch(compare("flag", True), action("a"), action("b")),
        [button(), button("b"), Widget(id="flag", label="Flag", kind="toggle", value=True)],
    )
    engine = Engine(spec)
    result = engine.step(SemanticAction(target="b", op="click"))
    assert result.done and not result.success and result.error == "wrong_action"
    assert engine.world.widget("b").value == 1  # physical effect remains visible
    with pytest.raises(ValueError, match="terminal"):
        engine.step(SemanticAction(target="a", op="click"))


@pytest.mark.parametrize(
    "kind,operator,start,limit,expected",
    [
        ("while", "lt", 0, 3, 3),
        ("while", "lt", 3, 3, 0),
        ("until", "ge", 0, 3, 3),
        ("until", "ge", 3, 3, 0),
    ],
)
def test_loop_conditions_are_rechecked_before_each_iteration(
    kind, operator, start, limit, expected
):
    program = Statement(
        kind=kind, condition=compare("count", limit, operator), body=action("count", "increment")
    )
    spec = episode(program, [Widget(id="count", label="Count", kind="counter", value=start)])
    trace, engine = oracle_trace(spec)
    assert len(trace) == expected + 1
    assert engine.world.widget("count").value == limit
    assert engine.loop_iterations == [expected]


def test_if_inside_loop_uses_current_state_not_initial_snapshot():
    body = sequence(
        branch(compare("flag", True), action("a"), action("b")), action("flag", "toggle")
    )
    spec = episode(
        repeat(3, body),
        [button(), button("b"), Widget(id="flag", label="Flag", kind="toggle", value=True)],
    )
    trace, _ = oracle_trace(spec)
    assert [event.target for event in trace if event.op == "click"] == ["a", "b", "a"]


def test_foreach_processes_ordered_items_and_requires_next():
    queue = Widget(
        id="queue",
        label="Queue",
        kind="queue",
        items=[Item(id="x", label="X", category="red"), Item(id="y", label="Y", category="blue")],
    )
    body = branch(
        compare("queue", "red", field="category"),
        action("queue", "approve"),
        action("queue", "reject"),
    )
    spec = episode(Statement(kind="foreach", collection="queue", body=body), [queue])
    trace, engine = oracle_trace(spec)
    assert [event.op for event in trace] == ["approve", "advance", "reject", "advance", "done"]
    assert [item.result for item in engine.world.widget("queue").items] == ["approved", "rejected"]
    assert engine.loop_iterations == [2]


def test_extra_iteration_and_premature_done_fail():
    spec = episode(repeat(2, action("a")), [button()])
    early = Engine(spec)
    assert early.step(SemanticAction(target="done", op="done")).error == "premature_done"
    late = Engine(spec)
    late.step(late.expected_action())
    late.step(late.expected_action())
    assert late.step(SemanticAction(target="a", op="click")).error == "extra_action"


def test_invalid_dead_branch_rejected_before_execution():
    program = branch(compare("flag", True), action("a"), action("missing"))
    spec = episode(program, [button(), Widget(id="flag", label="Flag", kind="toggle", value=True)])
    with pytest.raises(ValueError, match="Unknown widget"):
        Engine(spec)


def test_nonterminating_loop_is_bounded():
    spec = episode(
        Statement(kind="while", condition=compare("a", 0, "ge"), body=action("a")),
        [button()],
        max_steps=5,
    )
    with pytest.raises(ValueError, match="step_budget_exhausted"):
        oracle_trace(spec)


def test_no_action_nonterminating_loop_is_bounded():
    program = Statement(kind="while", condition=compare("a", 0), body=repeat(0, action("a")))
    engine = Engine(episode(program, [button()]))
    assert engine.done and not engine.success
    assert "invalid_program" in engine.error


def test_exhausted_queue_comparison_and_boolean_predicates():
    world = World(widgets=[Widget(id="queue", label="Queue", kind="queue")])
    assert not evaluate(compare("queue", "red", "ne", "category"), world)
    assert evaluate(Predicate(kind="not", operands=[compare("queue", 1, field="remaining")]), world)


def test_bad_ast_shape_and_duplicate_widgets_rejected():
    with pytest.raises(ValidationError):
        Statement(kind="repeat", body=action("a"))
    with pytest.raises(ValidationError):
        Statement(kind="action", action=SemanticAction(target="a", op="click"), count=2)
    with pytest.raises(ValidationError):
        World(widgets=[button(), button()])


def test_public_world_never_contains_evaluator_annotations():
    spec = generate_episode(0, 5)
    public = Engine(spec).world.public_dict()
    assert set(public) == {"widgets", "events"}
    assert not ({"program", "structural_id", "complexity", "expected_action"} & set(public))
