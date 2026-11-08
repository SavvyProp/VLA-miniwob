"""Analytic checks of the offline entropy analysis, independent of plot data."""

import importlib.util
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from scriptwob.generation import action, branch, compare, repeat, sequence
from scriptwob.schema import Episode, Item, Predicate, Statement, Widget, World

# The offline script deliberately lives outside the installed runtime package.
# Loading by file works with both `pytest` and `python -m pytest` entry points.
_spec = importlib.util.spec_from_file_location(
    "_script_entropy_tests", Path(__file__).resolve().parents[1] / "scripts" / "script_entropy.py"
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
EntropyLimitError = _module.EntropyLimitError
evaluate_entropy = _module.evaluate_entropy
trace_world = _module.trace_world


def episode(program, widgets, max_steps=64):
    return Episode(
        id="entropy-test", split="test", level=0, family="unit", seed=0, layout_seed=0,
        program=program, world=World(widgets=widgets), instruction="Test program",
        structural_id="unit", max_steps=max_steps,
    )


def button(name="a"):
    return Widget(id=name, label=name, kind="button")


def toggle(name="flag"):
    return Widget(id=name, label=name, kind="toggle", value=True)


def counter(value=0):
    return Widget(id="count", label="Count", kind="counter", value=value, minimum=0)


def queue(length=1, name="queue"):
    return Widget(
        id=name, label=name.title(), kind="queue",
        items=[Item(id=f"record{i}", label=f"Record {i + 1}", category="red")
               for i in range(length)],
    )


def classify_queue(name="queue"):
    return Statement(kind="foreach", collection=name, body=branch(
        compare(name, "red", field="category"),
        action(name, "approve"), action(name, "reject"),
    ))


def test_fixed_repetition_has_zero_entropy():
    summary, paths = evaluate_entropy(episode(repeat(10, action("a")), [button()]))
    assert summary["entropy_bits"] == 0
    assert summary["path_count"] == 1
    assert summary["min_required_actions"] == summary["max_required_actions"] == 11
    assert paths[0]["probability"]["exact"] == "1"


def test_balanced_conditional_has_one_bit():
    summary, paths = evaluate_entropy(episode(
        branch(compare("flag", True), action("a"), action("b")),
        [button(), button("b"), toggle()],
    ))
    assert summary["entropy_bits"] == 1
    assert summary["original_surprisal_bits"] == 1
    assert [path["probability"]["exact"] for path in paths] == ["1/2", "1/2"]


def test_identical_branch_actions_are_still_distinct_control_paths():
    shared_action = action("a")
    spec = episode(
        branch(compare("flag", True), shared_action, shared_action), [button(), toggle()]
    )
    summary, paths = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 1
    assert len(paths) == 2
    assert all(path["required_actions"] == 2 for path in paths)


@pytest.mark.parametrize("change_flag", [False, True])
def test_correlated_checks_do_not_add_independent_bits(change_flag):
    check = branch(compare("flag", True), action("a"), action("b"))
    body = sequence(check, action("flag", "toggle")) if change_flag else check
    summary, _ = evaluate_entropy(episode(repeat(10, body), [button(), button("b"), toggle()]))
    assert summary["entropy_bits"] == 1
    assert summary["path_count"] == 2


def test_independent_checks_add_two_bits():
    spec = episode(sequence(
        branch(compare("flag", True), action("a"), action("b")),
        branch(compare("other", True), action("a"), action("b")),
    ), [button(), button("b"), toggle(), toggle("other")])
    summary, _ = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 2
    assert summary["path_count"] == 4


def test_nested_conditional_counts_only_reached_decisions():
    inner = branch(compare("other", True), action("a"), action("b"))
    spec = episode(
        branch(compare("flag", True), inner, action("a")),
        [button(), button("b"), toggle(), toggle("other")],
    )
    summary, paths = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 1.5
    assert sorted(path["probability"]["value"] for path in paths) == [0.25, 0.25, 0.5]


def test_boolean_short_circuit_is_not_a_separate_path_decision():
    condition = Predicate(kind="or", operands=[compare("flag", True), compare("other", True)])
    spec = episode(
        branch(condition, action("a"), action("b")),
        [button(), button("b"), toggle(), toggle("other")],
    )
    summary, _ = evaluate_entropy(spec)
    assert summary["path_count"] == 2
    assert summary["entropy_bits"] == pytest.approx(-0.75 * math.log2(0.75) + 0.5)


def test_uniform_queue_length_entropy_and_unused_fields_marginalized():
    spec = episode(
        Statement(kind="foreach", collection="queue", body=action("queue", "approve")),
        [queue()],
    )
    summary, paths = evaluate_entropy(spec, prior="reference")
    assert summary["entropy_bits"] == pytest.approx(math.log2(5))
    assert summary["prior_version"] == "scriptwob-independent-initial-state-v1"
    assert summary["enumerated_state_count"] == 5
    assert summary["marginalized_queue_fields"] == {"queue": ["category", "status"]}
    assert all(path["probability"]["exact"] == "1/5" for path in paths)


def test_queue_length_is_weighted_before_record_assignments():
    body = branch(
        compare("queue", "red", field="category"),
        action("queue", "approve"), action("queue", "reject"),
    )
    spec = episode(Statement(kind="foreach", collection="queue", body=body), [queue()])
    summary, paths = evaluate_entropy(spec, prior="reference")
    assert summary["enumerated_state_count"] == 62
    assert summary["entropy_bits"] == pytest.approx(math.log2(5) + 3)
    for length in range(1, 6):
        mass = sum(
            Fraction(path["probability"]["exact"])
            for path in paths if path["required_actions"] == 2 * length + 1
        )
        assert mass == Fraction(1, 5)


def test_invalid_counter_states_are_conditioned_out_with_mass_reported():
    spec = episode(action("count", "decrement"), [counter(value=1)])
    summary, paths = evaluate_entropy(spec)
    assert summary["valid_prior_mass_exact"] == "3/4"
    assert summary["rejected_prior_mass_exact"] == "1/4"
    assert summary["rejected_state_count"] == 1
    assert summary["entropy_bits"] == 0
    assert paths[0]["probability"]["exact"] == "1"
    assert "unavailable action count:decrement" in next(iter(summary["rejection_reasons"]))


def test_invalid_queue_states_condition_globally_across_lengths():
    spec = episode(Statement(
        kind="until", condition=compare("queue", "red", field="category"),
        body=action("queue", "advance"),
    ), [queue()])
    summary, paths = evaluate_entropy(spec, prior="reference")
    assert summary["rejected_prior_mass_exact"] == "31/160"
    assert summary["valid_prior_mass_exact"] == "129/160"
    masses = {
        path["required_actions"]: Fraction(path["probability"]["exact"]) for path in paths
    }
    prior_masses = [
        Fraction(1, 2), Fraction(1, 5), Fraction(3, 40), Fraction(1, 40), Fraction(1, 160)
    ]
    assert masses == {i + 1: mass / Fraction(129, 160) for i, mass in enumerate(prior_masses)}


def test_counterfactuals_do_not_use_the_original_episode_budget():
    spec = episode(Statement(
        kind="until", condition=compare("count", 3, "ge"), body=action("count", "increment")
    ), [counter(value=2)], max_steps=2)
    summary, _ = evaluate_entropy(spec)
    assert summary["original_required_actions"] == 2
    assert summary["max_required_actions"] == 4
    assert summary["entropy_bits"] == 2
    assert spec.max_steps == 2
    assert spec.world.widget("count").value == 2


def test_limits_raise_instead_of_conditioning_on_truncation():
    spec = episode(Statement(
        kind="until", condition=compare("count", 3, "ge"), body=action("count", "increment")
    ), [counter(value=3)])
    with pytest.raises(EntropyLimitError, match="step_budget_exhausted"):
        evaluate_entropy(spec, max_steps=2)
    with pytest.raises(EntropyLimitError, match="exceed max_states"):
        evaluate_entropy(spec, max_states=2)
    endless = episode(Statement(
        kind="while", condition=compare("a", 0), body=repeat(0, action("a"))
    ), [button()])
    with pytest.raises(EntropyLimitError, match="1000 iterations"):
        trace_world(endless)


@pytest.mark.parametrize("length", [1, 5])
def test_observed_queue_length_conditions_entropy_and_choice_cost(length):
    spec = episode(classify_queue(), [queue(length), button()])
    original = spec.model_dump()
    summary, paths = evaluate_entropy(spec)
    assert summary["prior"] == "observed-workload"
    assert summary["prior_version"] == _module.PRIOR_VERSION
    assert summary["entropy_bits"] == length
    assert summary["path_count"] == summary["enumerated_state_count"] == 2 ** length
    assert all(path["required_actions"] == 2 * length + 1 for path in paths)
    assert summary["original_queue_lengths"] == {"queue": length}
    assert summary["original_queue_remaining"] == {"queue": length}
    assert summary["original_branch_evaluations"] == length
    assert summary["original_enabled_action_counts"] == [5, 3] * length + [2]
    assert summary["original_choice_bits"] == pytest.approx(length * math.log2(15) + 1)
    expected_probability = Fraction(1, 2 * 15 ** length)
    assert summary["original_uniform_button_success_probability_exact"] == str(expected_probability)
    assert summary["original_uniform_button_success_probability"] == float(expected_probability)
    assert summary["original_wrong_button_opportunities"] == 6 * length + 1
    assert spec.model_dump() == original


@pytest.mark.parametrize("length", [0, 1, 5])
def test_fixed_queue_foreach_without_branch_has_zero_entropy(length):
    spec = episode(
        Statement(kind="foreach", collection="queue", body=action("queue", "approve")),
        [queue(length)],
    )
    summary, paths = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 0
    assert summary["enumerated_state_count"] == 1
    assert paths[0]["required_actions"] == 2 * length + 1


@pytest.mark.parametrize("cursor", [1, 3])
def test_partially_consumed_or_exhausted_queue_keeps_progress(cursor):
    widget = queue(3)
    widget.cursor = cursor
    for item in widget.items[:cursor]:
        item.processed = True
        item.result = "approved"
    spec = episode(classify_queue(), [widget])
    original = spec.model_dump()
    summary, paths = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 3 - cursor
    assert summary["original_queue_lengths"] == {"queue": 3}
    assert summary["original_queue_remaining"] == {"queue": 3 - cursor}
    assert summary["path_count"] == 2 ** (3 - cursor)
    assert all(path["required_actions"] == 2 * (3 - cursor) + 1 for path in paths)
    assert spec.model_dump() == original


def test_observed_prior_preserves_record_metadata_and_unreferenced_fields():
    widget = queue(3)
    widget.cursor = 1
    widget.value = 9
    widget.items[0].category = "purple"
    widget.items[0].status = "archived"
    widget.items[0].processed = True
    widget.items[0].result = "approved"
    widget.items[1].id = "custom-id"
    widget.items[1].label = "Custom label"
    widget.items[1].status = "open"
    widget.items[2].status = "closed"
    widget.items[2].processed = True
    widget.items[2].result = "rejected"
    original = widget.model_dump()
    options = _module._widget_options(widget, {"category"}, prior="observed-workload")
    assert len(options) == 4
    assert sum(mass for _, mass in options) == 1
    for candidate, mass in options:
        assert mass == Fraction(1, 4)
        assert candidate.cursor == widget.cursor
        assert candidate.value == widget.value
        assert candidate.items[0] == widget.items[0]
        for new_item, old_item in zip(candidate.items[1:], widget.items[1:], strict=True):
            assert new_item.model_dump(exclude={"category"}) == old_item.model_dump(
                exclude={"category"}
            )
    assert widget.model_dump() == original


def test_multiple_queues_contribute_independent_remaining_records():
    first, second = queue(2), queue(3, "other")
    second.cursor = 1
    spec = episode(sequence(classify_queue(), classify_queue("other")), [first, second])
    summary, _ = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 4
    assert summary["path_count"] == summary["enumerated_state_count"] == 16
    assert summary["original_queue_lengths"] == {"queue": 2, "other": 3}
    assert summary["original_queue_remaining"] == {"queue": 2, "other": 2}


def test_category_and_status_bits_are_jointly_enumerated():
    body = branch(
        compare("queue", "red", field="category"),
        branch(compare("queue", "open", field="status"),
               action("queue", "approve"), action("queue", "reject")),
        action("queue", "reject"),
    )
    summary, _ = evaluate_entropy(episode(
        Statement(kind="foreach", collection="queue", body=body), [queue(2)],
    ))
    assert summary["entropy_bits"] == 3
    assert summary["enumerated_state_count"] == 16
    assert summary["path_count"] == 9


def test_queue_conditioning_retains_labels_and_processed_status_in_execution():
    widget = queue(2)
    widget.items[0].label = "Special"
    widget.items[1].processed = True
    widget.items[1].result = "rejected"
    spec = episode(Statement(kind="foreach", collection="queue", body=branch(
        compare("queue", "Special", field="label"), action("queue", "approve"),
        branch(compare("queue", True, field="processed"),
               repeat(0, action("queue", "approve")), action("queue", "reject")),
    )), [widget])
    summary, paths = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 0
    assert summary["rejected_prior_mass_exact"] == "0"
    assert paths[0]["required_actions"] == 4


def test_invalid_fixed_queue_worlds_report_conditioned_mass():
    spec = episode(Statement(
        kind="until", condition=compare("queue", "red", field="category"),
        body=action("queue", "advance"),
    ), [queue()])
    summary, paths = evaluate_entropy(spec)
    assert summary["entropy_bits"] == 0
    assert summary["valid_prior_mass_exact"] == "1/2"
    assert summary["rejected_prior_mass_exact"] == "1/2"
    assert paths[0]["probability"]["exact"] == "1"


def test_option_limit_is_checked_before_materializing_queue_assignments(monkeypatch):
    spec = episode(classify_queue(), [queue(12)])

    def should_not_generate_options(*args, **kwargs):
        raise AssertionError("Must reject before generating options")

    monkeypatch.setattr(_module, "_widget_options", should_not_generate_options)
    with pytest.raises(EntropyLimitError, match="4096 states exceed"):
        evaluate_entropy(spec, max_states=1000)


def test_unknown_prior_and_invalid_state_limit_raise():
    spec = episode(action("a"), [button()])
    with pytest.raises(ValueError, match="Unknown prior"):
        evaluate_entropy(spec, prior="unsupported")
    with pytest.raises(ValueError, match="max_states must be positive"):
        evaluate_entropy(spec, max_states=0)
