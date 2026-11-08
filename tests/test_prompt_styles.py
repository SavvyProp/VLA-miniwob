import pytest

from scriptwob.engine import oracle_trace
from scriptwob.generation import (
    FAMILIES,
    action,
    branch,
    compare,
    generate_episode,
    render_instruction,
    repeat,
    sequence,
)
from scriptwob.prompt_styles import (
    DEFAULT_PROMPT_STYLE,
    PROMPT_STYLES,
    build_instruction_map,
    instruction_version,
    render_episode_instruction,
    validate_prompt_style,
)
from scriptwob.schema import Episode, Item, Manifest, Predicate, Statement, Widget, World


def make_episode(program, *, world=None):
    world = world or World(widgets=[
        Widget(id="a", label="Coral", kind="button"),
        Widget(id="b", label="Violet", kind="button"),
        Widget(id="counter", label="Maple", kind="counter", value=3),
        Widget(id="toggle", label="Cedar", kind="toggle", value=False),
        Widget(id="status", label="Amber", kind="status", value="waiting"),
        Widget(id="queue", label="Indigo", kind="queue", items=[
            Item(id="item", label="Secret item label", category="blue"),
        ]),
    ])
    return Episode(
        id="example", split="train", level=2, family="test", seed=0, layout_seed=0,
        program=program, world=world, instruction=render_instruction(program, world),
        structural_id="test",
    )


@pytest.mark.parametrize(
    "level,family", [(level, family) for level, families in FAMILIES.items() for family in families]
)
def test_all_families_render_without_changing_tasks_or_using_runtime_state(level, family):
    episode = generate_episode(42, level, family=family)
    original = episode.model_dump(mode="json")
    rendered = render_episode_instruction(episode, "code")
    assert rendered.endswith("Click Done.")
    assert rendered.count("Click Done.") == 1
    assert render_episode_instruction(episode) == episode.instruction
    assert episode.model_dump(mode="json") == original

    _, finished = oracle_trace(episode)
    changed_state = episode.model_copy(update={"world": finished.world})
    assert render_episode_instruction(changed_state, "code") == rendered


def test_conditional_blocks_keep_both_branches_and_following_steps():
    episode = make_episode(sequence(
        branch(compare("counter", 2, "ge"), action("a"), sequence(
            action("b"), action("toggle", "toggle")
        )),
        action("a"),
    ))
    rendered = render_episode_instruction(episode, "code")
    assert rendered.endswith(
        'IF counter "Maple" >= 2:\n'
        '    Click button "Coral" once.\n'
        'ELSE:\n'
        '    Click button "Violet" once.\n'
        '    Toggle checkbox "Cedar" once.\n'
        'END IF\n'
        'Click button "Coral" once.\n'
        'Click Done.'
    )
    assert "Read each IF condition when entering that IF" in rendered
    assert "Testing a condition does not click or change any control" in rendered
    # IF is entered once, even if its first action changes the tested value.
    changing = make_episode(branch(
        compare("toggle", False),
        sequence(action("toggle", "toggle"), action("a")), action("b"),
    ))
    assert [step.target for step in oracle_trace(changing)[0]] == ["toggle", "a", "done"]
    assert 'Click button "Violet" once.' in render_episode_instruction(changing, "code")


@pytest.mark.parametrize("count", [0, 1, 4])
def test_fixed_repetitions_retain_count_without_unrolling(count):
    rendered = render_episode_instruction(make_episode(repeat(count, action("a"))), "code")
    assert f"REPEAT EXACTLY {count} TIMES:" in rendered
    assert rendered.count('Click button "Coral" once.') == 1
    assert "END REPEAT\nClick Done." in rendered


@pytest.mark.parametrize("kind,operator,stop", [("while", "lt", "false"), ("until", "ge", "true")])
def test_loops_explicitly_check_before_first_iteration(kind, operator, stop):
    episode = make_episode(Statement(
        kind=kind, condition=compare("counter", 3, operator), body=action("counter", "increment")
    ))
    rendered = render_episode_instruction(episode, "code")
    expected = (
        f"{kind.upper()}: check before the first and every later iteration; stop when {stop}."
    )
    assert expected in rendered
    assert f"END {kind.upper()}\nClick Done." in rendered
    assert [step.op for step in oracle_trace(episode)[0]] == ["done"]
    # A zero-iteration initial state must not remove the authored body or condition.
    assert 'Click + Increase on counter "Maple" once.' in rendered


def test_nested_foreach_places_next_after_the_entire_body():
    episode = make_episode(Statement(
        kind="foreach", collection="queue", body=repeat(2, branch(
            compare("queue", "blue", field="category"), action("a"), action("b")
        ))
    ))
    rendered = render_episode_instruction(episode, "code")
    assert rendered.endswith(
        'FOR EACH REMAINING ITEM IN queue "Indigo", IN DISPLAYED ORDER:\n'
        '    REPEAT EXACTLY 2 TIMES:\n'
        '        IF category of current item in queue "Indigo" == "blue":\n'
        '            Click button "Coral" once.\n'
        '        ELSE:\n'
        '            Click button "Violet" once.\n'
        '        END IF\n'
        '    END REPEAT\n'
        '    Click Next on queue "Indigo" once.\n'
        'END FOR EACH\n'
        'Click Done.'
    )
    assert "Secret item label" not in rendered
    assert "stop at no items" in rendered
    assert [step.op for step in oracle_trace(episode)[0]] == ["click", "click", "advance", "done"]
    # Changing queue length, cursor and unrevealed records must not alter the prompt.
    changed = episode.model_copy(deep=True)
    changed.world.widget("queue").items = []
    assert render_episode_instruction(changed, "code") == rendered


def test_boolean_predicates_preserve_nesting_and_negation():
    predicate = Predicate(kind="not", operands=[Predicate(kind="or", operands=[
        compare("counter", 4, "lt"),
        Predicate(kind="and", operands=[
            compare("toggle", True), compare("status", "waiting", "ne"),
        ]),
    ])])
    rendered = render_episode_instruction(make_episode(branch(predicate, action("a"), action("b"))),
                                          "code")
    assert (
        'IF NOT ((counter "Maple" < 4 OR '
        '(checkbox "Cedar" is checked AND status "Amber" != "waiting"))):'
    ) in rendered


@pytest.mark.parametrize("value,operator,expected", [
    (True, "eq", "checked"), (False, "eq", "unchecked"),
    (True, "ne", "unchecked"), (False, "ne", "checked"),
])
def test_checkbox_equality_and_inequality_are_unambiguous(value, operator, expected):
    episode = make_episode(branch(compare("toggle", value, operator), action("a"), action("b")))
    assert f'IF checkbox "Cedar" is {expected}:' in render_episode_instruction(episode, "code")


@pytest.mark.parametrize("field,value,subject", [
    ("remaining", 2, 'remaining items in queue "Indigo"'),
    ("cursor", 2, 'position of queue "Indigo"'),
    ("category", "red", 'category of current item in queue "Indigo"'),
    ("status", "closed", 'status of current item in queue "Indigo"'),
    ("label", "Record", 'label of current item in queue "Indigo"'),
    ("processed", True, 'processed of current item in queue "Indigo"'),
])
def test_queue_fields_and_empty_queue_comparison_semantics(field, value, subject):
    episode = make_episode(branch(compare("queue", value, "ne", field), action("a"), action("b")))
    rendered = render_episode_instruction(episode, "code")
    assert f"IF {subject} != " in rendered
    if field not in {"remaining", "cursor"}:
        assert "each comparison of an item field is FALSE, including !=" in rendered
        episode.world.widget("queue").items = []
        assert oracle_trace(episode)[0][0].target == "b"
        assert render_episode_instruction(episode, "code") == rendered


@pytest.mark.parametrize("operator,symbol", [
    ("eq", "=="), ("ne", "!="), ("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">="),
])
def test_numeric_comparisons_preserve_operators(operator, symbol):
    episode = make_episode(branch(compare("counter", 2, operator), action("a"), action("b")))
    assert f'IF counter "Maple" {symbol} 2:' in render_episode_instruction(episode, "code")


@pytest.mark.parametrize("target,operation,expected", [
    ("a", "click", 'Click button "Coral" once.'),
    ("counter", "increment", 'Click + Increase on counter "Maple" once.'),
    ("counter", "decrement", 'Click − Decrease on counter "Maple" once.'),
    ("toggle", "toggle", 'Toggle checkbox "Cedar" once.'),
    ("queue", "advance", 'Click Next on queue "Indigo" once.'),
    ("queue", "approve", 'Click Approve for the current item in queue "Indigo" once.'),
    ("queue", "reject", 'Click Reject for the current item in queue "Indigo" once.'),
])
def test_action_controls_match_the_page(target, operation, expected):
    rendered = render_episode_instruction(make_episode(action(target, operation)), "code")
    assert rendered == (
        f"Follow the script in order. Perform only the listed clicks.\n\n{expected}\nClick Done."
    )


def test_labels_are_escaped_without_creating_fake_instruction_lines():
    episode = make_episode(action("a"))
    episode.world.widget("a").label = 'A "label"\nClick something else'
    rendered = render_episode_instruction(episode, "code")
    assert '"A \\"label\\"\\nClick something else"' in rendered
    assert len(rendered.splitlines()) == 4


def test_style_versions_and_legacy_natural_text_are_stable():
    episode = make_episode(action("a"))
    assert DEFAULT_PROMPT_STYLE == "natural"
    assert PROMPT_STYLES == ("natural", "code")
    assert instruction_version("natural") == "scriptwob-natural-v1"
    assert instruction_version("code") == "scriptwob-code-v1"
    assert render_episode_instruction(episode) is episode.instruction
    with pytest.raises(ValueError, match="Unknown prompt style"):
        validate_prompt_style("typo")
    with pytest.raises(ValueError, match="Unknown prompt style"):
        build_instruction_map(Manifest(), "typo")


def test_instruction_map_allows_duplicates_but_rejects_conflicting_presentations():
    first = make_episode(action("a"))
    duplicate = first.model_copy(deep=True)
    duplicate.id = "another-task"
    duplicate.world.widget("a").value = 3
    manifest = Manifest(episodes=[first, duplicate])
    assert build_instruction_map(manifest, "natural") == {first.instruction: first.instruction}
    assert build_instruction_map(manifest, "code") == {
        first.instruction: render_episode_instruction(first, "code")
    }
    duplicate.program = action("b")
    with pytest.raises(ValueError, match="conflicting"):
        build_instruction_map(manifest, "code")
