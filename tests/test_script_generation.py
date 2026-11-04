import pytest

from scriptwob.engine import oracle_trace
from scriptwob.generation import (
    FAMILIES,
    action,
    compare,
    generate_episode,
    generate_manifest,
    instance_split,
    read_manifest,
    render_instruction,
    repeat,
    structural_id,
    task_fingerprint,
    validate_manifest,
    write_manifest,
)
from scriptwob.schema import Statement


@pytest.mark.parametrize(
    "level,family", [(level, family) for level, families in FAMILIES.items() for family in families]
)
def test_every_family_generates_solvable_bounded_tasks(level, family):
    for seed in range(20):
        episode = generate_episode(seed, level, family=family)
        trace, engine = oracle_trace(episode)
        assert engine.success
        assert episode.complexity.reference_actions == len(trace) <= episode.max_steps
        assert len(episode.world.widgets) <= 6
        assert all(len(widget.items) <= 5 for widget in episode.world.widgets)
        assert episode.instruction == render_instruction(episode.program, episode.world)


def test_seed_reproducibility_and_manifest_roundtrip(tmp_path):
    manifest = generate_manifest(train=18, dev=6, test=6, seed=57)
    assert manifest == generate_manifest(train=18, dev=6, test=6, seed=57)
    assert manifest.digest != generate_manifest(train=18, dev=6, test=6, seed=58).digest
    path = tmp_path / "tasks.json"
    write_manifest(path, manifest)
    assert read_manifest(path).digest == manifest.digest
    assert len(manifest.episodes) == 30
    for split in ["train", "dev", "test"]:
        counts = [
            sum(episode.level == level and episode.split == split for episode in manifest.episodes)
            for level in range(6)
        ]
        assert max(counts) - min(counts) <= 1


def test_structural_split_disjointness_and_not_literal_holdout():
    manifest = generate_manifest(train=64, dev=32, test=32, seed=7, split_strategy="structure")
    structures = {
        split: {episode.structural_id for episode in manifest.episodes if episode.split == split}
        for split in ["train", "dev", "test"]
    }
    assert not structures["train"] & structures["test"]
    assert not structures["train"] & structures["dev"]
    assert not structures["dev"] & structures["test"]
    assert structural_id(repeat(2, action("a"))) == structural_id(repeat(8, action("renamed")))
    assert structural_id(repeat(2, action("a"))) != structural_id(repeat(2, action("a", "toggle")))
    assert {episode.level for episode in manifest.episodes} == {2, 3, 4, 5}
    validate_manifest(manifest)


def test_structure_strategy_rejects_insufficient_low_level_space():
    with pytest.raises(ValueError, match="levels 2..5"):
        generate_manifest(levels=[0, 1], split_strategy="structure")


def test_generation_covers_zero_one_many_iterations_and_both_branches():
    iterations = set()
    branches = set()
    for seed in range(80):
        repeated = generate_episode(seed, 3, family="counted_repeat")
        iterations.add(repeated.complexity.loop_iterations[0])
        conditional = generate_episode(seed, 2, family="visible_branch")
        trace, _ = oracle_trace(conditional)
        branches.add(trace[0].target)
    assert {0, 1, 3} <= iterations
    assert branches == {"a", "b"}


def test_external_manifest_annotations_and_instruction_are_validated():
    manifest = generate_manifest(train=1, dev=0, test=0)
    manifest.episodes[0].instruction = "Ignore the actual procedure."
    with pytest.raises(ValueError, match="Instruction"):
        validate_manifest(manifest)
    manifest = generate_manifest(train=1, dev=0, test=0)
    manifest.episodes[0].complexity.reference_actions += 1
    with pytest.raises(ValueError, match="Complexity"):
        validate_manifest(manifest)


def test_structural_fingerprint_distinguishes_predicate_operators():
    def loop(operation):
        return Statement(
            kind="while",
            condition=compare("count", 3, operation),
            body=action("count", "increment"),
        )

    assert structural_id(loop("lt")) != structural_id(loop("le"))


def test_generated_branches_differ_and_boolean_terms_are_independent():
    def walk(node):
        if node.kind == "if":
            assert node.body != node.otherwise
        if node.condition and node.condition.kind in {"and", "or"}:
            terms = [(term.target, term.field) for term in node.condition.operands]
            assert len(set(terms)) == len(terms)
        for child in [*node.steps, node.body, node.otherwise]:
            if child:
                walk(child)

    for level in (2, 4, 5):
        for family in FAMILIES[level]:
            for seed in range(50):
                walk(generate_episode(seed, level, family=family).program)


def test_seed_splits_group_exact_tasks_not_just_seed_identifiers():
    owners = {}
    for seed in (2026, 2027):
        manifest = generate_manifest(train=360, dev=120, test=120, seed=seed)
        for episode in manifest.episodes:
            fingerprint = task_fingerprint(episode)
            assert owners.setdefault(fingerprint, episode.split) == episode.split
            assert instance_split(episode) == episode.split
        validate_manifest(manifest)


def test_task_identity_ignores_internal_ids_but_keeps_future_records():
    original = generate_episode(42, 0, family="single_button")
    renamed = original.model_copy(deep=True)
    renamed.id = "renamed-episode"
    renamed.seed += 1
    for widget in renamed.world.widgets:
        widget.id = "renamed_" + widget.id
    assert task_fingerprint(original) == task_fingerprint(renamed)

    queued = next(
        candidate
        for seed in range(30)
        if len(
            (candidate := generate_episode(seed, 3, family="queue_until"))
            .world.widget("queue")
            .items
        )
        > 1
    )
    alternate = queued.model_copy(deep=True)
    future = alternate.world.widget("queue").items[1]
    future.category = "changed-future"
    assert task_fingerprint(queued) != task_fingerprint(alternate)


def test_external_manifest_cannot_reassign_an_exact_instance_to_another_split():
    manifest = generate_manifest(train=1, dev=0, test=0)
    manifest.episodes[0].split = "test"
    with pytest.raises(ValueError, match="exact-instance split"):
        validate_manifest(manifest)
