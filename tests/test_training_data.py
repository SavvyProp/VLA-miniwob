import json
import shutil

import h5py
import numpy as np
import pytest
from test_dataset import fixture_run

from vla_miniwob.collection import write_json
from vla_miniwob.dataset import export_hdf5, fit_normalization
from vla_miniwob.training_data import (
    prepare_training_data,
    prepare_validation_data,
    sampling_weights,
    validate_export,
)


def annotated_run(tmp_path):
    root = fixture_run(tmp_path)
    for path in (root / "episodes").glob("*/private.json"):
        record = json.loads(path.read_text())
        number = int(path.parent.name)
        record["instance"].update(
            id=f"click-button/{record['instance']['split']}/{number}",
            task="click-button",
            order="forward",
            wording="direct",
            values_seed=number * 10,
            layout_seed=number * 10 + 1,
            order_seed=number * 10 + 2,
        )
        if number == 2:
            record["instance"]["split"] = "dev"
        write_json(path, record)
    return root


def test_raw_training_preparation_provenance_and_cache(tmp_path):
    root = annotated_run(tmp_path)
    output = tmp_path / "prepared"
    first = prepare_training_data(root, output, horizon=4)
    second = prepare_training_data(root, output, horizon=4)
    assert first.fingerprint == second.fingerprint
    assert first.dataset.horizon == 4
    assert first.sampling["mode"] == "balanced_component_task_order_wording"
    assert first.dataset.metadata["episodes"][0]["component_count"] == 1
    assert set(first.dataset[0]) == {"image", "state", "action", "is_pad", "raw_lang"}
    assert "SECRET" not in (output / "index.json").read_text()
    with pytest.raises(ValueError, match="differs"):
        prepare_training_data(root, output, horizon=8)
    (root / "run.json").write_text('{"mode":"collect", "changed":true}')
    with pytest.raises(ValueError, match="differs"):
        prepare_training_data(root, output, horizon=4)


def test_dev_uses_train_stats_and_is_disjoint(tmp_path):
    root = annotated_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    dev = prepare_validation_data(root, tmp_path / "dev", training=train)
    assert dev.dataset.normalizer.stats == train.dataset.normalizer.stats
    assert dev.dataset.metadata["split"] == "dev"
    with pytest.raises(ValueError, match="train export"):
        prepare_training_data(dev.root, tmp_path / "unused")


@pytest.mark.parametrize("field", ["values_seed", "layout_seed", "source_episode"])
def test_dev_rejects_overlap(tmp_path, field):
    root = annotated_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    output = tmp_path / "dev"
    export_hdf5(root, output, split="dev", stats=train.dataset.normalizer.stats)
    metadata = json.loads((output / "index.json").read_text())
    metadata["episodes"][0][field] = train.dataset.metadata["episodes"][0][field]
    write_json(output / "index.json", metadata)
    with pytest.raises(ValueError, match="overlap"):
        prepare_validation_data(output, tmp_path / "unused", training=train)


def test_eval_and_independently_fitted_dev_stats_rejected(tmp_path):
    root = annotated_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    stats = fit_normalization(root)
    stats["state_mean"][0] += 0.1
    export_hdf5(root, tmp_path / "dev", split="dev", stats=stats)
    with pytest.raises(ValueError, match="exactly the training normalization"):
        prepare_validation_data(tmp_path / "dev", tmp_path / "unused", training=train)
    metadata = json.loads((tmp_path / "dev" / "index.json").read_text())
    metadata["split"] = "eval"
    write_json(tmp_path / "dev" / "index.json", metadata)
    with pytest.raises(ValueError, match="dev export"):
        prepare_validation_data(tmp_path / "dev", tmp_path / "unused", training=train)


def test_balanced_mass_and_legacy_fallback():
    def episode(count, task, order, wording, length):
        return dict(component_count=count, task=task, order=order, wording=wording, length=length)

    metadata = {
        "episodes": [
            episode(1, "A", "forward", "direct", 2),
            episode(1, "A", "forward", "reverse_clauses", 4),
            episode(1, "B", "forward", "direct", 1),
            episode(2, "C", "forward", "direct", 3),
        ]
    }
    weights, report = sampling_weights(metadata)
    np.testing.assert_allclose(
        [weights[:2].sum(), weights[2:6].sum(), weights[6], weights[7:].sum()],
        [0.125, 0.125, 0.25, 0.5],
    )
    assert report["cells"] == 4
    weights, report = sampling_weights({"episodes": [{"length": 2}]})
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert report["reason"] == "legacy_export_missing_sampling_metadata"


def test_training_legacy_export_allowed_but_dev_requires_seed_metadata(tmp_path):
    root = fixture_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    assert train.sampling["mode"] == "uniform_transitions"
    path = root / "episodes" / "2" / "private.json"
    private = json.loads(path.read_text())
    private["instance"]["split"] = "dev"
    write_json(path, private)
    with pytest.raises(ValueError, match="Cannot verify split seeds"):
        prepare_validation_data(root, tmp_path / "dev", training=train)


def test_fingerprint_tracks_hdf5_contents(tmp_path):
    root = annotated_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    episode = train.root / train.dataset.metadata["episodes"][0]["file"]
    with h5py.File(episode, "r+") as file:
        file["action"][0] = [0.4, 0.5, 1]
    changed = prepare_training_data(train.root, tmp_path / "unused")
    assert train.fingerprint != changed.fingerprint


def test_export_path_escape_and_bad_length_rejected(tmp_path):
    root = annotated_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    metadata = train.dataset.metadata
    episode = train.root / metadata["episodes"][0]["file"]
    shutil.copy(episode, tmp_path / "outside.hdf5")
    metadata["episodes"][0]["file"] = "../outside.hdf5"
    write_json(train.root / "index.json", metadata)
    with pytest.raises(ValueError, match="escapes"):
        validate_export(train.root, "train")
    metadata["episodes"][0]["file"] = episode.name
    metadata["episodes"][0]["length"] = 3
    write_json(train.root / "index.json", metadata)
    with pytest.raises(ValueError, match="Misaligned"):
        validate_export(train.root, "train")


def test_hdf5_external_links_rejected(tmp_path):
    root = annotated_run(tmp_path)
    train = prepare_training_data(root, tmp_path / "train")
    episode = train.root / train.dataset.metadata["episodes"][0]["file"]
    shutil.copy(episode, tmp_path / "external.hdf5")
    with h5py.File(episode, "r+") as file:
        del file["action"]
        file["action"] = h5py.ExternalLink(str(tmp_path / "external.hdf5"), "/action")
    with pytest.raises(ValueError, match="linked HDF5 field"):
        validate_export(train.root, "train")
