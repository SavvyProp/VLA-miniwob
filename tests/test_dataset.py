import json

import h5py
import numpy as np
import pytest
from PIL import Image

from vla_miniwob.collection import write_json
from vla_miniwob.contracts import MouseAction, decode
from vla_miniwob.dataset import ChunkDataset, Normalizer, export_hdf5, fit_normalization


def fixture_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "run.json", {"mode": "collect"})
    for episode, split, success in [(0, "train", True), (1, "train", False), (2, "eval", True)]:
        path = root / "episodes" / str(episode)
        path.mkdir(parents=True)
        write_json(
            path / "private.json",
            {
                "instance": {"split": split},
                "verdict": {"success": success},
                "infrastructure_failure": False,
                "SECRET_NEXT_TARGET": "do not export",
            },
        )
        write_json(
            path / "episode.json",
            {
                "instruction": "Click A, then B.",
                "image_size": [20, 10],
                "action_contract": "mouse-absolute-click-v1",
                "initial_state": [0, 0, 0],
            },
        )
        rows = []
        for t in range(2):
            Image.new("RGB", (20, 10), color=(t * 50, episode, 0)).save(path / f"{t:06d}.png")
            action = MouseAction(x=0.1 + t * 0.2, y=0.2, click=1)
            rows.append(
                {
                    "step": t,
                    "image": f"{t:06d}.png",
                    "state": [t / 10, 0, 0],
                    "requested_action": action.model_dump(),
                    "executed": decode(action, 20, 10).model_dump(),
                    "feedback": "executed",
                    "next_image": "future.png",
                    "next_state": [0, 0, 0],
                    "terminal": t == 1,
                }
            )
        (path / "transitions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return root


def test_success_only_export_and_no_future_or_private_data(tmp_path):
    root = fixture_run(tmp_path)
    stats = fit_normalization(root)
    assert stats["source_episodes"] == ["0"]
    assert stats["transitions"] == 2
    output = tmp_path / "export"
    metadata = export_hdf5(root, output, horizon=16)
    assert len(metadata["episodes"]) == 1
    assert metadata["chunk_examples"] == 2
    data = ChunkDataset(output)
    assert len(data) == 2
    first, last = data[0], data[1]
    assert first["action"].shape == (16, 3)
    assert first["is_pad"].tolist() == [False, False] + [True] * 14
    assert last["is_pad"].tolist() == [False] + [True] * 15
    assert first["image"][0, 0, 0] == 0
    assert last["image"][0, 0, 0] == 50
    assert "(none)" in first["raw_lang"]
    assert "1:" in last["raw_lang"] and "2:" not in last["raw_lang"]
    assert not any("SECRET" in str(v) for v in last.values())
    with h5py.File(output / metadata["episodes"][0]["file"]) as f:
        assert f.attrs["sim"]
        assert "private" not in f and "next_image" not in f
    normalizer = Normalizer(stats)
    action = np.array([[0.2, 0.3, 1]], dtype=np.float32)
    np.testing.assert_allclose(normalizer.undo_actions(normalizer.actions(action)), action)


def test_eval_statistics_cannot_be_fitted_or_exported_implicitly(tmp_path):
    root = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="training run"):
        export_hdf5(root, tmp_path / "bad", split="eval")
    stats = fit_normalization(root)
    stats["fit_split"] = "eval"
    with pytest.raises(ValueError, match="non-training"):
        Normalizer(stats)
