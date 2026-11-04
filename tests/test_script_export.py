import hashlib
import json

import pytest
from PIL import Image

from scriptwob.contracts import ACTION_VERSION
from scriptwob.export import export_dataset
from scriptwob.generation import generate_manifest


def fixture_run(tmp_path, *, supervision="frontier", split="train", accepted=True):
    root = tmp_path / "run"
    directory = root / "attempts" / "episode-001"
    directory.mkdir(parents=True)
    manifest = generate_manifest(train=1, dev=0, test=0, levels=[0], seed=17)
    source = manifest.episodes[0]
    (root / "manifest.json").write_text(manifest.model_dump_json())
    (root / "run.json").write_text(
        json.dumps(
            {
                "config": {
                    "mode": "collect",
                    "split": split,
                    "manifest_digest": manifest.digest,
                    "action_contract": ACTION_VERSION,
                    "policy": {"supervision": supervision, "policy": "test_fixture"},
                }
            }
        )
    )
    episode = {
        "episode_id": source.id,
        "level": source.level,
        "split": split,
        "instruction": source.instruction,
        "family": source.family,
        "supervision": supervision,
        "action_contract": ACTION_VERSION,
    }
    (directory / "episode.json").write_text(json.dumps(episode))
    (directory / "private.json").write_text(
        json.dumps(
            {
                "accepted": accepted,
                "success": accepted,
                "environment": {"secret_ast": "PRIVATE"},
                "supervision": supervision,
                "episode_id": source.id,
            }
        )
    )
    rows = []
    for step in range(3):
        Image.new("RGB", (50, 40), (step * 20, 10, 30)).save(directory / f"{step:06}.png")
    for step in range(2):
        image, next_image = f"{step:06}.png", f"{step + 1:06}.png"
        rows.append(
            {
                "step": step + 1,
                "image": image,
                "next_image": next_image,
                "image_sha256": hashlib.sha256((directory / image).read_bytes()).hexdigest(),
                "next_image_sha256": hashlib.sha256(
                    (directory / next_image).read_bytes()
                ).hexdigest(),
                "action": {"x": 0.25, "y": 0.4},
                "feedback": "clicked",
                "terminal": step == 1,
            }
        )
    (directory / "transitions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, directory


def test_portable_export_preserves_public_history_and_no_private_state(tmp_path):
    root, _ = fixture_run(tmp_path)
    output = tmp_path / "dataset"
    summary = export_dataset(root, output)
    assert summary["episodes"] == 1 and summary["examples"] == 2
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert "(none)" in rows[0]["messages"][0]["content"]
    assert "1: (0.250000, 0.400000) clicked" in rows[1]["messages"][0]["content"]
    assert "PRIVATE" not in (output / "train.jsonl").read_text()
    assert (output / rows[0]["images"][0]).exists()


def test_seeclick_export_uses_original_action_and_conversation_format(tmp_path):
    root, _ = fixture_run(tmp_path)
    output = tmp_path / "dataset"
    export_dataset(root, output, format="seeclick")
    rows = json.loads((output / "train.json").read_text())
    assert rows[0]["conversations"][0]["from"] == "user"
    assert "<img>" in rows[0]["conversations"][0]["value"]
    assert json.loads(rows[0]["conversations"][1]["value"]) == {
        "action_type": 4,
        "click_point": [0.25, 0.4],
    }


def test_oracle_is_not_silently_exported_as_expert(tmp_path):
    root, _ = fixture_run(tmp_path, supervision="privileged_oracle")
    with pytest.raises(ValueError, match="oracle"):
        export_dataset(root, tmp_path / "dataset")
    result = export_dataset(root, tmp_path / "fixture", allow_oracle=True)
    assert result["supervision"] == "privileged_oracle"


@pytest.mark.parametrize("split", ["dev", "test"])
def test_eval_cannot_become_training_data(tmp_path, split):
    root, _ = fixture_run(tmp_path, split=split)
    with pytest.raises(ValueError, match="train"):
        export_dataset(root, tmp_path / "dataset")


def test_corruption_rejected_and_partial_export_cleaned(tmp_path):
    root, directory = fixture_run(tmp_path)
    (directory / "000001.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        export_dataset(root, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_no_success_is_not_an_empty_successful_dataset(tmp_path):
    root, _ = fixture_run(tmp_path, accepted=False)
    with pytest.raises(ValueError, match="No accepted"):
        export_dataset(root, tmp_path / "dataset")


@pytest.mark.parametrize(
    "field,value", [("instruction", "Click the wrong thing."), ("supervision", "privileged_oracle")]
)
def test_inconsistent_episode_metadata_rejected(tmp_path, field, value):
    root, directory = fixture_run(tmp_path)
    path = directory / "episode.json"
    episode = json.loads(path.read_text())
    episode[field] = value
    path.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="differs"):
        export_dataset(root, tmp_path / "dataset")


def test_changed_manifest_rejected(tmp_path):
    root, _ = fixture_run(tmp_path)
    path = root / "run.json"
    run = json.loads(path.read_text())
    run["config"]["manifest_digest"] = "wrong"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Manifest integrity"):
        export_dataset(root, tmp_path / "dataset")


def test_uncommitted_staging_attempt_is_not_exported(tmp_path):
    root, directory = fixture_run(tmp_path)
    directory.rename(directory.with_name(".episode-001.partial"))
    with pytest.raises(ValueError, match="No accepted"):
        export_dataset(root, tmp_path / "dataset")
