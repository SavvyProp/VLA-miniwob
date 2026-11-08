import hashlib
import json
import shutil

import pytest
from PIL import Image

from scriptwob.contracts import ACTION_VERSION
from scriptwob.export import export_dataset, reformat_seeclick_export
from scriptwob.generation import generate_manifest
from scriptwob.prompt_styles import instruction_version, render_episode_instruction
from scriptwob.seeclick_data import load_training_export


def fixture_run(tmp_path, *, supervision="frontier", split="train", accepted=True, level=0):
    root = tmp_path / "run"
    directory = root / "attempts" / "episode-001"
    directory.mkdir(parents=True)
    manifest = generate_manifest(train=1, dev=0, test=0, levels=[level], seed=17)
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


def _procedure_and_history(sample):
    context = sample["conversations"][0]["value"].split("</img>\nPROCEDURE\n", 1)[1]
    return context.split("\nEXECUTED CLICK HISTORY\n", 1)


@pytest.mark.parametrize("format", ["messages", "seeclick"])
@pytest.mark.parametrize("level", range(6))
def test_code_export_preserves_labels_images_and_history(tmp_path, format, level):
    root, _ = fixture_run(tmp_path, level=level)
    natural_path, code_path = tmp_path / "natural", tmp_path / "code"
    natural = export_dataset(root, natural_path, format=format)
    code = export_dataset(root, code_path, format=format, prompt_style="code")
    assert code["prompt_style"] == "code"
    assert code["instruction_version"] == instruction_version("code")
    assert code["manifest_digest"] == natural["manifest_digest"]
    assert code["image_sha256"] == natural["image_sha256"]
    assert (code["episodes"], code["examples"]) == (1, 2)
    for relative in code["image_sha256"]:
        assert (natural_path / relative).read_bytes() == (code_path / relative).read_bytes()
    if format == "seeclick":
        first = load_training_export(natural_path)
        second = load_training_export(code_path)
        assert first.fingerprint != second.fingerprint
        for before, after in zip(first.samples, second.samples, strict=True):
            assert before["id"] == after["id"]
            assert before["conversations"][1] == after["conversations"][1]
            old_procedure, old_history = _procedure_and_history(before)
            new_procedure, new_history = _procedure_and_history(after)
            assert old_procedure != new_procedure
            assert old_history == new_history
    else:
        before = [
            json.loads(line) for line in (natural_path / "train.jsonl").read_text().splitlines()
        ]
        after = [json.loads(line) for line in (code_path / "train.jsonl").read_text().splitlines()]
        for old, new in zip(before, after, strict=True):
            assert old["messages"][1] == new["messages"][1]
            assert old["images"] == new["images"]
            assert "\nEXECUTED CLICK HISTORY\n" in new["messages"][0]["content"]


def test_reformat_reuses_frozen_export_and_round_trips_without_collection(tmp_path):
    root, _ = fixture_run(tmp_path)
    source = tmp_path / "source"
    export_dataset(root, source, format="seeclick")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes((root / "manifest.json").read_bytes())
    # Legacy metadata and missing optional episode index remain reusable.
    metadata_path = source / "dataset.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("prompt_style")
    metadata.pop("instruction_version")
    metadata_path.write_text(json.dumps(metadata))
    (source / "episodes.json").unlink()
    shutil.rmtree(root)
    before = load_training_export(source)
    code = tmp_path / "code"
    summary = reformat_seeclick_export(source, manifest_path, code)
    assert summary["source_export_fingerprint"] == before.fingerprint
    assert summary["source_prompt_style"] == "natural"
    after = load_training_export(code)
    assert after.image_hashes == before.image_hashes
    for old, new in zip(before.samples, after.samples, strict=True):
        assert old["id"] == new["id"]
        assert old["conversations"][1] == new["conversations"][1]
        assert _procedure_and_history(old)[1] == _procedure_and_history(new)[1]
        assert _procedure_and_history(old)[0] != _procedure_and_history(new)[0]
    round_trip = tmp_path / "roundtrip"
    reformat_seeclick_export(code, manifest_path, round_trip, prompt_style="natural")
    for old, new in zip(before.samples, load_training_export(round_trip).samples, strict=True):
        assert _procedure_and_history(old) == _procedure_and_history(new)
        assert old["conversations"][1] == new["conversations"][1]
    # Copies are independent files, not hard links that allow shared mutation.
    relative = next(iter(before.image_hashes))
    assert (source / relative).stat().st_ino != (code / relative).stat().st_ino


@pytest.mark.parametrize("failure", ["manifest", "instruction", "nontrain", "missing_step"])
def test_reformat_rejects_mismatched_or_nontraining_source(tmp_path, failure):
    root, _ = fixture_run(tmp_path)
    source = tmp_path / "source"
    export_dataset(root, source, format="seeclick")
    manifest_path = root / "manifest.json"
    rows_path = source / "train.json"
    metadata_path = source / "dataset.json"
    rows = json.loads(rows_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if failure == "manifest":
        other = generate_manifest(train=1, dev=0, test=0, levels=[0], seed=18)
        manifest_path = tmp_path / "other.json"
        manifest_path.write_text(other.model_dump_json())
        expected = "manifest digest"
    elif failure == "instruction":
        rows[0]["conversations"][0]["value"] = rows[0]["conversations"][0]["value"].replace(
            "\nPROCEDURE\n", "\nPROCEDURE\nChoose the answer in advance. "
        )
        rows_path.write_text(json.dumps(rows))
        expected = "procedure differs"
    elif failure == "nontrain":
        other = generate_manifest(train=0, dev=1, test=0, levels=[0], seed=18)
        manifest_path = tmp_path / "dev.json"
        manifest_path.write_text(other.model_dump_json())
        metadata["manifest_digest"] = other.digest
        metadata_path.write_text(json.dumps(metadata))
        for index, row in enumerate(rows):
            row["id"] = f"{other.episodes[0].id}/{index}"
        rows_path.write_text(json.dumps(rows))
        (source / "episodes.json").unlink()
        expected = "not in the training manifest"
    else:
        rows[0]["id"] = rows[0]["id"].rsplit("/", 1)[0] + "/2"
        rows_path.write_text(json.dumps(rows))
        (source / "episodes.json").unlink()
        expected = "Noncontiguous"
    with pytest.raises(ValueError, match=expected):
        reformat_seeclick_export(source, manifest_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reformat_cleans_partial_output_on_copy_failure(tmp_path, monkeypatch):
    root, _ = fixture_run(tmp_path)
    source = tmp_path / "source"
    export_dataset(root, source, format="seeclick")

    def fail_copy(*args):
        raise OSError("disk full")

    monkeypatch.setattr("scriptwob.export.shutil.copyfile", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        reformat_seeclick_export(source, root / "manifest.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reformatted_procedures_match_manifest_renderer(tmp_path):
    root, _ = fixture_run(tmp_path)
    source, output = tmp_path / "source", tmp_path / "code"
    export_dataset(root, source, format="seeclick")
    reformat_seeclick_export(source, root / "manifest.json", output)
    manifest = generate_manifest(train=1, dev=0, test=0, levels=[0], seed=17)
    expected = render_episode_instruction(manifest.episodes[0], "code")
    assert all(
        _procedure_and_history(row)[0] == expected
        for row in load_training_export(output).samples
    )
