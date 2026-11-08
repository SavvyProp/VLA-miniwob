import json
import shutil

import pytest
from PIL import Image

from scriptwob.contracts import ClickAction, HistoryEntry
from scriptwob.seeclick_data import (
    IGNORE_INDEX,
    SeeClickCollator,
    SeeClickDataset,
    load_training_export,
    tokenize_sample,
)
from scriptwob.seeclick_prompt import (
    DEFAULT_SYSTEM_MESSAGE,
    build_assistant_answer,
    build_inference_inputs,
    build_user_prompt,
    encode_text,
)


class CharacterTokenizer:
    """Readable token boundaries make loss masking assertions independent of Torch."""

    im_start_id = 1
    im_end_id = 2
    pad_token_id = 3

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [1000 + ord(char) for char in text]}


def make_export(directory, *, supervision="frontier"):
    images = directory / "images"
    images.mkdir(parents=True)
    image = images / "step.png"
    Image.new("RGB", (8, 8), "red").save(image)
    action = ClickAction(x=0.125, y=0.75)
    rows = [
        {
            "id": "train-example/0",
            "conversations": [
                {"from": "user", "value": build_user_prompt("Click red.", (), image)},
                {"from": "assistant", "value": build_assistant_answer(action)},
            ],
        }
    ]
    metadata = {
        "schema_version": "scriptwob-sft-v1",
        "format": "seeclick",
        "supervision": supervision,
        "policy": {"supervision": supervision},
        "examples": 1,
        "episodes": 1,
        "data_file": "train.json",
    }
    (directory / "train.json").write_text(json.dumps(rows))
    (directory / "dataset.json").write_text(json.dumps(metadata))
    return directory


def rewrite_rows(directory, mutate):
    path = directory / "train.json"
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def rewrite_metadata(directory, **changes):
    path = directory / "dataset.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_export_fingerprint_tracks_content_and_images(tmp_path):
    directory = make_export(tmp_path / "data")
    before = load_training_export(directory)
    assert before.directory == directory.resolve()
    assert set(before.image_hashes) == {"images/step.png"}
    assert load_training_export(directory).fingerprint == before.fingerprint
    Image.new("RGB", (8, 8), "blue").save(directory / "images" / "step.png")
    after = load_training_export(directory)
    assert before.fingerprint != after.fingerprint
    rewrite_rows(directory, lambda rows: rows[0].update(id="train-example/1"))
    assert after.fingerprint != load_training_export(directory).fingerprint


def test_fingerprint_survives_relocated_export_with_rewritten_image_tag(tmp_path):
    original = make_export(tmp_path / "original")
    copied = tmp_path / "copy"
    shutil.copytree(original, copied)
    path = copied / "train.json"
    path.write_text(path.read_text().replace(str(original), str(copied)))
    assert load_training_export(original).fingerprint == load_training_export(copied).fingerprint


def test_stored_image_integrity_digest_is_checked(tmp_path):
    directory = make_export(tmp_path / "data")
    original = load_training_export(directory)
    rewrite_metadata(directory, image_sha256=original.image_hashes)
    assert load_training_export(directory).image_hashes == original.image_hashes
    Image.new("RGB", (8, 8), "blue").save(directory / "images" / "step.png")
    with pytest.raises(ValueError, match="integrity"):
        load_training_export(directory)


def test_oracle_requires_explicit_flag_and_metadata_agreement(tmp_path):
    directory = make_export(tmp_path / "data", supervision="privileged_oracle")
    with pytest.raises(ValueError, match="allow-oracle"):
        load_training_export(directory)
    assert load_training_export(directory, allow_oracle=True).metadata["examples"] == 1
    rewrite_metadata(directory, supervision="frontier")
    with pytest.raises(ValueError, match="differs"):
        load_training_export(directory, allow_oracle=True)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"format": "messages"}, "seeclick"),
        ({"examples": 2}, "count"),
        ({"episodes": 2}, "episode count"),
        ({"split": "test"}, "training export"),
        ({"data_file": "../test.json"}, "training export"),
        ({"prompt_version": "different-prompt"}, "prompt version"),
        ({"prompt_style": "unknown"}, "prompt style"),
        ({"instruction_version": "different-instruction"}, "instruction version"),
        ({"prompt_style": "code"}, "instruction version"),
    ],
)
def test_rejects_wrong_dataset_contract(tmp_path, changes, match):
    directory = make_export(tmp_path / "data")
    rewrite_metadata(directory, **changes)
    with pytest.raises(ValueError, match=match):
        load_training_export(directory)


@pytest.mark.parametrize("image_failure", ["missing", "corrupt", "escape", "symlink"])
def test_rejects_unreadable_or_external_images(tmp_path, image_failure):
    directory = make_export(tmp_path / "data")
    image = directory / "images" / "step.png"
    if image_failure == "missing":
        image.unlink()
    elif image_failure == "corrupt":
        image.write_text("not an image")
    else:
        outside = tmp_path / "outside.png"
        shutil.copyfile(image, outside)
        if image_failure == "symlink":
            image.unlink()
            image.symlink_to(outside)
        else:
            path = directory / "train.json"
            path.write_text(path.read_text().replace(str(image), str(outside)))
    with pytest.raises(ValueError, match="image"):
        load_training_export(directory)


@pytest.mark.parametrize(
    "answer",
    [
        '{"action_type": 4, "click_point": [1.01, 0]}',
        '{"action_type": 4, "click_point": [NaN, 0]}',
        '{"action_type": 4, "click_point": [true, 0]}',
        '{"action_type": 5, "click_point": [0, 0]}',
        '{"action_type": 4, "click_point": [0, 0], "reasoning": "extra"}',
    ],
)
def test_rejects_wrong_assistant_action_contract(tmp_path, answer):
    directory = make_export(tmp_path / "data")
    rewrite_rows(directory, lambda rows: rows[0]["conversations"][1].update(value=answer))
    with pytest.raises(ValueError, match="assistant action"):
        load_training_export(directory)


def test_prompt_uses_literal_history_and_shared_public_format(tmp_path):
    history = (HistoryEntry(step=1, action=ClickAction(x=0.1, y=0.2)),)
    prompt = build_user_prompt("Click the second button.", history, tmp_path / "image.png")
    assert prompt.startswith(f"Picture 1: <img>{tmp_path}/image.png</img>\nPROCEDURE\n")
    assert "1: (0.100000, 0.200000) clicked" in prompt
    with pytest.raises(ValueError, match="254"):
        build_user_prompt("Click.", (), tmp_path / ("x" * 260))


@pytest.mark.parametrize("length", [254, 255, 256])
def test_image_paths_keep_padding_required_by_native_qwen_decoder(tmp_path, length):
    prefix = str(tmp_path) + "/"
    path = prefix + "x" * (length - len(prefix.encode()))
    if length == 254:
        assert f"<img>{path}</img>" in build_user_prompt("Click.", (), path)
    else:
        with pytest.raises(ValueError, match="254"):
            build_user_prompt("Click.", (), path)


def test_chatml_masks_every_prompt_token_and_trains_answer_and_stop(tmp_path):
    export = load_training_export(make_export(tmp_path / "data"))
    sample = export.samples[0]
    tokenizer = CharacterTokenizer()
    result = tokenize_sample(sample, tokenizer, 4096)
    prompt = build_inference_inputs(tokenizer, sample["conversations"][0]["value"], 4096)
    assert result["input_ids"][: len(prompt["input_ids"])] == prompt["input_ids"]
    assert result["labels"][: len(prompt["input_ids"])] == [IGNORE_INDEX] * len(prompt["input_ids"])
    learned = [label for label in result["labels"] if label != IGNORE_INDEX]
    assert learned == encode_text(tokenizer, sample["conversations"][1]["value"]) + [
        tokenizer.im_end_id
    ]
    assert result["attention_mask"] == [1] * len(result["input_ids"])
    system = [tokenizer.im_start_id] + encode_text(tokenizer, "system\n" + DEFAULT_SYSTEM_MESSAGE)
    assert result["input_ids"][: len(system)] == system


def test_no_silent_truncation_in_training_or_inference(tmp_path):
    export = load_training_export(make_export(tmp_path / "data"))
    tokenizer = CharacterTokenizer()
    full = tokenize_sample(export.samples[0], tokenizer, 4096)
    length = len(full["input_ids"])
    assert len(tokenize_sample(export.samples[0], tokenizer, length)["input_ids"]) == length
    with pytest.raises(ValueError, match=rf"train-example/0 has {length} tokens"):
        SeeClickDataset(export, tokenizer, length - 1)
    user = export.samples[0]["conversations"][0]["value"]
    prefix = build_inference_inputs(tokenizer, user, 4096)
    with pytest.raises(ValueError, match="reserved output"):
        build_inference_inputs(tokenizer, user, len(prefix["input_ids"]), reserve_tokens=1)


def test_collator_masks_padding_by_position_not_token_identity():
    collator = SeeClickCollator(pad_token_id=3)
    first = {"input_ids": [1, 3, 2], "labels": [-100, 3, 2], "attention_mask": [1, 1, 1]}
    second = {"input_ids": [1], "labels": [-100], "attention_mask": [1]}
    batch = collator.pad([first, second])
    assert batch["input_ids"] == [[1, 3, 2], [1, 3, 3]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[-100, 3, 2], [-100, -100, -100]]
    assert first["input_ids"] == [1, 3, 2]


def test_duplicate_ids_and_extra_image_tags_rejected(tmp_path):
    directory = make_export(tmp_path / "data")
    rewrite_rows(directory, lambda rows: rows.append(rows[0]))
    rewrite_metadata(directory, examples=2)
    with pytest.raises(ValueError, match="duplicate"):
        load_training_export(directory)
    rewrite_rows(directory, lambda rows: rows.pop())
    rewrite_metadata(directory, examples=1)
    rewrite_rows(
        directory,
        lambda rows: rows[0]["conversations"][0].update(
            value=rows[0]["conversations"][0]["value"] + "<img>other.png</img>"
        ),
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_training_export(directory)


def test_episode_metadata_must_match_training_sample_ids(tmp_path):
    directory = make_export(tmp_path / "data")
    episodes = [{"id": "train-example", "steps": 1, "level": 0}]
    path = directory / "episodes.json"
    path.write_text(json.dumps(episodes))
    assert load_training_export(directory).metadata["episodes"] == 1
    episodes[0]["steps"] = 2
    path.write_text(json.dumps(episodes))
    with pytest.raises(ValueError, match="sample ids"):
        load_training_export(directory)
    episodes[0]["steps"] = 1
    episodes[0]["split"] = "dev"
    path.write_text(json.dumps(episodes))
    with pytest.raises(ValueError, match="non-training"):
        load_training_export(directory)
