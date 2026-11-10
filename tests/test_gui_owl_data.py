import json
import re

import pytest
from PIL import Image

from scriptwob.contracts import ClickAction, HistoryEntry
from scriptwob.gui_owl_data import (
    GUIOwlCollator,
    GUIOwlDataset,
    reconstruct_samples,
    tokenize_sample,
)
from scriptwob.gui_owl_prompt import build_inference_inputs, build_messages
from scriptwob.seeclick_data import IGNORE_INDEX, load_training_export
from scriptwob.seeclick_prompt import build_assistant_answer, build_user_prompt


class TinyTokenizer:
    pad_token_id = 0
    unk_token_id = -1
    special = {"<|im_start|>": 1, "<|im_end|>": 2, "<|image_pad|>": 3}

    def convert_tokens_to_ids(self, token):
        return self.special.get(token, self.unk_token_id)

    def encode(self, text):
        parts = re.split(r"(<\|(?:im_start|im_end|image_pad)\|>)", text)
        ids = []
        for part in parts:
            if part in self.special:
                ids.append(self.special[part])
            else:
                ids.extend(1000 + ord(character) for character in part)
        return ids


class TinyProcessor:
    """Image-aware processor double, without requiring Transformers or weights."""

    def __init__(self):
        self.torch = pytest.importorskip("torch")
        self.tokenizer = TinyTokenizer()
        self.image_calls = []
        self.alter_training_prefix = False

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        result = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<|image_pad|>" if item["type"] == "image" else item["text"]
                    for item in content
                )
            result += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
        elif self.alter_training_prefix:
            result = result.replace("helpful assistant", "different assistant")
        return result

    def __call__(
        self, *, text, images, padding, truncation, return_token_type_ids, return_tensors
    ):
        assert padding is False and truncation is False and return_token_type_ids is False
        assert return_tensors == "pt"
        assert len(text) == 1
        assert text[0].count("<|image_pad|>") == len(images)
        colors = [image.getpixel((0, 0))[0] for image in images]
        self.image_calls.append(colors)
        ids = self.torch.tensor([self.tokenizer.encode(text[0])])
        return {
            "input_ids": ids,
            "attention_mask": self.torch.ones_like(ids),
            "pixel_values": self.torch.tensor(
                [[color] * 3 for color in colors for _ in range(4)], dtype=self.torch.float32
            ),
            "image_grid_thw": self.torch.tensor([[1, 2, 2] for _ in images]),
        }


def make_trajectories(directory):
    images = directory / "images"
    images.mkdir(parents=True)
    rows = []
    for episode, count in (("first", 4), ("second", 2)):
        history = []
        instruction = f"IF {episode} checkbox is checked:\n    Click blue.\nELSE:\n    Click red."
        for step in range(count):
            image = images / f"{episode}-{step}.png"
            Image.new("RGB", (8, 8), (step + (10 if episode == "second" else 0), 0, 0)).save(image)
            action = ClickAction(x=0.123456789 + step / 10, y=0.75)
            rows.append(
                {
                    "id": f"{episode}/{step}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": build_user_prompt(instruction, tuple(history), image),
                        },
                        {"from": "assistant", "value": build_assistant_answer(action)},
                    ],
                }
            )
            history.append(HistoryEntry(step=step + 1, action=action))
    # The dataset's physical row order need not be chronological.
    rows = [rows[index] for index in (3, 0, 5, 2, 4, 1)]
    (directory / "train.json").write_text(json.dumps(rows))
    (directory / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "scriptwob-sft-v1",
                "format": "seeclick",
                "supervision": "frontier",
                "policy": {"supervision": "frontier"},
                "examples": len(rows),
                "episodes": 2,
                "data_file": "train.json",
            }
        )
    )
    return load_training_export(directory)


def test_reconstructs_shuffled_episodes_without_future_or_cross_episode_images(tmp_path):
    exported = make_trajectories(tmp_path / "data")
    samples = reconstruct_samples(exported, history_images=2)
    assert [sample.id for sample in samples] == [row["id"] for row in exported.samples]
    current = samples[0]
    assert current.id == "first/3"
    assert current.image_steps == (1, 2, 3)
    assert [image.name for image in current.images] == ["first-1.png", "first-2.png", "first-3.png"]
    assert len(current.history) == 3
    # Keep original action precision; formatting alone rounds coordinates to six decimals.
    assert current.history[0].action.x == 0.123456789
    for sample in samples:
        episode, step = sample.id.split("/")
        assert sample.image_steps[-1] == int(step)
        assert all(image.name.startswith(episode + "-") for image in sample.images)
        assert max(sample.image_steps) == len(sample.history)
    current_only = reconstruct_samples(exported, history_images=0)[0]
    assert current_only.image_steps == (3,)
    assert len(current_only.history) == 3


@pytest.mark.parametrize("failure", ["missing_step", "wrong_history", "future", "instruction"])
def test_rejects_broken_trajectory_alignment(tmp_path, failure):
    exported = make_trajectories(tmp_path / "data")
    sample = exported.samples[0]
    if failure == "missing_step":
        exported.samples[:] = [row for row in exported.samples if row["id"] != "first/1"]
        message = "contiguous"
    elif failure == "wrong_history":
        sample["conversations"][0]["value"] = sample["conversations"][0]["value"].replace(
            "0.123457", "0.765432"
        )
        message = "exact preceding actions"
    elif failure == "future":
        # A row that claims step 0 cannot contain action history from step 3.
        sample["id"] = "third/0"
        message = "exact preceding actions"
    else:
        sample["conversations"][0]["value"] = sample["conversations"][0]["value"].replace(
            "first checkbox", "different checkbox"
        )
        message = "Instruction changed"
    with pytest.raises(ValueError, match=message):
        reconstruct_samples(exported, history_images=0)


def test_prompt_labels_current_page_and_preserves_full_history(tmp_path):
    sample = reconstruct_samples(make_trajectories(tmp_path / "data"), 1)[0]
    messages = sample.messages()
    content = messages[1]["content"]
    text = "".join(item.get("text", "") for item in content)
    assert "Screenshot after 2 clicks (previous)" in text
    assert "Screenshot after 3 clicks (current)" in text
    assert "1: (0.123457, 0.750000) clicked" in text
    assert "3: (0.323457, 0.750000) clicked" in text
    assert sample.instruction in text
    assert len([item for item in content if item["type"] == "image"]) == 2
    with pytest.raises(ValueError, match="chronological"):
        build_messages(sample.instruction, sample.history, list(sample.images), [2, 4])
    with pytest.raises(ValueError, match="contiguous"):
        build_messages(sample.instruction, sample.history[1:], list(sample.images), [1, 2])


def test_masking_matches_inference_and_learns_answer_and_end_only(tmp_path):
    processor = TinyProcessor()
    sample = reconstruct_samples(make_trajectories(tmp_path / "data"), 2)[0]
    item, boundary = tokenize_sample(sample, processor, max_length=4096, reserve_tokens=128)
    prefix = build_inference_inputs(processor, sample.messages(), 4096, reserve_tokens=128)
    assert item["input_ids"][:boundary].tolist() == prefix["input_ids"][0].tolist()
    assert item["labels"][:boundary].tolist() == [IGNORE_INDEX] * boundary
    learned = item["labels"][item["labels"] != IGNORE_INDEX].tolist()
    assert learned == processor.tokenizer.encode(sample.answer) + [2]
    assert item["labels"][-1] == IGNORE_INDEX  # Template newline after assistant end.
    assert processor.image_calls == [[1, 2, 3]] * 3


def test_prefix_mismatch_is_not_silently_used_for_masking(tmp_path):
    processor = TinyProcessor()
    processor.alter_training_prefix = True
    sample = reconstruct_samples(make_trajectories(tmp_path / "data"), 0)[0]
    with pytest.raises(ValueError, match="training prefix differs"):
        tokenize_sample(sample, processor, 4096, 128)


def test_checks_context_and_answer_budget_without_truncating(tmp_path):
    processor = TinyProcessor()
    sample = reconstruct_samples(make_trajectories(tmp_path / "data"), 2)[0]
    prefix = build_inference_inputs(processor, sample.messages(), 4096, reserve_tokens=128)
    length = prefix["input_ids"].shape[1]
    with pytest.raises(ValueError, match="never truncated"):
        tokenize_sample(sample, processor, length + 127, reserve_tokens=128)
    with pytest.raises(ValueError, match="answer needs"):
        tokenize_sample(sample, processor, 4096, reserve_tokens=1)


def test_dataset_preflight_keeps_pixels_lazy(tmp_path, capsys):
    processor = TinyProcessor()
    exported = make_trajectories(tmp_path / "data")
    dataset = GUIOwlDataset(exported, processor, 4096, history_images=1, reserve_tokens=128)
    assert processor.image_calls == []
    assert dataset.token_lengths == [None] * 6
    report = dataset.preflight()
    progress = capsys.readouterr().err
    assert "0/6 examples" in progress and "6/6 examples" in progress
    assert report["examples"] == 6 and report["episodes"] == 2 and report["images"] == 6
    assert report["tokens_max"] > report["tokens_min"] > 0
    assert report["truncated_examples"] == 0
    assert report["reserved_output_tokens"] == 128
    assert all(isinstance(length, int) for length in dataset.token_lengths)
    assert all(not isinstance(value, processor.torch.Tensor) for value in vars(dataset).values())
    assert all(not isinstance(value, Image.Image) for value in vars(dataset.samples[0]).values())
    calls = len(processor.image_calls)
    assert dataset.preflight() == report
    assert len(processor.image_calls) == calls
    assert capsys.readouterr().err == ""


def test_collator_preserves_flat_image_order_and_masks_padding_by_position(tmp_path):
    processor = TinyProcessor()
    dataset = GUIOwlDataset(
        make_trajectories(tmp_path / "data"), processor, 4096, history_images=2, reserve_tokens=128
    )
    long = dataset[0]  # first episode, image steps 1,2,3
    short = dataset[4]  # second episode, image step 0 only
    short["input_ids"][-2] = processor.tokenizer.pad_token_id
    short["labels"][-2] = processor.tokenizer.pad_token_id
    batch = GUIOwlCollator(processor)([long, short])
    assert batch["input_ids"].shape == (2, len(long["input_ids"]))
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2]] * 4
    assert batch["pixel_values"][:, 0].tolist() == [1] * 4 + [2] * 4 + [3] * 4 + [10] * 4
    short_len = len(short["input_ids"])
    assert batch["labels"][1, short_len - 2] == 0  # A genuine token equal to pad remains learned.
    assert bool(batch["labels"][1, short_len:].eq(IGNORE_INDEX).all())
    assert bool(batch["attention_mask"][1, short_len:].eq(0).all())
    long["image_grid_thw"][0, 2] += 1
    with pytest.raises(ValueError, match="patch counts"):
        GUIOwlCollator(processor)([long])


@pytest.mark.parametrize("history_images", [-1, 1.5, True])
def test_rejects_invalid_history_window(tmp_path, history_images):
    with pytest.raises(ValueError, match="nonnegative integer"):
        reconstruct_samples(make_trajectories(tmp_path / "data"), history_images)
