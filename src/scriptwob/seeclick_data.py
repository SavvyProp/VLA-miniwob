"""Validated screenshot/action data and assistant-only original Qwen-VL SFT.

ChatML framing follows the official SeeClick/Qwen-VL model convention. Unlike
upstream's demonstration trainer, this module masks every user/system token and
refuses to truncate long procedures or action histories.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import ClickAction
from .prompt_styles import DEFAULT_PROMPT_STYLE, instruction_version, validate_prompt_style
from .seeclick_prompt import (
    DEFAULT_SYSTEM_MESSAGE,
    MAX_IMAGE_PATH_BYTES,
    PROMPT_VERSION,
    RESPONSE_INSTRUCTION,
    build_inference_inputs,
    encode_text,
)

IGNORE_INDEX = -100
_IMAGE_TAG = re.compile(r"\APicture 1: <img>([^<>\r\n]+)</img>\n")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@dataclass(frozen=True)
class TrainingExport:
    directory: Path
    samples: list[dict]
    metadata: dict
    fingerprint: str
    image_hashes: dict[str, str]


def _assistant_action(value: str, sample_id: str) -> ClickAction:
    try:
        action = json.loads(value)
        if not isinstance(action, dict) or set(action) != {"action_type", "click_point"}:
            raise ValueError("expected action_type and click_point only")
        if type(action["action_type"]) is not int or action["action_type"] != 4:
            raise ValueError("only click action_type=4 is supported")
        point = action["click_point"]
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("click_point must contain two coordinates")
        if any(isinstance(x, bool) or not isinstance(x, int | float) for x in point):
            raise ValueError("coordinates must be numbers")
        return ClickAction(x=point[0], y=point[1])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid assistant action in sample {sample_id}: {exc}") from exc


def load_training_export(path: str | Path, *, allow_oracle: bool = False) -> TrainingExport:
    """Validate a scriptWoB SeeClick export before loading any model weights.

    The fingerprint covers the JSON content and decoded image file bytes, with
    image locations made relative to the export so relocating an export does not
    change its identity. Absolute image tags must point inside this export.
    """
    directory = Path(path).resolve()
    metadata = json.loads((directory / "dataset.json").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("dataset.json must contain an object")
    if metadata.get("schema_version") != "scriptwob-sft-v1" or metadata.get("format") != "seeclick":
        raise ValueError("Training requires a scriptWoB seeclick export, not messages JSONL")
    if metadata.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
        raise ValueError("Unsupported SeeClick prompt version")
    prompt_style = metadata.get("prompt_style", DEFAULT_PROMPT_STYLE)
    validate_prompt_style(prompt_style)
    # Original exports predate presentation metadata. Only their original natural
    # instruction renderer has an implicit version; new styles require an explicit one.
    default_version = (
        instruction_version(DEFAULT_PROMPT_STYLE) if prompt_style == DEFAULT_PROMPT_STYLE else None
    )
    if metadata.get("instruction_version", default_version) != instruction_version(prompt_style):
        raise ValueError("Unsupported instruction version for the dataset prompt style")
    supervision = metadata.get("supervision")
    if supervision != "frontier" and not (allow_oracle and supervision == "privileged_oracle"):
        raise ValueError("Training requires frontier supervision; oracle data needs --allow-oracle")
    policy = metadata.get("policy")
    if not isinstance(policy, dict) or policy.get("supervision") != supervision:
        raise ValueError("Dataset policy supervision differs from dataset supervision")
    if metadata.get("split", "train") != "train" or metadata.get("data_file") != "train.json":
        raise ValueError("Only train.json from a training export can be used for fine-tuning")
    samples = json.loads((directory / "train.json").read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError("Training export must contain at least one example")
    if metadata.get("examples") != len(samples):
        raise ValueError("Dataset example count differs from train.json")
    seen: set[str] = set()
    image_hashes: dict[str, str] = {}
    expected_hashes = metadata.get("image_sha256")
    if expected_hashes is not None and not isinstance(expected_hashes, dict):
        raise ValueError("Dataset image_sha256 must map relative image paths to SHA256 digests")
    canonical_samples: list[dict] = []
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Every training sample must be an object")
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate training sample id: {sample_id}")
        seen.add(sample_id)
        if row.get("split", "train") != "train":
            raise ValueError(f"Non-training sample in export: {sample_id}")
        messages = row.get("conversations")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or any(not isinstance(message, dict) for message in messages)
            or [message.get("from") for message in messages] != ["user", "assistant"]
            or any(not isinstance(message.get("value"), str) for message in messages)
        ):
            raise ValueError(f"Sample {sample_id} requires one user/assistant conversation pair")
        user = messages[0]["value"]
        match = _IMAGE_TAG.match(user)
        if match is None or user.count("<img>") != 1 or user.count("</img>") != 1:
            raise ValueError(f"Sample {sample_id} requires exactly one local Qwen-VL image tag")
        image = Path(match.group(1))
        if not image.is_absolute():
            raise ValueError(f"Sample {sample_id} image path must be absolute; re-export the data")
        image = image.resolve()
        if not image.is_relative_to(directory / "images") or not image.is_file():
            raise ValueError(f"Missing or unsafe image for sample {sample_id}: {image}")
        if len(str(image).encode("utf-8")) > MAX_IMAGE_PATH_BYTES:
            raise ValueError(f"Sample {sample_id} image path exceeds Qwen-VL's 254-byte safe limit")
        body = user[match.end() :]
        if (
            not body.startswith("PROCEDURE\n")
            or "\nEXECUTED CLICK HISTORY\n" not in body
            or not body.endswith("\n" + RESPONSE_INSTRUCTION)
        ):
            raise ValueError(f"Sample {sample_id} does not use the scriptWoB SeeClick prompt")
        if any(token in user for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            raise ValueError(f"Sample {sample_id} contains unexpected ChatML control tokens")
        relative = image.relative_to(directory).as_posix()
        if relative not in image_hashes:
            try:
                with Image.open(image) as decoded:
                    if min(decoded.size) < 2:
                        raise ValueError("screenshot dimensions must be at least two pixels")
                    decoded.load()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable image for sample {sample_id}: {image}") from exc
            image_hashes[relative] = hashlib.sha256(image.read_bytes()).hexdigest()
            if (
                expected_hashes is not None
                and expected_hashes.get(relative) != image_hashes[relative]
            ):
                raise ValueError(f"Image integrity check failed for sample {sample_id}: {relative}")
        _assistant_action(messages[1]["value"], sample_id)
        canonical_row = json.loads(json.dumps(row))
        canonical_row["conversations"][0]["value"] = f"Picture 1: <img>{relative}</img>\n" + body
        canonical_samples.append(canonical_row)
    if expected_hashes is not None and set(expected_hashes) != set(image_hashes):
        raise ValueError("Dataset image_sha256 contains missing or unreferenced images")
    if metadata.get("episodes") != len({sample_id.rsplit("/", 1)[0] for sample_id in seen}):
        raise ValueError("Dataset episode count differs from train.json")
    episode_metadata = None
    if (directory / "episodes.json").is_file():
        episode_metadata = json.loads((directory / "episodes.json").read_text())
        if not isinstance(episode_metadata, list) or len(episode_metadata) != metadata.get(
            "episodes"
        ):
            raise ValueError("Dataset episode count differs from episodes.json")
        episode_ids: set[str] = set()
        expected_samples: set[str] = set()
        for episode in episode_metadata:
            if (
                not isinstance(episode, dict)
                or not isinstance(episode.get("id"), str)
                or not episode["id"]
                or episode["id"] in episode_ids
                or type(episode.get("steps")) is not int
                or episode["steps"] < 1
                or episode.get("split", "train") != "train"
            ):
                raise ValueError("Invalid or non-training episode in episodes.json")
            episode_ids.add(episode["id"])
            expected_samples.update(f"{episode['id']}/{step}" for step in range(episode["steps"]))
        if expected_samples != seen:
            raise ValueError("Training sample ids differ from episodes.json")
    fingerprint = hashlib.sha256(
        _canonical(
            {
                "prompt_version": PROMPT_VERSION,
                "metadata": metadata,
                "samples": canonical_samples,
                "images": image_hashes,
                "episodes": episode_metadata,
            }
        )
    ).hexdigest()
    return TrainingExport(directory, samples, metadata, fingerprint, image_hashes)


def tokenize_sample(
    sample: dict,
    tokenizer,
    max_length: int,
    *,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> dict[str, list[int]]:
    """Produce unpadded tokens with loss on assistant content and its end token only."""
    user, assistant = sample["conversations"]
    # Build with an effectively unbounded limit, then report the full sample length
    # below; truncation is never used even when the prompt itself is already long.
    prompt = build_inference_inputs(
        tokenizer, user["value"], max_length=2**63 - 1, system_message=system_message
    )["input_ids"]
    answer = encode_text(tokenizer, assistant["value"]) + [tokenizer.im_end_id]
    newline = encode_text(tokenizer, "\n")
    ids = prompt + answer + newline
    if len(ids) > max_length:
        raise ValueError(
            f"SeeClick sample {sample['id']} has {len(ids)} tokens, exceeding "
            f"max_length={max_length}; samples are never truncated. Increase the context "
            "limit or generate shorter scripts."
        )
    return {
        "input_ids": ids,
        "labels": [IGNORE_INDEX] * len(prompt) + answer + [IGNORE_INDEX] * len(newline),
        "attention_mask": [1] * len(ids),
    }


class SeeClickDataset:
    """A Trainer-compatible map dataset that preflights every sequence length."""

    def __init__(self, export: TrainingExport, tokenizer, max_length: int):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.fingerprint = export.fingerprint
        self.samples = [tokenize_sample(sample, tokenizer, max_length) for sample in export.samples]
        self.token_lengths = [len(sample["input_ids"]) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


class SeeClickCollator:
    """Right-pad a batch without masking real tokens that happen to equal pad_token_id."""

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("Set the original Qwen tokenizer pad_token_id to eod_id")
        self.pad_token_id = pad_token_id

    def pad(self, samples: list[dict[str, list[int]]]) -> dict[str, list[list[int]]]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        maximum = max(len(sample["input_ids"]) for sample in samples)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for sample in samples:
            length = len(sample["input_ids"])
            if len(sample["labels"]) != length or len(sample["attention_mask"]) != length:
                raise ValueError("Token, label, and attention lengths differ")
            padding = maximum - length
            batch["input_ids"].append(sample["input_ids"] + [self.pad_token_id] * padding)
            batch["labels"].append(sample["labels"] + [IGNORE_INDEX] * padding)
            batch["attention_mask"].append(sample["attention_mask"] + [0] * padding)
        return batch

    def __call__(self, samples: list[dict[str, list[int]]]):
        import torch

        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in self.pad(samples).items()
        }
