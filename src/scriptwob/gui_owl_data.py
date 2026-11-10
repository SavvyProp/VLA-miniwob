"""Validated, lazy screenshot-history SFT from existing scriptWoB exports."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .contracts import HistoryEntry, format_context
from .gui_owl_prompt import build_inference_inputs, build_messages, process_messages
from .seeclick_data import IGNORE_INDEX, TrainingExport, _assistant_action
from .seeclick_prompt import RESPONSE_INSTRUCTION

_IMAGE_TAG = re.compile(r"\APicture 1: <img>([^<>\r\n]+)</img>\n")


@dataclass(frozen=True)
class GUIOwlSample:
    id: str
    instruction: str
    history: tuple[HistoryEntry, ...]
    images: tuple[Path, ...]
    image_steps: tuple[int, ...]
    answer: str

    def messages(self) -> list[dict]:
        return build_messages(
            self.instruction, self.history, list(self.images), list(self.image_steps)
        )


def reconstruct_samples(exported: TrainingExport, history_images: int) -> list[GUIOwlSample]:
    """Recover only the prior frames of the same successful demonstration.

    Missing steps and mismatching action histories are errors, even when a sample
    happens to fall outside the configured image window. Text histories are the
    exact six-decimal serialization used by the original export.
    """
    if type(history_images) is not int or history_images < 0:
        raise ValueError("history_images must be a nonnegative integer")
    episodes: dict[str, dict[int, dict]] = {}
    for row in exported.samples:
        sample_id = row["id"]
        try:
            episode, suffix = sample_id.rsplit("/", 1)
            step = int(suffix)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid episode/step sample id: {sample_id}") from exc
        if not episode or step < 0 or str(step) != suffix:
            raise ValueError(f"Invalid episode/step sample id: {sample_id}")
        bucket = episodes.setdefault(episode, {})
        if step in bucket:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        bucket[step] = row

    resolved: dict[str, GUIOwlSample] = {}
    for episode, rows in episodes.items():
        if sorted(rows) != list(range(len(rows))):
            raise ValueError(f"Episode {episode} must have contiguous steps starting at 0")
        history: list[HistoryEntry] = []
        image_paths: list[Path] = []
        instruction = None
        for step in range(len(rows)):
            row = rows[step]
            sample_id = row["id"]
            user, assistant = row["conversations"]
            match = _IMAGE_TAG.match(user["value"])
            if match is None:
                raise ValueError(f"Missing screenshot in {sample_id}")
            body = user["value"][match.end() :]
            if not body.startswith("PROCEDURE\n") or "\nEXECUTED CLICK HISTORY\n" not in body:
                raise ValueError(f"Invalid procedure/history in {sample_id}")
            current_instruction = body[len("PROCEDURE\n") :].rsplit(
                "\nEXECUTED CLICK HISTORY\n", 1
            )[0]
            if instruction is None:
                instruction = current_instruction
            elif instruction != current_instruction:
                raise ValueError(f"Instruction changed within episode {episode}")
            expected = format_context(instruction, tuple(history)) + "\n" + RESPONSE_INSTRUCTION
            if body != expected:
                raise ValueError(
                    f"Sample {sample_id} history differs from its exact preceding actions; "
                    "cannot reconstruct screenshot history safely"
                )
            image_paths.append(Path(match.group(1)))
            first_image = max(0, step - history_images)
            resolved[sample_id] = GUIOwlSample(
                id=sample_id,
                instruction=instruction,
                history=tuple(history),
                images=tuple(image_paths[first_image : step + 1]),
                image_steps=tuple(range(first_image, step + 1)),
                answer=assistant["value"],
            )
            history.append(
                HistoryEntry(
                    step=step + 1, action=_assistant_action(assistant["value"], sample_id)
                )
            )
    return [resolved[row["id"]] for row in exported.samples]


def tokenize_sample(
    sample: GUIOwlSample, processor, max_length: int, reserve_tokens: int = 64
) -> tuple[dict, int]:
    """Mask the inference prefix; learn the assistant answer and end token only."""
    messages = sample.messages()
    prompt = build_inference_inputs(processor, messages, max_length, reserve_tokens)
    prompt_ids = prompt["input_ids"][0]
    full = process_messages(
        processor,
        messages + [{"role": "assistant", "content": sample.answer}],
        add_generation_prompt=False,
    )
    ids = full["input_ids"][0]
    boundary = len(prompt_ids)
    if len(ids) <= boundary or not bool(ids[:boundary].eq(prompt_ids).all()):
        raise ValueError(f"Sample {sample.id} training prefix differs from inference prompt")
    if not bool(full["image_grid_thw"].eq(prompt["image_grid_thw"]).all()):
        raise ValueError(f"Sample {sample.id} training image grids differ from inference")
    if len(ids) > max_length:
        raise ValueError(
            f"GUI-Owl sample {sample.id} has {len(ids)} tokens, exceeding "
            f"max_length={max_length}; samples are never truncated"
        )
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == getattr(processor.tokenizer, "unk_token_id", None):
        raise ValueError("Qwen3-VL tokenizer must define the assistant <|im_end|> token")
    endings = (ids[boundary:] == end_id).nonzero(as_tuple=False).flatten().tolist()
    if len(endings) != 1:
        raise ValueError(f"Sample {sample.id} must have exactly one assistant end token")
    end = boundary + endings[0] + 1
    answer_tokens = end - boundary
    if answer_tokens > reserve_tokens:
        raise ValueError(
            f"Sample {sample.id} answer needs {answer_tokens} tokens, exceeding "
            f"reserve_tokens={reserve_tokens}; increase max_new_tokens for train and evaluation"
        )
    labels = ids.clone()
    labels[:boundary] = IGNORE_INDEX
    labels[end:] = IGNORE_INDEX
    return {
        "input_ids": ids,
        "attention_mask": full["attention_mask"][0],
        "labels": labels,
        "pixel_values": full["pixel_values"],
        "image_grid_thw": full["image_grid_thw"],
    }, boundary


class GUIOwlDataset:
    """Map dataset: only metadata is resident; each item decodes its own window."""

    def __init__(
        self,
        exported: TrainingExport,
        processor,
        max_length: int,
        history_images: int = 3,
        reserve_tokens: int = 64,
    ):
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if type(reserve_tokens) is not int or reserve_tokens < 1:
            raise ValueError("reserve_tokens must be a positive integer")
        self.samples = reconstruct_samples(exported, history_images)
        self.fingerprint = exported.fingerprint
        self.processor = processor
        self.max_length = max_length
        self.history_images = history_images
        self.reserve_tokens = reserve_tokens
        self.token_lengths: list[int | None] = [None] * len(self.samples)
        self.prompt_lengths: list[int | None] = [None] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        result, prompt_length = tokenize_sample(
            self.samples[index], self.processor, self.max_length, self.reserve_tokens
        )
        self.token_lengths[index] = len(result["input_ids"])
        self.prompt_lengths[index] = prompt_length
        return result

    def preflight(self) -> dict:
        """Check all context windows before training, without retaining pixel tensors."""
        completed = sum(length is not None for length in self.token_lengths)
        if completed < len(self):
            print(
                f"GUI-Owl data preflight: {completed}/{len(self)} examples",
                file=sys.stderr,
                flush=True,
            )
        for index, length in enumerate(self.token_lengths):
            if length is None:
                self[index]
                completed += 1
                if completed % 100 == 0 or completed == len(self):
                    print(
                        f"GUI-Owl data preflight: {completed}/{len(self)} examples",
                        file=sys.stderr,
                        flush=True,
                    )
        lengths = [length for length in self.token_lengths if length is not None]
        return {
            "examples": len(self),
            "episodes": len({sample.id.rsplit("/", 1)[0] for sample in self.samples}),
            "images": len({image for sample in self.samples for image in sample.images}),
            "tokens_min": min(lengths),
            "tokens_max": max(lengths),
            "tokens_mean": sum(lengths) / len(lengths),
            "prompt_tokens_max": max(self.prompt_lengths),
            "reserved_output_tokens": self.reserve_tokens,
            "truncated_examples": 0,
        }


class GUIOwlCollator:
    """Right-pad text and concatenate Qwen's flattened image patches and grids."""

    def __init__(self, processor):
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("GUI-Owl tokenizer must define pad_token_id")

    def __call__(self, samples: list[dict]) -> dict:
        import torch

        if not samples:
            raise ValueError("Cannot collate an empty GUI-Owl batch")
        maximum = max(len(sample["input_ids"]) for sample in samples)
        batch = {
            "input_ids": torch.full((len(samples), maximum), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(samples), maximum), dtype=torch.long),
            "labels": torch.full((len(samples), maximum), IGNORE_INDEX, dtype=torch.long),
        }
        for index, sample in enumerate(samples):
            length = len(sample["input_ids"])
            if any(sample[key].shape != (length,) for key in batch):
                raise ValueError("Token, label, and attention lengths differ")
            grids = sample["image_grid_thw"]
            pixels = sample["pixel_values"]
            if grids.ndim != 2 or grids.shape[1] != 3 or pixels.ndim != 2:
                raise ValueError("Expected Qwen flattened patches and (images,3) image grids")
            if bool((grids <= 0).any()) or int(grids.prod(dim=1).sum()) != pixels.shape[0]:
                raise ValueError("Image grid patch counts differ from flattened pixels")
            for key in batch:
                batch[key][index, :length] = sample[key]
        batch["pixel_values"] = torch.cat([sample["pixel_values"] for sample in samples], dim=0)
        batch["image_grid_thw"] = torch.cat(
            [sample["image_grid_thw"] for sample in samples], dim=0
        )
        return batch
