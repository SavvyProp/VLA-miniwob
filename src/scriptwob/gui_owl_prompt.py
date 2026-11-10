"""Shared screenshot-history prompts for GUI-Owl's native Qwen3-VL processor.

Both training and rollout inference use the processor's official chat template.
Images are pre-action observations: image index N is the page after N clicks,
and the last image is always the current page. No hidden task state is included.
"""

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import HistoryEntry, format_context
from .seeclick_prompt import DEFAULT_SYSTEM_MESSAGE, RESPONSE_INSTRUCTION

PROMPT_VERSION = "scriptwob-gui-owl-v1"


def build_messages(
    instruction: str,
    history: tuple[HistoryEntry, ...],
    images: list[Image.Image | str | Path],
    image_steps: list[int],
) -> list[dict]:
    """Build public multimodal messages, retaining the full text click history."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("GUI-Owl requires a nonempty instruction")
    if "<|" in instruction or any("<|" in item.feedback for item in history):
        raise ValueError("Instruction/history must not contain chat-template control tokens")
    if [item.step for item in history] != list(range(1, len(history) + 1)):
        raise ValueError("GUI-Owl history must contain contiguous steps starting at 1")
    if not images or len(images) != len(image_steps):
        raise ValueError("Each screenshot must have its corresponding image step")
    current_step = len(history)
    expected = list(range(current_step - len(images) + 1, current_step + 1))
    if (
        any(type(step) is not int for step in image_steps)
        or expected[0] < 0
        or image_steps != expected
    ):
        raise ValueError("Screenshot steps must be chronological and end at the current step")
    content: list[dict] = [
        {"type": "text", "text": format_context(instruction, history) + "\n"},
        {
            "type": "text",
            "text": (
                "Screenshots are chronological. Only the final screenshot is the current page. "
                "Choose the next click on that page.\n"
            ),
        },
    ]
    for image, step in zip(images, image_steps, strict=True):
        if not isinstance(image, Image.Image | str | Path):
            raise ValueError("Screenshots must be PIL images or local image paths")
        label = "current" if step == current_step else "previous"
        content.extend(
            [
                {"type": "text", "text": f"Screenshot after {step} clicks ({label}):\n"},
                {"type": "image", "image": image},
            ]
        )
    content.append({"type": "text", "text": "\n" + RESPONSE_INSTRUCTION})
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def process_messages(processor, messages: list[dict], *, add_generation_prompt: bool) -> dict:
    """Process one conversation without retaining decoded images or truncating text.

    Qwen3-VL expands each image placeholder using its grid size. Consequently,
    token counts and loss boundaries must be measured *after* image processing.
    """
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    with ExitStack() as stack:
        images = []
        for message in messages:
            if not isinstance(message["content"], list):
                continue
            for item in message["content"]:
                if item["type"] != "image":
                    continue
                source = item["image"]
                if isinstance(source, Image.Image):
                    images.append(source)
                else:
                    opened = stack.enter_context(Image.open(Path(source)))
                    converted = opened.convert("RGB")
                    stack.callback(converted.close)
                    images.append(converted)
        result = processor(
            text=[rendered],
            images=images,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
    expected = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not expected.issubset(result):
        raise ValueError("GUI-Owl processor must return text tokens, image pixels, and image grids")
    result = {key: result[key] for key in expected}
    if result["input_ids"].ndim != 2 or result["input_ids"].shape[0] != 1:
        raise ValueError("Expected exactly one unpadded GUI-Owl conversation")
    if result["image_grid_thw"].shape != (len(images), 3):
        raise ValueError("Processor image grids do not match the supplied screenshots")
    if result["attention_mask"].shape != result["input_ids"].shape:
        raise ValueError("GUI-Owl token and attention lengths differ")
    if not bool(result["attention_mask"].eq(1).all()):
        raise ValueError("Expected an unpadded GUI-Owl conversation")
    return result


def build_inference_inputs(
    processor,
    messages: list[dict],
    max_length: int,
    reserve_tokens: int = 64,
) -> dict[str, Any]:
    """Return a complete inference prefix with explicit reserved output capacity."""
    if max_length < 1 or reserve_tokens < 0:
        raise ValueError("max_length must be positive and reserve_tokens nonnegative")
    result = process_messages(processor, messages, add_generation_prompt=True)
    length = result["input_ids"].shape[1]
    if length + reserve_tokens > max_length:
        raise ValueError(
            f"GUI-Owl context overflow: prompt has {length} tokens plus {reserve_tokens} "
            f"reserved output tokens, exceeding max_length={max_length}. "
            "Context is never truncated; increase the limit or reduce the configured history."
        )
    return result
