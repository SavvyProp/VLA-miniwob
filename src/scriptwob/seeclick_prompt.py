"""One prompt and ChatML contract for original SeeClick training and inference.

The native Qwen-VL tokenizer expands ``<img>path</img>`` into its image-token
span. Do not replace it with the image-placeholder conventions of Qwen2-VL.
"""

import json
from pathlib import Path

from .contracts import ClickAction, HistoryEntry, format_context

PROMPT_VERSION = "scriptwob-seeclick-v1"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
# The tokenizer accepts 256 bytes, but original Qwen-VL's forward drops the
# final span token before searching for an image-pad delimiter: keep two pads.
MAX_IMAGE_PATH_BYTES = 254
RESPONSE_INSTRUCTION = (
    "Return the next click as JSON with action_type=4 and click_point=[x,y], normalized to [0,1]."
)


def build_user_prompt(
    instruction: str, history: tuple[HistoryEntry, ...], image_path: str | Path
) -> str:
    """Format the literal public observation identically at export and inference."""
    path = str(Path(image_path).resolve())
    if "<" in path or ">" in path or "\n" in path or "\r" in path:
        raise ValueError("SeeClick image path contains unsupported tag or newline characters")
    if len(path.encode("utf-8")) > MAX_IMAGE_PATH_BYTES:
        raise ValueError("Original Qwen-VL image paths must fit within 254 UTF-8 bytes")
    return f"Picture 1: <img>{path}</img>\n{format_context(instruction, history)}\n" + (
        RESPONSE_INSTRUCTION
    )


def build_assistant_answer(action: ClickAction) -> str:
    return json.dumps({"action_type": 4, "click_point": [action.x, action.y]})


def encode_text(tokenizer, text: str) -> list[int]:
    """Use the original tokenizer, including its special image-token expansion."""
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(ids, list) or any(not isinstance(token, int) for token in ids):
        raise ValueError("Expected an unbatched original Qwen-VL token sequence")
    return ids


def chatml_header(tokenizer, role: str) -> list[int]:
    return [tokenizer.im_start_id] + encode_text(tokenizer, role) + encode_text(tokenizer, "\n")


def build_inference_inputs(
    tokenizer,
    user_prompt: str,
    max_length: int,
    reserve_tokens: int = 0,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> dict[str, list[int]]:
    """Build a complete, untruncated ChatML prompt ending at the assistant header.

    Reserve space for generation explicitly: upstream ``chat`` helpers may silently
    remove old context to satisfy their window setting, which changes this benchmark.
    """
    if max_length < 1 or reserve_tokens < 0:
        raise ValueError("max_length must be positive and reserve_tokens nonnegative")
    newline = encode_text(tokenizer, "\n")
    ids: list[int] = []
    for role, body in (("system", system_message), ("user", user_prompt)):
        ids += chatml_header(tokenizer, role)
        ids += encode_text(tokenizer, body) + [tokenizer.im_end_id] + newline
    ids += chatml_header(tokenizer, "assistant")
    required = len(ids) + reserve_tokens
    if required > max_length:
        raise ValueError(
            f"SeeClick context overflow: prompt has {len(ids)} tokens plus "
            f"{reserve_tokens} reserved output tokens, exceeding max_length={max_length}. "
            "Increase the configured context limit or use shorter tasks; "
            "context is never truncated."
        )
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}
