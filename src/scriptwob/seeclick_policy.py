"""Screenshot-only SeeClick policy for ScriptWoB evaluation.

The heavy inference dependencies are deliberately imported only when constructing
the policy. Dataset generation and frontier collection do not require PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from .contracts import ACTION_VERSION, ClickAction, Observation
from .policies import PolicyOutputError


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate action JSON keys")
        result[key] = value
    return result


def parse_click(text: str) -> ClickAction:
    """Accept one JSON click, optionally enclosed by one Markdown code fence.

    No code evaluation, coordinate clipping, regex action extraction, or fallback
    click is performed. An invalid prediction is an evaluated policy failure.
    """
    if not isinstance(text, str):
        raise PolicyOutputError("SeeClick output must be a JSON string")
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0] not in {"```", "```json"} or lines[-1] != "```":
            raise PolicyOutputError("SeeClick output must contain exactly one JSON action")
        value = "\n".join(lines[1:-1])
    try:
        action = json.loads(value, object_pairs_hook=_unique_object)
        if not isinstance(action, dict) or set(action) != {"action_type", "click_point"}:
            raise ValueError("Expected action_type and click_point only")
        if type(action["action_type"]) is not int or action["action_type"] != 4:
            raise ValueError("Only action_type 4 (click) is supported")
        point = action["click_point"]
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("click_point must contain two coordinates")
        if any(type(number) not in {int, float} or not math.isfinite(number) for number in point):
            raise ValueError("Click coordinates must be finite JSON numbers")
        return ClickAction(x=point[0], y=point[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise PolicyOutputError("SeeClick returned an invalid normalized JSON click") from exc


def checkpoint_hashes(directory: Path | None) -> dict[str, str]:
    """Hash local model artifacts once so changing weights invalidates run resume."""
    if directory is None:
        return {}
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".safetensors", ".bin", ".json", ".py", ".tiktoken", ".model",
        }:
            continue
        # Optimizer/scheduler state is irrelevant to inference and can be huge.
        if path.name in {"optimizer.pt", "scheduler.pt", "rng_state.pth"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path.relative_to(directory))] = digest.hexdigest()
    return hashes


class SeeClickPolicy:
    """Load a pinned SeeClick base model or trained adapter once and click greedily."""

    def __init__(
        self,
        checkpoint: str | Path = "base",
        *,
        device: str | None = None,
        quantization: str | None = None,
        precision: str | None = None,
        max_context_tokens: int | None = None,
        max_new_tokens: int = 64,
    ) -> None:
        from . import seeclick_runtime as runtime
        from .seeclick_prompt import PROMPT_VERSION

        config, adapter_path, metadata = runtime.resolve_checkpoint(
            checkpoint, device=device, quantization=quantization, precision=precision,
        )
        if max_context_tokens is None:
            max_context_tokens = metadata.get("max_length", 2048)
        if max_context_tokens < 1 or max_new_tokens < 1 or max_new_tokens >= max_context_tokens:
            raise ValueError("Token limits must be positive, with room for both prompt and output")
        hashes = checkpoint_hashes(adapter_path)
        if adapter_path is not None:
            expected = metadata.get("adapter_hashes")
            if not isinstance(expected, dict) or not expected:
                raise ValueError("Checkpoint metadata is missing adapter_hashes")
            for filename, expected_hash in expected.items():
                if hashes.get(filename) != expected_hash:
                    raise ValueError(f"Checkpoint artifact checksum mismatch: {filename}")
        self.tokenizer = runtime.load_tokenizer(config)
        self.model = runtime.load_model(config, training=False, adapter_path=adapter_path)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.calls: list[dict] = []
        self._provenance = {
            "policy": "seeclick",
            "supervision": "model",
            "action_contract": ACTION_VERSION,
            "model_config": config.model_dump(mode="json"),
            "checkpoint_metadata": deepcopy(metadata),
            "checkpoint_files_sha256": hashes,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "do_sample": False, "num_beams": 1,
                "max_context_tokens": max_context_tokens,
                "max_new_tokens": max_new_tokens,
            },
        }

    def reset(self, seed: int) -> None:
        # Greedy generation has no episode sampling seed or hidden chat history.
        # All executed actions arrive through Observation.history.
        self.calls = []

    def predict(self, observation: Observation) -> ClickAction:
        if self.model is None:
            raise RuntimeError("SeeClick policy has been closed")
        import torch

        from .seeclick_prompt import build_inference_inputs, build_user_prompt

        call: dict[str, Any] = {"call": len(self.calls) + 1, "status": "requested"}
        self.calls.append(call)
        # Qwen-VL reads image paths during model.forward; keep the file alive for
        # the whole generation, then remove it on success and every error path.
        with tempfile.TemporaryDirectory(prefix="scriptwob-seeclick-") as temporary:
            image_path = Path(temporary) / "observation.png"
            observation.image.convert("RGB").save(image_path, format="PNG")
            prompt = build_user_prompt(observation.instruction, observation.history, image_path)
            try:
                encoded = build_inference_inputs(
                    self.tokenizer, prompt, max_length=self.max_context_tokens,
                    reserve_tokens=self.max_new_tokens,
                )
            except ValueError as exc:
                call.update(status="invalid_output", error_type="ContextLimit")
                raise PolicyOutputError(
                    "SeeClick context limit reached; instruction and history were not truncated"
                ) from exc
            input_device = self.model.get_input_embeddings().weight.device
            inputs = {
                name: torch.tensor([values], dtype=torch.long, device=input_device)
                for name, values in encoded.items()
            }
            prompt_length = len(encoded["input_ids"])
            eos_tokens = list(dict.fromkeys([
                self.tokenizer.im_end_id, self.tokenizer.eod_id,
            ]))
            call["input_tokens"] = prompt_length
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, do_sample=False, num_beams=1, use_cache=True,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_tokens, pad_token_id=self.tokenizer.eod_id,
                    return_dict_in_generate=False,
                )
            # generate returns the prompt followed by new tokens. Decoding the
            # prompt would confuse echoed examples/history with the next action.
            tokens = outputs[0][prompt_length:]
            call["output_tokens"] = len(tokens)
            result = self.tokenizer.decode(tokens, skip_special_tokens=True)
            try:
                action = parse_click(result)
            except PolicyOutputError:
                call.update(status="invalid_output", error_type="InvalidClickJSON")
                raise
            call["status"] = "completed"
            return action

    def provenance(self) -> dict:
        """Return a defensive copy; fingerprints stay fixed for the loaded model."""
        return deepcopy(self._provenance)

    def close(self) -> None:
        """Release model references; screenshot files are already scoped per call."""
        self.model = None
        self.tokenizer = None
