"""GUI-Owl policy using public instructions, executed clicks, and past screenshots."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import ACTION_VERSION, ClickAction, HistoryEntry, Observation
from .policies import PolicyOutputError
from .seeclick_policy import checkpoint_hashes, parse_click


def _image_digest(image: Image.Image) -> str:
    """Hash decoded RGB pixels, independent of the screenshot's PNG compression."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


class GUIOwlPolicy:
    """Load one model and maintain a bounded chronological screenshot window.

    All past clicks remain in the text history. Cached images are copied only
    from public observations, cleared on reset, and associated with the number
    of already executed actions. No task AST, hidden state, or oracle is used.
    """

    def __init__(
        self,
        checkpoint: str | Path = "base-2b",
        *,
        device: str | None = None,
        quantization: str | None = None,
        precision: str | None = None,
        max_context_tokens: int | None = None,
        max_new_tokens: int = 64,
        history_images: int | None = None,
        prompt_style: str | None = None,
        instruction_map: dict[str, str] | None = None,
    ) -> None:
        from . import gui_owl_runtime as runtime
        from .gui_owl_prompt import PROMPT_VERSION
        from .prompt_styles import (
            DEFAULT_PROMPT_STYLE,
            instruction_version,
            validate_prompt_style,
        )

        config, adapter_path, metadata = runtime.resolve_checkpoint(
            checkpoint, device=device, quantization=quantization, precision=precision,
        )
        dataset_metadata = metadata.get("dataset", {}).get("metadata", {})
        trained_style = dataset_metadata.get(
            "prompt_style", metadata.get("prompt_style", DEFAULT_PROMPT_STYLE)
        )
        validate_prompt_style(trained_style)
        selected_style = trained_style if prompt_style is None else prompt_style
        validate_prompt_style(selected_style)
        version = instruction_version(selected_style)
        trained_version = dataset_metadata.get(
            "instruction_version", metadata.get("instruction_version")
        )
        if trained_style != DEFAULT_PROMPT_STYLE and trained_version is None:
            raise ValueError("Code checkpoint metadata is missing instruction_version")
        if selected_style == trained_style and trained_version not in (None, version):
            raise ValueError("Checkpoint instruction version differs from the current renderer")
        if metadata.get("prompt_version") not in (None, PROMPT_VERSION):
            raise ValueError("Checkpoint GUI-Owl prompt version differs from the current renderer")
        self._instruction_map = None
        map_digest = None
        if selected_style != DEFAULT_PROMPT_STYLE:
            if not isinstance(instruction_map, dict) or not instruction_map:
                raise ValueError("Code prompt style requires a nonempty instruction_map")
            if any(
                not isinstance(source, str) or not source.strip()
                or not isinstance(presented, str) or not presented.strip()
                for source, presented in instruction_map.items()
            ):
                raise ValueError("instruction_map must contain only nonempty instruction strings")
            self._instruction_map = dict(instruction_map)
            map_digest = hashlib.sha256(json.dumps(
                self._instruction_map, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
        self.prompt_style = selected_style
        trained_history = metadata.get("history_images", 3)
        history_images = trained_history if history_images is None else history_images
        if type(history_images) is not int or not 0 <= history_images <= 32:
            raise ValueError("history_images must be an integer from 0 through 32")
        if max_context_tokens is None:
            max_context_tokens = metadata.get("max_length", 8192)
        if (
            type(max_context_tokens) is not int or type(max_new_tokens) is not int
            or not 1 <= max_new_tokens < max_context_tokens <= 32768
        ):
            raise ValueError(
                "Token limits must be positive, with room for prompt and output, "
                "within GUI-Owl's 32768-token context"
            )
        self.processor = runtime.load_processor(config)
        self.model = runtime.load_model(config, training=False, adapter_path=adapter_path)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.history_images = history_images
        self._provenance = {
            "policy": "gui_owl",
            "supervision": "model",
            "action_contract": ACTION_VERSION,
            "model_config": config.model_dump(mode="json"),
            "checkpoint_metadata": deepcopy(metadata),
            "checkpoint_files_sha256": checkpoint_hashes(adapter_path),
            "prompt_version": PROMPT_VERSION,
            "prompt_style": selected_style,
            "instruction_version": version,
            "instruction_map_sha256": map_digest,
            "trained_prompt_style": trained_style,
            "trained_instruction_version": trained_version,
            "prompt_style_overridden": selected_style != trained_style,
            "history_images": history_images,
            "trained_history_images": trained_history,
            "history_images_overridden": history_images != trained_history,
            "screenshot_history": "previous_observations_chronological_current_last",
            "action_history": "complete",
            "generation": {
                "do_sample": False, "num_beams": 1,
                "max_context_tokens": max_context_tokens,
                "max_new_tokens": max_new_tokens,
            },
        }
        self.reset(0)

    def reset(self, seed: int) -> None:
        """Discard all previous-episode visual state; greedy generation is unseeded."""
        self.calls: list[dict] = []
        self._images: list[tuple[int, Image.Image]] = []
        self._history: tuple[HistoryEntry, ...] = ()
        self._pending_action: ClickAction | None = None
        self._pending_size: tuple[int, int] | None = None
        self._instruction: str | None = None

    def _validate_history(self, observation: Observation) -> None:
        history = observation.history
        if [entry.step for entry in history] != list(range(1, len(history) + 1)):
            raise PolicyOutputError("GUI-Owl needs contiguous executed history starting at step 1")
        if self._pending_action is None:
            if history:
                raise PolicyOutputError("GUI-Owl must start at the initial observation after reset")
            return
        if (
            len(history) != len(self._history) + 1
            or tuple(history[:-1]) != self._history
            or observation.instruction != self._instruction
        ):
            raise PolicyOutputError(
                "GUI-Owl observation history changed or skipped a step; reset between episodes"
            )
        # The browser rounds normalized coordinates onto the screenshot's pixel
        # grid. Compare the executed pixel, not the original normalized float.
        assert self._pending_size is not None
        if history[-1].action.pixels(*self._pending_size) != self._pending_action.pixels(
            *self._pending_size
        ):
            raise PolicyOutputError(
                "The executed click differs from GUI-Owl's preceding prediction"
            )

    def predict(self, observation: Observation) -> ClickAction:
        if self.model is None:
            raise RuntimeError("GUI-Owl policy has been closed")
        from .gui_owl_prompt import build_inference_inputs, build_messages

        call: dict[str, Any] = {"call": len(self.calls) + 1, "status": "requested"}
        self.calls.append(call)
        instruction = observation.instruction
        if self._instruction_map is not None:
            try:
                instruction = self._instruction_map[instruction]
            except KeyError as exc:
                call.update(status="invalid_output", error_type="UnknownInstruction")
                raise PolicyOutputError(
                    "Instruction is missing from the static code prompt map"
                ) from exc
        call["presented_instruction"] = instruction
        try:
            self._validate_history(observation)
        except PolicyOutputError:
            call.update(status="invalid_output", error_type="HistoryMismatch")
            raise
        step = len(observation.history)
        current = observation.image.convert("RGB").copy()
        images = [image for _, image in self._images] + [current]
        image_steps = [index for index, _ in self._images] + [step]
        call.update(
            history_image_count=len(self._images),
            image_count=len(images),
            image_steps=image_steps,
            images_sha256=[_image_digest(image) for image in images],
            image_digest_format="sha256(RGB:width:height: + decoded_RGB_pixels)",
            executed_action_count=step,
        )
        messages = build_messages(instruction, observation.history, images, image_steps)
        try:
            inputs = build_inference_inputs(
                self.processor, messages, max_length=self.max_context_tokens,
                reserve_tokens=self.max_new_tokens,
            )
        except ValueError as exc:
            call.update(status="invalid_output", error_type="ContextLimit")
            raise PolicyOutputError(
                "GUI-Owl input could not fit its context; "
                "instruction and history were not truncated"
            ) from exc
        import torch

        input_device = self.model.get_input_embeddings().weight.device
        inputs = {name: value.to(input_device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[-1]
        call["input_tokens"] = prompt_length
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, do_sample=False, num_beams=1, use_cache=True,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )
        # Never decode the prompt: it contains example action JSON and history.
        tokens = outputs[0][prompt_length:]
        call["output_tokens"] = len(tokens)
        result = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        call["output_text"] = result
        try:
            action = parse_click(result)
        except PolicyOutputError:
            call.update(status="invalid_output", error_type="InvalidClickJSON")
            raise
        self._images = (self._images + [(step, current)])[-self.history_images:]
        if self.history_images == 0:
            self._images = []
        self._history = tuple(observation.history)
        self._pending_action = action
        self._pending_size = current.size
        self._instruction = observation.instruction
        call["status"] = "completed"
        return action

    def provenance(self) -> dict:
        return deepcopy(self._provenance)

    def close(self) -> None:
        """Release model and screenshot memory; safe to call repeatedly."""
        self.model = None
        self.processor = None
        self.reset(0)
