"""Pinned GUI-Owl-1.5/Qwen3-VL loaders with language-only LoRA adapters.

Heavy imports are lazy: collecting/exporting demonstrations requires neither
PyTorch nor the separate ``integrations/gui_owl`` environment.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GUI_OWL_MODELS = {
    "2b": ("mPLUG/GUI-Owl-1.5-2B-Instruct", "528ceaec795bbfbe6103bd79e03db849feadfb24"),
    "4b": ("mPLUG/GUI-Owl-1.5-4B-Instruct", "3f061c2c562cc860c42bf32542a70e07a7ff4840"),
    "8b": ("mPLUG/GUI-Owl-1.5-8B-Instruct", "06d5faecff74840bab2be2425e9c42667a5d04fc"),
}
TRAINING_METADATA = "scriptwob_training.json"
TRAINING_SCHEMA = "scriptwob-gui-owl-training-v1"
RUNTIME_INSTALL = "uv sync --project integrations/gui_owl --extra cuda --frozen"
# A suffix such as q_proj would also select visual projections in other models.
# This namespace is verified against Transformers 4.57.6's Qwen3-VL implementation.
LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


class GUIOwlModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = GUI_OWL_MODELS["2b"][0]
    model_revision: str = Field(default=GUI_OWL_MODELS["2b"][1], pattern=r"^[0-9a-f]{40}$")
    quantization: Literal["4bit", "none"] = "4bit"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda(:[0-9]+)?)$")
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1, allow_inf_nan=False)
    gradient_checkpointing: bool = True
    # Qwen3-VL uses 16-pixel patches and spatial_merge_size=2. These bounds
    # correspond approximately to 64--256 visual tokens per screenshot.
    min_pixels: int = Field(default=65536, ge=1024)
    max_pixels: int = Field(default=262144, ge=1024)

    @model_validator(mode="before")
    @classmethod
    def default_revision_for_model(cls, value: Any) -> Any:
        if isinstance(value, dict) and "model_revision" not in value:
            revisions = dict(GUI_OWL_MODELS.values())
            if value.get("model_id") in revisions:
                value = {**value, "model_revision": revisions[value["model_id"]]}
        return value

    @model_validator(mode="after")
    def validate_runtime(self) -> GUIOwlModelConfig:
        if self.model_id not in {identifier for identifier, _ in GUI_OWL_MODELS.values()}:
            raise ValueError("Choose an official GUI-Owl-1.5 2B, 4B, or 8B Instruct model")
        if self.quantization == "4bit" and self.device == "cpu":
            raise ValueError("The GUI-Owl 4-bit runtime requires a CUDA GPU")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU GUI-Owl checks require precision='fp32'")
        if self.quantization == "4bit" and self.precision == "fp32":
            raise ValueError("4-bit GUI-Owl requires bf16 or fp16 compute precision")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        return self


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(f"GUI-Owl dependencies are missing. Run: {RUNTIME_INSTALL}") from exc
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        raise RuntimeError(
            f"This environment lacks Qwen3-VL support. Run GUI-Owl in: {RUNTIME_INSTALL}"
        )
    return torch, transformers, peft


def load_processor(config: GUIOwlModelConfig) -> Any:
    """Use the model's pinned tokenizer, image processor, and native chat template."""
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            f"GUI-Owl processor dependencies are missing. Run: {RUNTIME_INSTALL}"
        ) from exc
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=False,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        padding_side="right",
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        if processor.tokenizer.eos_token_id is None:
            raise ValueError("GUI-Owl tokenizer has neither a padding nor an EOS token")
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return processor


def _validate_hardware(config: GUIOwlModelConfig, torch: Any) -> Any:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.precision]
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("GUI-Owl was configured for CUDA, but PyTorch cannot access a GPU")
        index = int(config.device.partition(":")[2] or 0)
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        if config.precision == "bf16":
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("This GPU does not support bf16; select precision='fp16'")
    return dtype


def adapter_hashes(directory: Path) -> dict[str, str]:
    """Stream the two required adapter artifacts without loading tensors."""
    result = {}
    for name in ADAPTER_FILES:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Incomplete GUI-Owl adapter checkpoint: missing {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def _validate_adapter(directory: Path, config: GUIOwlModelConfig) -> None:
    if not (directory / "adapter_model.safetensors").is_file():
        raise ValueError(f"Missing safe LoRA adapter weights in {directory}")
    data = json.loads((directory / "adapter_config.json").read_text())
    if not isinstance(data, dict) or data.get("peft_type") != "LORA":
        raise ValueError("Only GUI-Owl LoRA adapters are supported")
    if data.get("base_model_name_or_path") != config.model_id:
        raise ValueError("Adapter base model differs from its scriptWoB model configuration")
    if data.get("revision") != config.model_revision:
        raise ValueError("Adapter base revision differs from its scriptWoB model configuration")
    if data.get("target_modules") != LORA_TARGETS or data.get("modules_to_save"):
        raise ValueError("Adapter is not a scriptWoB language-only LoRA adapter")
    for field, value in (
        ("r", config.lora_rank), ("lora_alpha", config.lora_alpha),
        ("lora_dropout", config.lora_dropout), ("bias", "none"), ("task_type", "CAUSAL_LM"),
    ):
        if data.get(field) != value:
            raise ValueError(f"Adapter {field} differs from its scriptWoB model configuration")


def load_model(
    config: GUIOwlModelConfig,
    *,
    training: bool = False,
    adapter_path: str | Path | None = None,
) -> Any:
    """Load a single-device Qwen3-VL model, freeze vision, and attach language LoRA.

    Vision remains in its pretrained floating point format; language projections
    use NF4 by default. No automatic CPU offload or remote Python code is enabled.
    """
    torch, transformers, peft = _dependencies()
    dtype = _validate_hardware(config, torch)
    if adapter_path is not None:
        adapter_path = Path(adapter_path).resolve()
        _validate_adapter(adapter_path, config)
    kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "trust_remote_code": False,
        "dtype": dtype,
        "device_map": {
            "": int(config.device.partition(":")[2] or 0)
            if config.device.startswith("cuda") else config.device
        },
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "attn_implementation": "sdpa",
    }
    if config.quantization == "4bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"4-bit GUI-Owl requires bitsandbytes. Run: {RUNTIME_INSTALL}"
            ) from exc
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["model.visual", "lm_head"],
        )
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(config.model_id, **kwargs)
    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise ValueError("This loader requires GUI-Owl's Qwen3-VL architecture")
    model.model.visual.requires_grad_(False)
    model.config.use_cache = not training
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = not training

    if training:
        if config.quantization == "4bit":
            model = peft.prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            # PEFT promotes all unquantized weights to fp32. Restore the frozen
            # visual encoder and vocabulary matrices to avoid doubling their VRAM.
            model.model.visual.to(dtype=dtype)
            model.get_input_embeddings().to(dtype=dtype)
            model.get_output_embeddings().to(dtype=dtype)
        else:
            model.requires_grad_(False)
            if config.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

    if adapter_path is not None:
        model = peft.PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training)
    elif training:
        targets = [name for name, _ in model.named_modules() if re.fullmatch(LORA_TARGETS, name)]
        if not targets:
            raise ValueError("No Qwen3-VL language projections matched the LoRA configuration")
        adapter_config = peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
            revision=config.model_revision,
        )
        model = peft.get_peft_model(model, adapter_config)

    if training:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=torch.float32)
        unexpected = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and ("lora_" not in name or ".language_model.layers." not in name)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable GUI-Owl parameters: {unexpected[:5]}")
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("GUI-Owl has no trainable LoRA parameters")
        model.train()
    else:
        model.eval()
    return model


def resolve_checkpoint(
    checkpoint: str | Path = "base-2b",
    *,
    device: str | None = None,
    quantization: str | None = None,
    precision: str | None = None,
) -> tuple[GUIOwlModelConfig, Path | None, dict[str, Any]]:
    """Resolve a pinned base or explicit complete adapter, never guess 'latest'."""
    adapter_path: Path | None = None
    metadata: dict[str, Any] = {}
    selector = "base-2b" if str(checkpoint) == "base" else str(checkpoint)
    if selector in {f"base-{size}" for size in GUI_OWL_MODELS}:
        identifier, revision = GUI_OWL_MODELS[selector.removeprefix("base-")]
        values = GUIOwlModelConfig(model_id=identifier, model_revision=revision).model_dump()
    else:
        directory = Path(checkpoint).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"GUI-Owl checkpoint directory does not exist: {directory}")
        if (directory / "adapter_config.json").is_file():
            adapter_path = directory
        elif (directory / "adapter" / "adapter_config.json").is_file():
            adapter_path = directory / "adapter"
        else:
            raise ValueError(
                f"No completed adapter in {directory}; pass an explicit checkpoints/checkpoint-N "
                "directory to evaluate an intermediate checkpoint"
            )
        metadata_path = adapter_path / TRAINING_METADATA
        if not metadata_path.is_file():
            raise ValueError(f"Missing GUI-Owl training provenance: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict) or metadata.get("schema_version") != TRAINING_SCHEMA:
            raise ValueError("Checkpoint is not a scriptWoB GUI-Owl training checkpoint")
        if not isinstance(metadata.get("model"), dict):
            raise ValueError("GUI-Owl training metadata must contain the saved model configuration")
        values = GUIOwlModelConfig.model_validate(metadata["model"]).model_dump()
        expected_hashes = metadata.get("adapter_hashes")
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(ADAPTER_FILES):
            raise ValueError("Checkpoint metadata is missing required adapter_hashes")
        if adapter_hashes(adapter_path) != expected_hashes:
            raise ValueError("Checkpoint adapter checksum mismatch")
    for key, value in (
        ("device", device), ("quantization", quantization), ("precision", precision),
    ):
        if value is not None:
            values[key] = value
    config = GUIOwlModelConfig.model_validate(values)
    if adapter_path is not None:
        _validate_adapter(adapter_path, config)
    return config, adapter_path, metadata
