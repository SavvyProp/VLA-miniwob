"""Pinned original SeeClick/Qwen-VL loaders; heavyweight imports are deliberately lazy.

SeeClick supplies merged GUI-grounding weights, but its tokenizer comes from Qwen-VL-Chat.
The original model loads each local image from the path encoded in its input token IDs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEECLICK_MODEL = "cckevinn/SeeClick"
SEECLICK_REVISION = "b79618049920e209d15cda4d35004e32e8b485d9"
QWEN_TOKENIZER = "Qwen/Qwen-VL-Chat"
QWEN_TOKENIZER_REVISION = "f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
TRAINING_METADATA = "scriptwob_training.json"
# A suffix such as c_proj also matches the vision encoder. Restrict LoRA to language layers.
LORA_TARGETS = r"transformer\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(w1|w2|c_proj))"
RUNTIME_INSTALL = "uv sync --project integrations/seeclick --extra cuda --frozen"


class SeeClickModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = SEECLICK_MODEL
    model_revision: str = Field(default=SEECLICK_REVISION, pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str = QWEN_TOKENIZER
    tokenizer_revision: str = Field(default=QWEN_TOKENIZER_REVISION, pattern=r"^[0-9a-f]{40}$")
    quantization: Literal["4bit", "none"] = "4bit"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda(:[0-9]+)?)$")
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    gradient_checkpointing: bool = True

    @model_validator(mode="after")
    def validate_device(self) -> SeeClickModelConfig:
        if self.quantization == "4bit" and self.device == "cpu":
            raise ValueError("The SeeClick 4-bit runtime requires a CUDA GPU")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU SeeClick checks require precision='fp32'")
        if self.quantization == "4bit" and self.precision == "fp32":
            raise ValueError("4-bit SeeClick requires bf16 or fp16 compute precision")
        if not self.model_id.strip() or not self.tokenizer_id.strip():
            raise ValueError("Model and tokenizer identifiers cannot be empty")
        return self


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            f"SeeClick runtime dependencies are missing. Run: {RUNTIME_INSTALL}"
        ) from exc
    return torch, transformers, peft


def load_tokenizer(config: SeeClickModelConfig) -> Any:
    """Load only the independently pinned tokenizer, never any model weights."""
    try:
        import transformers.utils
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"SeeClick tokenizer dependencies are missing. Run: {RUNTIME_INSTALL}"
        ) from exc
    font_path = hf_hub_download(QWEN_TOKENIZER, "SimSun.ttf", revision=QWEN_TOKENIZER_REVISION)
    original_lookup = transformers.utils.try_to_load_from_cache

    def pinned_font_lookup(repo_id: str, filename: str, *args: Any, **kwargs: Any) -> Any:
        if repo_id == QWEN_TOKENIZER and filename == "SimSun.ttf":
            return font_path
        return original_lookup(repo_id, filename, *args, **kwargs)

    # The upstream tokenizer looks up this font at unpinned 'main' during import and
    # otherwise downloads it from a third-party host into cwd. Supply our pinned copy.
    with patch.object(transformers.utils, "try_to_load_from_cache", pinned_font_lookup):
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id,
            revision=config.tokenizer_revision,
            code_revision=config.tokenizer_revision,
            trust_remote_code=True,
            use_fast=False,
            padding_side="right",
        )
    # Qwen's eod token is already in the pretrained vocabulary; never resize image tokens.
    tokenizer.pad_token_id = tokenizer.eod_id
    return tokenizer


def _validate_hardware(config: SeeClickModelConfig, torch: Any) -> Any:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.precision]
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("SeeClick was configured for CUDA, but PyTorch cannot access a GPU")
        index = int(config.device.split(":")[1]) if ":" in config.device else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        if config.precision == "bf16":
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("This GPU does not support bf16; select precision='fp16'")
    return dtype


def _validate_adapter(adapter_path: Path, config: SeeClickModelConfig) -> None:
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError(f"Missing safe LoRA adapter weights in {adapter_path}")
    data = json.loads((adapter_path / "adapter_config.json").read_text())
    if data.get("peft_type") != "LORA":
        raise ValueError("Only SeeClick LoRA adapters are supported")
    if data.get("base_model_name_or_path") != config.model_id:
        raise ValueError("Adapter base model differs from its scriptWoB model configuration")
    if data.get("revision") not in (None, config.model_revision):
        raise ValueError("Adapter base revision differs from its scriptWoB model configuration")
    if data.get("target_modules") != LORA_TARGETS or data.get("modules_to_save"):
        raise ValueError("Adapter is not a scriptWoB language-only LoRA adapter")


def load_model(
    config: SeeClickModelConfig,
    *,
    training: bool = False,
    adapter_path: str | Path | None = None,
) -> Any:
    """Load pretrained SeeClick, optionally attaching trainable or frozen LoRA weights.

    This deliberately keeps the visual encoder in its original floating-point format.
    Its custom forward reads projection weight dtypes, which packed 4-bit weights break.
    No device_map='auto' or CPU offload is used in the single-GPU training configuration.
    """
    torch, transformers, peft = _dependencies()
    dtype = _validate_hardware(config, torch)
    if adapter_path is not None:
        adapter_path = Path(adapter_path).resolve()
        _validate_adapter(adapter_path, config)

    kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "code_revision": config.model_revision,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        # Accelerate 0.27 compares cuda:0 with its unindexed cuda device. Its
        # GPU-zero allowance requires integer 0, not the string "cuda:0".
        # Resolve bare "cuda" to the same GPU zero checked by _validate_hardware.
        "device_map": {
            "": int(config.device.partition(":")[2] or 0)
            if config.device.startswith("cuda") else config.device
        },
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "bf16": config.precision == "bf16",
        "fp16": config.precision == "fp16",
        "fp32": config.precision == "fp32",
        "use_flash_attn": False,
    }
    if config.quantization == "4bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"4-bit SeeClick requires bitsandbytes. Run: {RUNTIME_INSTALL}"
            ) from exc
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["transformer.visual", "lm_head"],
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "visual"):
        raise ValueError("This loader requires the original SeeClick/Qwen-VL architecture")
    model.transformer.visual.requires_grad_(False)
    model.config.use_cache = not training

    if training:
        if config.quantization == "4bit":
            model = peft.prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing
            )
            # PEFT upcasts every unquantized parameter to fp32. The frozen 1.9B vision
            # encoder and two large vocabulary matrices needlessly double VRAM that way.
            model.transformer.visual.to(dtype=dtype)
            model.get_input_embeddings().to(dtype=dtype)
            model.get_output_embeddings().to(dtype=dtype)
        else:
            model.requires_grad_(False)
            if config.gradient_checkpointing:
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable()

    if adapter_path is not None:
        model = peft.PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training)
    elif training:
        targets = [name for name, _ in model.named_modules() if re.fullmatch(LORA_TARGETS, name)]
        if not targets:
            raise ValueError("No original Qwen language projections matched the LoRA configuration")
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
        # Keep optimizer parameters in fp32 even for unquantized fp16 LoRA. AMP's
        # GradScaler rejects fp16 gradients, while autocast handles forward compute.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=torch.float32)
        bad = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ("lora_" not in name or ".visual." in name)
        ]
        if bad:
            raise RuntimeError(f"Unexpected trainable parameters in frozen SeeClick: {bad[:5]}")
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("SeeClick has no trainable LoRA parameters")
        model.train()
    else:
        model.eval()
    return model


def resolve_checkpoint(
    checkpoint: str | Path = "base",
    *,
    device: str | None = None,
    quantization: str | None = None,
    precision: str | None = None,
) -> tuple[SeeClickModelConfig, Path | None, dict[str, Any]]:
    """Resolve an explicit checkpoint or a completed run's adapter, never 'latest'."""
    adapter_path: Path | None = None
    metadata: dict[str, Any] = {}
    if str(checkpoint) == "base":
        values = SeeClickModelConfig().model_dump()
    else:
        directory = Path(checkpoint).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"SeeClick checkpoint directory does not exist: {directory}")
        if (directory / "adapter_config.json").is_file():
            adapter_path = directory
        elif (directory / "adapter" / "adapter_config.json").is_file():
            adapter_path = directory / "adapter"
        else:
            raise ValueError(
                f"No completed adapter in {directory}; pass an explicit checkpoints/checkpoint-N "
                "directory to evaluate an intermediate checkpoint"
            )
        # Native outputs include their own immutable metadata. Do not borrow run-level
        # metadata for an arbitrary adapter, since its saved settings may differ.
        metadata_path = adapter_path / TRAINING_METADATA
        if not metadata_path.is_file():
            raise ValueError(f"Missing SeeClick training provenance: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict) or not isinstance(metadata.get("model"), dict):
            raise ValueError(
                "SeeClick training metadata must contain the saved model configuration"
            )
        values = SeeClickModelConfig.model_validate(metadata["model"]).model_dump()
    for key, value in (
        ("device", device),
        ("quantization", quantization),
        ("precision", precision),
    ):
        if value is not None:
            values[key] = value
    config = SeeClickModelConfig.model_validate(values)
    if adapter_path is not None:
        _validate_adapter(adapter_path, config)
    return config, adapter_path, metadata
