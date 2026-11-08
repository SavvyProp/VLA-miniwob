"""Single-GPU SeeClick LoRA training using the native Transformers/PEFT runtime."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .collection import _writer_lock, implementation_digest, write_json
from .export import export_dataset
from .prompt_styles import instruction_version, validate_prompt_style
from .seeclick_data import SeeClickCollator, SeeClickDataset, load_training_export
from .seeclick_prompt import PROMPT_VERSION
from .seeclick_runtime import SeeClickModelConfig, load_model, load_tokenizer

METADATA_FILE = "scriptwob_training.json"


class SeeClickTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: SeeClickModelConfig = Field(default_factory=SeeClickModelConfig)
    max_length: int = Field(default=2048, ge=512, le=8192)
    epochs: float = Field(default=3, gt=0, allow_inf_nan=False)
    max_steps: int = Field(default=-1, ge=-1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.1, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1, allow_inf_nan=False)
    max_grad_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    seed: int = Field(default=2026, ge=0, lt=2**32)

    @model_validator(mode="after")
    def valid_steps(self):
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 (use epochs) or a positive optimizer-step count")
        if self.model.device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError(
                "Single-GPU training uses cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES"
            )
        return self


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_versions() -> dict:
    result = {}
    for package in ("torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def checkpoint_hashes(path: Path) -> dict[str, str]:
    """Bind evaluation to the actual saved adapter, never its mutable path alone."""
    names = ("adapter_config.json", "adapter_model.safetensors")
    if any(not (path / name).is_file() for name in names):
        raise ValueError(f"Incomplete SeeClick adapter checkpoint: {path}")
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names}


def write_checkpoint_metadata(path: Path, metadata: dict) -> None:
    write_json(path / METADATA_FILE, {**metadata, "adapter_hashes": checkpoint_hashes(path)})


def latest_checkpoint(output: Path) -> Path | None:
    """Resume only complete checkpoints written with optimizer, scheduler and RNG state."""
    candidates = []
    for path in (output / "checkpoints").glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if not path.is_dir() or not suffix.isdigit():
            continue
        required = (
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "adapter_config.json",
            "adapter_model.safetensors",
            METADATA_FILE,
        )
        if all((path / name).is_file() for name in required):
            candidates.append((int(suffix), path))
    return max(candidates, default=(0, None))[1]


def _training_stack():
    try:
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "SeeClick requires its separate runtime. Run uv sync --project integrations/seeclick "
            "--extra cuda, then uv run --project integrations/seeclick --extra cuda scriptwob ..."
        ) from exc
    return Trainer, TrainerCallback, TrainingArguments, set_seed


def _arguments(config: SeeClickTrainingConfig, output: Path, training_arguments):
    return training_arguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=config.model.precision == "bf16",
        fp16=config.model.precision == "fp16",
        gradient_checkpointing=config.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_safetensors=True,
        logging_steps=config.logging_steps,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=config.model.device.startswith("cuda"),
        seed=config.seed,
        data_seed=config.seed,
        label_names=["labels"],
        use_cpu=config.model.device == "cpu",
    )


def train_seeclick(
    source: str | Path,
    output: str | Path,
    config: SeeClickTrainingConfig | None = None,
    *,
    resume: bool = False,
    prepare_only: bool = False,
    allow_oracle: bool = False,
    prompt_style: str | None = None,
) -> dict:
    """Fine-tune pretrained SeeClick from a collection run or a validated SFT export.

    Collection input is exported once into output/dataset. Resume reuses this
    snapshot. Export-folder input stays in place and must retain the same content.
    A larger positive max_steps is allowed on resume and recorded explicitly.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    config = config or SeeClickTrainingConfig()
    if prompt_style is not None:
        validate_prompt_style(prompt_style)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This SeeClick training entry point supports one process and one GPU")
    if source == output or source.is_relative_to(output):
        raise ValueError("Training output must not contain or replace its input data")
    if not source.is_dir():
        raise ValueError(f"Training source does not exist: {source}")
    is_collection = (source / "run.json").is_file()
    if not is_collection and not (source / "dataset.json").is_file():
        raise ValueError("Source must be a scriptWoB collection run or a SeeClick export")
    # Invalid external data must not create an output directory that blocks a retry.
    preloaded_export = (
        None if is_collection else load_training_export(source, allow_oracle=allow_oracle)
    )
    if preloaded_export is not None and prompt_style is not None:
        if preloaded_export.metadata.get("prompt_style", "natural") != prompt_style:
            raise ValueError(
                "Export prompt style differs from --prompt-style; use reformat-seeclick "
                "with the original manifest to convert the existing demonstrations first"
            )
    if output.exists() and not output.is_dir():
        raise ValueError("Training output must be a directory")
    if output.exists() and any(p.name != ".writer.lock" for p in output.iterdir()) and not resume:
        raise ValueError("Training output already exists; use --resume or a new directory")
    if resume and not (output / METADATA_FILE).is_file():
        raise ValueError("No training metadata to resume; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output):
        metadata_path = output / METADATA_FILE
        previous = json.loads(metadata_path.read_text()) if resume else None
        if previous and previous["source"] != str(source):
            raise ValueError("Resume source differs from the original source")
        if is_collection:
            dataset_path = output / "dataset"
            if not dataset_path.exists():
                export_dataset(
                    source, dataset_path, format="seeclick", allow_oracle=allow_oracle,
                    prompt_style=prompt_style or "natural",
                )
        else:
            dataset_path = source
        exported = preloaded_export or load_training_export(dataset_path, allow_oracle=allow_oracle)
        selected_style = exported.metadata.get("prompt_style", "natural")
        if prompt_style is not None and selected_style != prompt_style:
            raise ValueError("Frozen dataset prompt style differs; use a new training output")
        identity_config = config.model_dump(mode="json")
        identity_config.pop("max_steps")
        identity = {
            "config": identity_config,
            "dataset_fingerprint": exported.fingerprint,
            "implementation_digest": implementation_digest(),
            "prompt_version": PROMPT_VERSION,
            "prompt_style": selected_style,
            "instruction_version": instruction_version(selected_style),
            "runtime_versions": runtime_versions(),
            "allow_oracle": allow_oracle,
        }
        fingerprint = _digest(identity)
        limit_history = previous.get("limit_history", []) if previous else []
        if previous:
            if previous["training_fingerprint"] != fingerprint:
                raise ValueError(
                    "Resume data, model, configuration, source code or runtime differs"
                )
            old_limit = previous["config"]["max_steps"]
            if old_limit != config.max_steps:
                if config.max_steps <= 0 or (old_limit > 0 and config.max_steps < old_limit):
                    raise ValueError("Resume may only increase the positive max_steps target")
                limit_history.append(
                    {"utc": _now(), "old_max_steps": old_limit, "new_max_steps": config.max_steps}
                )
        metadata = {
            "schema_version": "scriptwob-seeclick-training-v1",
            "created_utc": previous["created_utc"] if previous else _now(),
            "updated_utc": _now(),
            "source": str(source),
            "dataset_path": str(dataset_path),
            "model": config.model.model_dump(mode="json"),
            "max_length": config.max_length,
            "config": config.model_dump(mode="json"),
            "dataset": {"fingerprint": exported.fingerprint, "metadata": exported.metadata},
            "identity": identity,
            "training_fingerprint": fingerprint,
            "limit_history": limit_history,
            "prompt_version": PROMPT_VERSION,
            "prompt_style": selected_style,
            "instruction_version": instruction_version(selected_style),
            "status": "preparing",
            "optimization_started": (
                previous.get("optimization_started", False) if previous else False
            ),
        }
        write_json(metadata_path, metadata)
        try:
            tokenizer = load_tokenizer(config.model)
            dataset = SeeClickDataset(exported, tokenizer, config.max_length)
            lengths = dataset.token_lengths
            prompt_lengths = [
                next(index for index, label in enumerate(sample["labels"]) if label != -100)
                for sample in dataset.samples
            ]
            # Evaluation reserves 64 generation tokens by default. An example that
            # barely fits with a shorter gold answer must not pass this preflight.
            if max(prompt_lengths) + 64 > config.max_length:
                raise ValueError(
                    f"Longest prompt has {max(prompt_lengths)} tokens and needs 64 output tokens; "
                    f"increase max_length above {config.max_length}. No context is truncated."
                )
            preflight = {
                "examples": len(dataset),
                "images": len(exported.image_hashes),
                "tokens_min": min(lengths),
                "tokens_max": max(lengths),
                "tokens_mean": sum(lengths) / len(lengths),
                "prompt_tokens_max": max(prompt_lengths),
                "reserved_output_tokens": 64,
                "max_length": config.max_length,
                "truncated_examples": 0,
                "dataset_fingerprint": exported.fingerprint,
            }
            write_json(output / "data_preflight.json", preflight)
            metadata.update(status="prepared", preflight=preflight, updated_utc=_now())
            write_json(metadata_path, metadata)
            if prepare_only:
                return {"output": str(output), "status": "prepared", **preflight}
            trainer_type, callback_type, arguments_type, set_seed = _training_stack()
            checkpoint = latest_checkpoint(output) if resume else None
            if previous and previous.get("optimization_started", False) and checkpoint is None:
                raise ValueError("No complete optimizer checkpoint to resume; use a new output")
            if checkpoint is not None:
                saved = json.loads((checkpoint / METADATA_FILE).read_text())
                if saved["training_fingerprint"] != fingerprint:
                    raise ValueError("Checkpoint belongs to a different training run")
                if saved.get("adapter_hashes") != checkpoint_hashes(checkpoint):
                    raise ValueError("Checkpoint adapter integrity check failed")
                step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
                if config.max_steps > 0 and step >= config.max_steps:
                    raise ValueError("Checkpoint already reached max_steps; increase the target")
            set_seed(config.seed)
            model = load_model(config.model, training=True)
            parameters = {
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "parameters_in_memory": sum(p.numel() for p in model.parameters()),
            }
            if parameters["trainable_parameters"] == 0:
                raise ValueError("SeeClick has no trainable adapter parameters")
            metadata.update(status="training", parameters=parameters, updated_utc=_now())
            write_json(metadata_path, metadata)

            class MetadataCallback(callback_type):
                def on_save(self, args, state, control, **kwargs):
                    path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    write_checkpoint_metadata(path, metadata)
                    return control

            trainer = trainer_type(
                model=model,
                args=_arguments(config, output, arguments_type),
                train_dataset=dataset,
                data_collator=SeeClickCollator(tokenizer.pad_token_id),
                callbacks=[MetadataCallback()],
            )
            metadata.update(optimization_started=True, updated_utc=_now())
            write_json(metadata_path, metadata)
            result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            # The pinned Trainer saves optimizer/scheduler/RNG state here. Persist
            # a final resumable checkpoint even when save_steps exceeds this run.
            final_checkpoint = output / "checkpoints" / f"checkpoint-{trainer.state.global_step}"
            if latest_checkpoint(output) != final_checkpoint:
                trainer._save_checkpoint(trainer.model, trial=None)
            write_checkpoint_metadata(final_checkpoint, metadata)
            adapter = output / "adapter"
            model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
            metadata.update(
                status="complete",
                global_step=trainer.state.global_step,
                metrics=result.metrics,
                updated_utc=_now(),
            )
            write_checkpoint_metadata(adapter, metadata)
            write_json(metadata_path, metadata)
            summary = {
                "output": str(output),
                "adapter": str(adapter),
                "status": "complete",
                "global_step": trainer.state.global_step,
                **parameters,
                **preflight,
                "metrics": result.metrics,
            }
            write_json(output / "train_results.json", summary)
            return summary
        except BaseException as exc:
            metadata.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__,
                updated_utc=_now(),
            )
            try:
                write_json(metadata_path, metadata)
            except Exception as metadata_error:
                exc.add_note(f"Could not update training metadata: {type(metadata_error).__name__}")
            raise
