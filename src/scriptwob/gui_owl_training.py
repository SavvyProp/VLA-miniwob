"""Single-GPU GUI-Owl QLoRA SFT over existing, validated expert trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import Field

from .collection import _writer_lock, implementation_digest, write_json
from .export import export_dataset
from .gui_owl_data import GUIOwlCollator, GUIOwlDataset
from .gui_owl_prompt import PROMPT_VERSION
from .gui_owl_runtime import GUIOwlModelConfig, load_model, load_processor
from .prompt_styles import instruction_version, validate_prompt_style
from .seeclick_data import load_training_export
from .seeclick_training import (
    METADATA_FILE,
    SeeClickTrainingConfig,
    _arguments,
    _digest,
    _now,
    checkpoint_hashes,
    latest_checkpoint,
    runtime_versions,
    write_checkpoint_metadata,
)


class GUIOwlTrainingConfig(SeeClickTrainingConfig):
    """Share optimizer settings with SeeClick; keep model/input contracts separate."""

    model: GUIOwlModelConfig = Field(default_factory=GUIOwlModelConfig)
    max_length: int = Field(default=8192, ge=512, le=32768)
    history_images: int = Field(default=3, ge=0, le=32)


def _training_stack():
    try:
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "Use GUI-Owl's runtime: uv run --project integrations/gui_owl --extra cuda "
            "--frozen scriptwob train-gui-owl ..."
        ) from exc
    return Trainer, TrainerCallback, TrainingArguments, set_seed


def parameter_counts(model) -> dict[str, int]:
    """PEFT counts packed 4-bit weights correctly, unlike a naive numel sum."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters_including_adapter": total}


def train_gui_owl(
    source: str | Path,
    output: str | Path,
    config: GUIOwlTrainingConfig | None = None,
    *,
    resume: bool = False,
    prepare_only: bool = False,
    allow_oracle: bool = False,
    prompt_style: str | None = None,
) -> dict:
    """Train from a collection's train split or reuse its existing SeeClick export.

    Image history is reconstructed solely from earlier steps of the same episode.
    The format called 'seeclick' is the existing interchange format, not the
    tokenizer or model used here. No new expert calls or task regeneration occur.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    config = config or GUIOwlTrainingConfig()
    if (source / "suite.json").is_file():
        source = source / "train"
    if prompt_style is not None:
        validate_prompt_style(prompt_style)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("GUI-Owl training supports one process and one GPU")
    if source == output or source.is_relative_to(output):
        raise ValueError("Training output must not contain or replace its input data")
    if not source.is_dir():
        raise ValueError(f"Training source does not exist: {source}")
    is_collection = (source / "run.json").is_file()
    if not is_collection and not (source / "dataset.json").is_file():
        raise ValueError("Source must be a train collection, collection suite, or SeeClick export")
    exported = None if is_collection else load_training_export(source, allow_oracle=allow_oracle)
    if exported and prompt_style is not None:
        if exported.metadata.get("prompt_style", "natural") != prompt_style:
            raise ValueError(
                "Export prompt style differs; use reformat-seeclick with the original manifest "
                "to convert the existing demonstrations first"
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
        if previous and previous.get("schema_version") != "scriptwob-gui-owl-training-v1":
            raise ValueError("Resume requires a GUI-Owl training run")
        if previous and previous["source"] != str(source):
            raise ValueError("Resume source differs from the original source")
        dataset_path = output / "dataset" if is_collection else source
        if is_collection and not dataset_path.exists():
            export_dataset(
                source, dataset_path, format="seeclick", allow_oracle=allow_oracle,
                prompt_style=prompt_style or "natural",
            )
        exported = exported or load_training_export(dataset_path, allow_oracle=allow_oracle)
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
        limit_history = list(previous.get("limit_history", [])) if previous else []
        if previous:
            if previous["training_fingerprint"] != fingerprint:
                raise ValueError(
                    "Resume data, model, history, configuration, code or runtime differs"
                )
            old_limit = previous["config"]["max_steps"]
            if old_limit != config.max_steps:
                if config.max_steps <= 0 or (old_limit > 0 and config.max_steps < old_limit):
                    raise ValueError("Resume may only increase the positive max_steps target")
                limit_history.append(
                    {"utc": _now(), "old_max_steps": old_limit, "new_max_steps": config.max_steps}
                )
        metadata = {
            "schema_version": "scriptwob-gui-owl-training-v1",
            "created_utc": previous["created_utc"] if previous else _now(),
            "updated_utc": _now(),
            "source": str(source),
            "dataset_path": str(dataset_path),
            "model": config.model.model_dump(mode="json"),
            "max_length": config.max_length,
            "history_images": config.history_images,
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
            processor = load_processor(config.model)
            dataset = GUIOwlDataset(
                exported, processor, config.max_length, history_images=config.history_images,
                reserve_tokens=64,
            )
            preflight = {
                **dataset.preflight(), "max_length": config.max_length,
                "history_images": config.history_images,
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
            parameters = parameter_counts(model)
            if parameters["trainable_parameters"] == 0:
                raise ValueError("GUI-Owl has no trainable adapter parameters")
            metadata.update(status="training", parameters=parameters, updated_utc=_now())
            write_json(metadata_path, metadata)

            class MetadataCallback(callback_type):
                def on_save(self, args, state, control, **kwargs):
                    path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    write_checkpoint_metadata(path, {
                        **metadata, "global_step": state.global_step, "epoch": state.epoch,
                    })
                    return control

            trainer = trainer_type(
                model=model,
                args=_arguments(config, output, arguments_type),
                train_dataset=dataset,
                data_collator=GUIOwlCollator(processor),
                callbacks=[MetadataCallback()],
            )
            metadata.update(optimization_started=True, updated_utc=_now())
            write_json(metadata_path, metadata)
            result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            metadata.update(
                status="complete", global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                metrics=result.metrics, updated_utc=_now(),
            )
            final_checkpoint = output / "checkpoints" / f"checkpoint-{trainer.state.global_step}"
            if latest_checkpoint(output) != final_checkpoint:
                trainer._save_checkpoint(trainer.model, trial=None)
            write_checkpoint_metadata(final_checkpoint, metadata)
            adapter = output / "adapter"
            model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
            write_checkpoint_metadata(adapter, metadata)
            write_json(metadata_path, metadata)
            summary = {
                "output": str(output), "adapter": str(adapter), "status": "complete",
                "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
                **parameters, **preflight, "metrics": result.metrics,
            }
            write_json(output / "train_results.json", summary)
            return summary
        except BaseException as exc:
            metadata.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__, updated_utc=_now(),
            )
            try:
                write_json(metadata_path, metadata)
            except Exception as metadata_error:
                exc.add_note(f"Could not update metadata: {type(metadata_error).__name__}")
            raise
