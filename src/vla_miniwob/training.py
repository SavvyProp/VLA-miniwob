"""Single-device, resumable TinyVLA diffusion training on frozen GUI trajectories."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .collection import implementation_digest, write_json
from .training_data import prepare_training_data, prepare_validation_data, sampling_weights


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    non_lora_lr: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    save_every: int = Field(default=250, ge=1)
    validate_every: int = Field(default=250, ge=1)
    validation_batches: int = Field(default=32, ge=1)
    seed: int = Field(default=2026, ge=0, lt=2**32)
    balanced_sampling: bool = True
    gradient_checkpointing: bool = True


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def resolve_checkpoint(path: Path) -> Path:
    path = Path(path).resolve()
    if (path / "training_run.json").is_file():
        # A process can die after atomically publishing a checkpoint but before
        # updating latest.json. Complete directories are authoritative.
        candidates = [
            item
            for item in path.glob("checkpoint-*")
            if item.is_dir()
            and item.resolve().parent == path
            and item.name.removeprefix("checkpoint-").isdigit()
            and (item / "checkpoint.json").is_file()
        ]
        if candidates:
            path = max(candidates, key=lambda item: int(item.name.split("-")[-1]))
        elif (path / "latest.json").is_file():
            raise ValueError("Training checkpoint pointer has no complete checkpoint")
    if not (path / "checkpoint.json").is_file():
        raise ValueError(f"No complete TinyVLA checkpoint at {path}")
    return path


def verify_checkpoint(path: Path) -> dict:
    metadata = json.loads((path / "checkpoint.json").read_text())
    if metadata.get("format") != 1 or not metadata.get("files"):
        raise ValueError("Unsupported or empty training checkpoint")
    payloads = {
        str(item.relative_to(path))
        for item in path.rglob("*")
        if item.is_file() and item != path / "checkpoint.json"
    }
    if set(metadata["files"]) != payloads:
        raise ValueError("Checkpoint checksum manifest does not cover every payload file")
    for name, expected in metadata["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError(f"Missing/unsafe checkpoint file: {name}")
        if file_digest(target) != expected:
            raise ValueError(f"Checkpoint file checksum differs: {name}")
    return metadata


def _rng_state(torch, sampler, noise):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.get_state(),
        "noise": noise.get_state(),
    }


def _restore_rng(torch, state, sampler, noise):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    sampler.set_state(state["sampler"].cpu())
    noise.set_state(state["noise"].cpu())


def _save_checkpoint(output, step, backend, optimizer, scaler, rng, run_config, stats):
    import torch

    name = f"checkpoint-{step:07d}"
    destination, temporary = output / name, output / f".{name}.tmp"
    if destination.exists():
        raise ValueError(f"Refusing to replace an existing checkpoint: {destination}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        backend.save(temporary)
        write_json(temporary / "normalization.json", stats)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng,
                "step": step,
            },
            temporary / "training_state.pt",
        )
        files = {
            str(path.relative_to(temporary)): file_digest(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        write_json(
            temporary / "checkpoint.json",
            {
                "format": 1,
                "step": step,
                "run": run_config,
                "files": files,
            },
        )
        temporary.rename(destination)
        write_json(output / "latest.json", {"checkpoint": name, "step": step})
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _preflight(backend, prepared, output, label):
    # Inspect every *full* context with the real tokenizer; do not truncate history.
    import h5py

    largest, count = 0, 0
    for episode in prepared.dataset.metadata["episodes"]:
        with h5py.File(prepared.root / episode["file"], "r") as file:
            for text in file["language_per_timestep"].asstr()[:]:
                ids = backend.processor.text_ids(text)
                largest = max(largest, len(ids) + backend.image_tokens - 1)
                count += 1
    # A representative single-control and longest composition preview, after
    # actual resizing/padding. Invert CLIP normalization only for display.
    offsets, offset = [], 0
    for episode in prepared.dataset.metadata["episodes"]:
        offsets.append((episode.get("component_count", 0), offset))
        offset += episode["length"]
    for index in sorted({min(offsets)[1], max(offsets)[1]}):
        pixels = backend.processor.image(prepared.dataset[index]["image"]).float().cpu().numpy()
        processor = backend.processor.image_processor
        mean = np.asarray(processor.image_mean)[:, None, None]
        std = np.asarray(processor.image_std)[:, None, None]
        display = np.clip((pixels * std + mean) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(display.transpose(1, 2, 0)).save(output / f"{label}-vision-{index}.png")
    return {
        "chunks": count,
        "max_multimodal_tokens": largest,
        "visual_tokens": backend.image_tokens,
        "context_capacity": backend.processor.max_length,
    }


def _validate(backend, data, settings, device):
    import torch

    devices = [device.index or 0] if device.type == "cuda" else []
    losses, examples = 0.0, 0
    backend.model.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(settings.seed + 1)
            for start in range(
                0,
                min(len(data), settings.validation_batches * settings.batch_size),
                settings.batch_size,
            ):
                samples = [
                    backend.processor(data[i])
                    for i in range(start, min(start + settings.batch_size, len(data)))
                ]
                value = float(backend.loss(backend.collate(samples), generator=generator))
                if not np.isfinite(value):
                    raise ValueError("Nonfinite development diffusion loss")
                losses += value * len(samples)
                examples += len(samples)
    finally:
        backend.model.train()
    return {"dev_epsilon_loss": losses / examples, "dev_chunks": examples}


def train(
    source: Path,
    output: Path,
    recipe: dict,
    *,
    dev_source: Path | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    resume: bool = False,
    preflight_only: bool = False,
    backend_class=None,
    progress=print,
) -> dict:
    """Train optimizer steps, with exact local sampler/RNG restoration on resume.

    The optional backend_class is an injection point for lightweight regression
    tests, not a replacement model exposed in the production CLI.
    """
    import torch

    if backend_class is None:
        from .tinyvla_model import TinyVLABackend

        backend_class = TinyVLABackend
    settings = TrainingSettings.model_validate(recipe.get("training", {}))
    recipe = {**recipe, "training": settings.model_dump()}
    output = Path(output).resolve()
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("TinyVLA training supports cpu or cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install the cuda environment and check the driver")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Training supports float32 or bfloat16; float16 AdamW is not supported")
    if device.type == "cpu" and dtype != "float32":
        raise ValueError("Use --dtype float32 for CPU training")
    if device.type == "cuda" and dtype == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU lacks bfloat16 support; use --dtype float32")
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Training output exists; pass --resume or choose a new directory")
    if resume and not (output / "training_run.json").is_file():
        raise ValueError("No training run to resume in this output directory")
    checkpoint = (
        resolve_checkpoint(output)
        if resume
        and (any(output.glob("checkpoint-*/checkpoint.json")) or (output / "latest.json").exists())
        else None
    )
    saved = verify_checkpoint(checkpoint) if checkpoint else None
    previous_run = (
        saved["run"]
        if saved
        else json.loads((output / "training_run.json").read_text())
        if resume
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    progress("Preparing accepted training trajectories and checking frozen splits...", flush=True)
    prepared = prepare_training_data(
        source, output / "data" / "train", horizon=recipe["chunk_size"]
    )
    development = (
        prepare_validation_data(dev_source, output / "data" / "dev", training=prepared)
        if dev_source
        else None
    )
    weights, sampling = sampling_weights(
        prepared.dataset.metadata, balanced=settings.balanced_sampling
    )
    run_config = {
        "format": 1,
        "recipe": recipe,
        "device": str(device),
        "dtype": dtype,
        "training_fingerprint": prepared.fingerprint,
        "development_fingerprint": development.fingerprint if development else None,
        "sampling": sampling,
        "normalization": prepared.dataset.normalizer.stats,
        "implementation_digest": implementation_digest(),
        "runtime": {
            name: version(name) for name in ("torch", "transformers", "peft", "diffusers", "numpy")
        },
        "lr_schedule": "linear_warmup_then_constant",
    }
    if previous_run:
        previous = json.loads(json.dumps(previous_run))
        previous_steps = previous["recipe"]["training"]["max_steps"]
        previous["recipe"]["training"]["max_steps"] = settings.max_steps
        if previous != run_config or settings.max_steps < previous_steps:
            raise ValueError(
                "Resume settings/data/code differ; only increasing max_steps is allowed"
            )
        if saved and saved["step"] >= settings.max_steps:
            return {"step": saved["step"], "checkpoint": str(checkpoint), "already_complete": True}
    # An interrupted weight download/model preflight can restart from the same
    # verified dataset and seed even before the first optimizer checkpoint.
    write_json(output / "training_run.json", run_config)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    progress("Loading pinned TinyVLA-B backbone and diffusion decoder...", flush=True)
    backend = (
        backend_class.from_checkpoint(checkpoint, device=str(device), dtype=dtype, training=True)
        if checkpoint
        else backend_class.from_pretrained(recipe, device=str(device), dtype=dtype, training=True)
    )
    # A resumed run may extend its step budget; persist the current recipe too.
    backend.recipe = recipe
    if settings.gradient_checkpointing:
        backend.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    audit = {"train": _preflight(backend, prepared, output, "train")}
    if development:
        audit["dev"] = _preflight(backend, development, output, "dev")
    parameters = [
        (name, parameter)
        for name, parameter in backend.model.named_parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("TinyVLA has no trainable parameters")
    audit["trainable_parameters"] = [
        {
            "name": name,
            "numel": parameter.numel(),
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in parameters
    ]
    audit["trainable_count"] = sum(parameter.numel() for _, parameter in parameters)
    audit["trainable_bytes"] = sum(
        parameter.numel() * parameter.element_size() for _, parameter in parameters
    )
    audit["estimated_adam_moment_bytes"] = 2 * audit["trainable_bytes"]
    write_json(output / "preflight.json", audit)
    progress(
        f"Preflight passed: {len(prepared.dataset)} chunks, "
        f"{audit['trainable_count']:,} trainable parameters. Vision previews: {output}",
        flush=True,
    )
    if preflight_only:
        return {"preflight": str(output / "preflight.json"), "trained": False}
    groups = []
    for lora, lr in ((True, settings.learning_rate), (False, settings.non_lora_lr)):
        group = [value for name, value in parameters if ("lora_" in name) == lora]
        if group:
            groups.append({"params": group, "lr": lr, "initial_lr": lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    # Kept in checkpoints for a stable state format; bf16/fp32 need no scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = torch.Generator(device="cpu").manual_seed(settings.seed)
    noise = torch.Generator(device=device).manual_seed(settings.seed + 2)
    step = 0
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if state["step"] != saved["step"]:
            raise ValueError("Checkpoint optimizer step differs from metadata")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(torch, state["rng"], sampler, noise)
        step = state["step"]
    if resume and (output / "metrics.jsonl").exists():
        # Discard metrics for updates lost after the last complete checkpoint.
        rows = []
        for line in (output / "metrics.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted final write may be incomplete.
            if row["step"] <= step:
                rows.append(json.dumps(row))
        temporary = output / "metrics.jsonl.tmp"
        temporary.write_text("".join(row + "\n" for row in rows))
        temporary.replace(output / "metrics.jsonl")
    probabilities = torch.from_numpy(weights)
    backend.model.train()
    last_checkpoint = checkpoint
    first_step = step + 1
    for step in range(first_step, settings.max_steps + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        multiplier = min(1.0, step / max(1, settings.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        loss_sum = 0.0
        for _ in range(settings.gradient_accumulation):
            indices = torch.multinomial(
                probabilities, settings.batch_size, replacement=True, generator=sampler
            ).tolist()
            samples = [backend.processor(prepared.dataset[i]) for i in indices]
            loss = backend.loss(backend.collate(samples), generator=noise)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Nonfinite diffusion loss at step {step}; last checkpoint retained"
                )
            (loss / settings.gradient_accumulation).backward()
            loss_sum += float(loss.detach()) / settings.gradient_accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in parameters],
            settings.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        row = {
            "step": step,
            "epsilon_loss": loss_sum,
            "gradient_norm": float(gradient_norm),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "seconds": time.monotonic() - started,
        }
        if development and (step % settings.validate_every == 0 or step == settings.max_steps):
            row.update(_validate(backend, development.dataset, settings, device))
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        progress(json.dumps(row), flush=True)
        if step % settings.save_every == 0 or step == settings.max_steps:
            last_checkpoint = _save_checkpoint(
                output,
                step,
                backend,
                optimizer,
                scaler,
                _rng_state(torch, sampler, noise),
                run_config,
                prepared.dataset.normalizer.stats,
            )
    return {"step": step, "checkpoint": str(last_checkpoint), "trained": True}
