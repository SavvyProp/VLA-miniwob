"""Prepare auditable train/dev chunks without exposing split metadata to the policy."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .collection import write_json
from .contracts import CONTRACT_VERSION
from .dataset import ChunkDataset, Normalizer, accepted_episodes, export_hdf5, read_rows


@dataclass
class PreparedData:
    root: Path
    dataset: ChunkDataset
    fingerprint: str
    sampling_weights: np.ndarray
    sampling: dict


def _contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Dataset file paths must be relative to the dataset directory")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Dataset file is missing or escapes its directory: {name}")
    return path


def _fingerprint(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        name = str(path.relative_to(root)).encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _raw_fingerprint(source: Path, split: str) -> str:
    paths = [_contained(source, "run.json")]
    for directory in accepted_episodes(source, split):
        if not directory.resolve().is_relative_to(source):
            raise ValueError("Collection episode escapes its run directory")
        for name in ("private.json", "episode.json", "transitions.jsonl"):
            paths.append(_contained(directory, name))
        paths.extend(_contained(directory, row["image"]) for row in read_rows(directory))
    return _fingerprint(source, paths)


def _hdf5_dataset(file: h5py.File, key: str) -> h5py.Dataset:
    group = file
    parts = key.split("/")
    for part in parts:
        # Do not follow external or symbolic links inside an untrusted export.
        if not isinstance(group, h5py.Group) or not isinstance(
            group.get(part, getlink=True), h5py.HardLink
        ):
            raise ValueError(f"Missing or linked HDF5 field: {key}")
        group = group[part]
    if not isinstance(group, h5py.Dataset):
        raise ValueError(f"Expected HDF5 dataset: {key}")
    if group.is_virtual or group.external:
        raise ValueError(f"External HDF5 storage is unsupported: {key}")
    return group


def validate_export(root: Path, split: str) -> tuple[dict, str]:
    """Validate policy inputs once, before workers or model loading are started."""
    root = Path(root).resolve()
    index_path, stats_path = (
        _contained(root, name) for name in ("index.json", "normalization.json")
    )
    metadata = json.loads(index_path.read_text())
    if metadata.get("version") != 1 or metadata.get("split") != split:
        raise ValueError(f"Expected a version-1 {split} export; held-out eval cannot train/select")
    Normalizer(json.loads(stats_path.read_text()))
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Training and validation exports must contain accepted episodes")
    files, identities, paths = set(), set(), [index_path, stats_path]
    for episode in episodes:
        path = _contained(root, episode["file"])
        length, identity = episode.get("length"), episode.get("source_episode")
        if type(length) is not int or length < 1:
            raise ValueError("Episode lengths must be positive integers")
        if not isinstance(identity, str) or not identity or identity in identities or path in files:
            raise ValueError("Duplicate or missing episode identity/file in export")
        identities.add(identity)
        files.add(path)
        paths.append(path)
        with h5py.File(path, "r") as file:
            if file.attrs.get("action_contract") != CONTRACT_VERSION:
                raise ValueError("Export action contract differs from the GUI benchmark")
            for key in ("action", "observations/qpos"):
                array = _hdf5_dataset(file, key)
                if array.shape != (length, 3):
                    raise ValueError(f"Misaligned three-channel episode field: {key}")
                values = array[:]
                if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError(f"Episode field must be finite physical [0,1] values: {key}")
            images = _hdf5_dataset(file, "observations/images/screen")
            if (
                images.ndim != 4
                or images.shape[0] != length
                or images.shape[-1] != 3
                or min(images.shape[1:3]) < 2
                or images.dtype != np.dtype("uint8")
            ):
                raise ValueError("Expected one RGB uint8 screenshot per transition")
            language = _hdf5_dataset(file, "language_per_timestep")
            if language.shape != (length,) or h5py.check_string_dtype(language.dtype) is None:
                raise ValueError("Expected one language context per transition")
    return metadata, _fingerprint(root, paths)


def sampling_weights(metadata: dict, *, balanced: bool = True) -> tuple[np.ndarray, dict]:
    """Equal mass per complexity, task, then order/wording cell; uniform within a cell.

    Legacy exports have no task metadata. They remain usable, with an explicitly
    recorded uniform-transition fallback rather than guessed task labels.
    """
    episodes = metadata["episodes"]
    total = sum(item["length"] for item in episodes)
    if total < 1:
        raise ValueError("Cannot sample an empty export")
    complete = all(
        isinstance(item.get("task"), str)
        and item["task"]
        and type(item.get("component_count")) is int
        and item["component_count"] > 0
        and item.get("order") in ("forward", "reverse")
        and item.get("wording") in ("direct", "reverse_clauses")
        for item in episodes
    )
    if not balanced or not complete:
        return np.full(total, 1 / total, dtype=np.float64), {
            "mode": "uniform_transitions",
            "reason": "requested" if not balanced else "legacy_export_missing_sampling_metadata",
        }
    lengths = Counter()
    tasks, cells = defaultdict(set), defaultdict(set)
    for item in episodes:
        count, task = item["component_count"], item["task"]
        cell = (count, task, item["order"], item["wording"])
        lengths[cell] += item["length"]
        tasks[count].add(task)
        cells[(count, task)].add(cell)
    probabilities = []
    for item in episodes:
        count, task = item["component_count"], item["task"]
        cell = (count, task, item["order"], item["wording"])
        mass = 1 / len(tasks) / len(tasks[count]) / len(cells[(count, task)])
        probabilities.extend([mass / lengths[cell]] * item["length"])
    weights = np.asarray(probabilities, dtype=np.float64)
    weights /= weights.sum()
    return weights, {
        "mode": "balanced_component_task_order_wording",
        "component_counts": sorted(tasks),
        "cells": len(lengths),
        "within_cell": "uniform_transitions",
    }


def _prepare(source: Path, output: Path, *, split: str, horizon: int, stats=None) -> PreparedData:
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    source, output = Path(source).resolve(), Path(output).resolve()
    if (source / "index.json").is_file():
        root = source
    elif (source / "episodes").is_dir() and (source / "run.json").is_file():
        record = {
            "version": 1,
            "split": split,
            "prediction_horizon": horizon,
            "source_fingerprint": _raw_fingerprint(source, split),
            "normalization": stats,
        }
        if output.exists():
            preparation = output / "preparation.json"
            if not preparation.is_file() or json.loads(preparation.read_text()) != record:
                raise ValueError("Prepared export differs from source/settings; use a new output")
        else:
            # An interruption must not leave a partially prepared cache at output.
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as scratch:
                staged = Path(scratch) / "export"
                export_hdf5(source, staged, split=split, stats=stats, horizon=horizon)
                write_json(staged / "preparation.json", record)
                staged.rename(output)
        root = output
    else:
        raise ValueError(
            "Expected an HDF5 export (index.json) or collection run (episodes/run.json)"
        )
    metadata, fingerprint = validate_export(root, split)
    if stats is not None and json.loads((root / "normalization.json").read_text()) != stats:
        raise ValueError("Validation must use exactly the training normalization statistics")
    weights, sampling = sampling_weights(metadata)
    return PreparedData(root, ChunkDataset(root, horizon), fingerprint, weights, sampling)


def prepare_training_data(source: Path, output: Path, *, horizon: int = 16) -> PreparedData:
    """Use a train export or export accepted raw training episodes into output."""
    return _prepare(source, output, split="train", horizon=horizon)


def _assert_disjoint(training: PreparedData, validation: PreparedData) -> None:
    train = training.dataset.metadata["episodes"]
    dev = validation.dataset.metadata["episodes"]
    for key in ("source_episode", "instance_id"):
        train_ids = {episode[key] for episode in train if key in episode}
        if train_ids.intersection(episode[key] for episode in dev if key in episode):
            raise ValueError(f"Training/validation episode overlap: {key}")
    # Missing seeds cannot prove a disjoint held-out set. Re-export legacy files
    # from the raw collection to add the provenance; they remain valid for train.
    required = ("task", "values_seed", "layout_seed")
    if any(any(key not in episode for key in required) for episode in train + dev):
        raise ValueError("Cannot verify split seeds in legacy exports; re-export raw runs")
    for key in ("values_seed", "layout_seed"):
        seeds = {(episode["task"], episode[key]) for episode in train}
        if seeds.intersection((episode["task"], episode[key]) for episode in dev):
            raise ValueError(f"Training/validation seed overlap: {key}")


def prepare_validation_data(
    source: Path, output: Path, *, training: PreparedData, horizon: int | None = None
) -> PreparedData:
    """Only dev data may select checkpoints; never fit statistics on dev or eval."""
    result = _prepare(
        source,
        output,
        split="dev",
        horizon=horizon or training.dataset.horizon,
        stats=training.dataset.normalizer.stats,
    )
    _assert_disjoint(training, result)
    return result
