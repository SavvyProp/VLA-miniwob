"""Leakage-safe, episode-local action chunks and TinyVLA-shaped HDF5 export."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .collection import write_json
from .contracts import HistoryEntry, MouseAction, format_context
from .tasks import FAMILIES


def read_rows(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "transitions.jsonl").read_text().splitlines()]


def accepted_episodes(root: Path, split: str = "train") -> list[Path]:
    directories = []
    for path in sorted((root / "episodes").glob("*/private.json")):
        result = json.loads(path.read_text())
        if (
            result["instance"]["split"] == split
            and result["verdict"]["success"]
            and not result["infrastructure_failure"]
        ):
            directories.append(path.parent)
    return directories


def numerical_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    # Train on the primitive actually executed, including coordinate quantization.
    actions = np.asarray(
        [MouseAction.model_validate(r["executed"]["action"]).vector() for r in rows],
        dtype=np.float32,
    )
    states = np.asarray([r["state"] for r in rows], dtype=np.float32)
    if not rows or actions.shape != states.shape or actions.shape[1] != 3:
        raise ValueError("Episode must contain aligned numerical actions/states")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Non-finite episode")
    return actions, states


def fit_normalization(root: Path) -> dict:
    directories = accepted_episodes(root, "train")
    if not directories:
        raise ValueError("Normalization requires successful training episodes")
    values = [numerical_rows(read_rows(path)) for path in directories]
    actions, states = (np.concatenate([v[i] for v in values]) for i in (0, 1))
    low, high = actions.min(0), actions.max(0)
    constant = (high - low) < 1e-6
    low, high = np.maximum(0, low - 1e-4), np.minimum(1, high + 1e-4)
    low[constant], high[constant] = 0, 1
    return {
        "version": "gui-train-minmax-v1",
        "fit_split": "train",
        "source_run_sha256": hashlib.sha256((root / "run.json").read_bytes()).hexdigest(),
        "source_episodes": [path.name for path in directories],
        "transitions": len(actions),
        "action_min": low.tolist(),
        "action_max": high.tolist(),
        "state_mean": states.mean(0).tolist(),
        "state_std": np.maximum(states.std(0), 0.01).tolist(),
        "constant_channel_rule": "use physical [0,1] bounds",
    }


class Normalizer:
    def __init__(self, stats: dict):
        if stats.get("version") != "gui-train-minmax-v1" or stats.get("fit_split") != "train":
            raise ValueError("Unsupported normalization or non-training fit")
        self.stats = stats
        self.low = np.asarray(stats["action_min"], dtype=np.float32)
        self.high = np.asarray(stats["action_max"], dtype=np.float32)
        self.mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.std = np.asarray(stats["state_std"], dtype=np.float32)
        if any(
            x.shape != (3,) or not np.isfinite(x).all()
            for x in (self.low, self.high, self.mean, self.std)
        ):
            raise ValueError("Invalid three-channel normalization statistics")
        if np.any(self.high <= self.low) or np.any(self.std <= 0):
            raise ValueError("Normalization scales must be positive")

    def actions(self, actions):
        return (2 * (actions - self.low) / (self.high - self.low) - 1).astype(np.float32)

    def undo_actions(self, actions):
        return ((np.asarray(actions) + 1) / 2 * (self.high - self.low) + self.low).astype(
            np.float32
        )

    def states(self, states):
        return ((states - self.mean) / self.std).astype(np.float32)


def contexts(instruction: str, rows: list[dict]) -> list[str]:
    history = []
    result = []
    for row in rows:
        result.append(format_context(instruction, tuple(history)))
        history.append(
            HistoryEntry(
                step=row["step"] + 1,
                action=MouseAction.model_validate(row["executed"]["action"]),
                feedback=row["feedback"],
            )
        )
    return result


def export_hdf5(
    root: Path, output: Path, *, split: str = "train", stats: dict | None = None, horizon: int = 16
) -> dict:
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    if split != "train" and stats is None:
        raise ValueError("Dev/eval exports require statistics fitted on the training run")
    directories = accepted_episodes(root, split)
    if not directories:
        raise ValueError(f"No accepted {split} episodes to export")
    stats = stats or fit_normalization(root)
    Normalizer(stats)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "normalization.json", stats)
    index = []
    for number, directory in enumerate(directories):
        episode = json.loads((directory / "episode.json").read_text())
        rows = read_rows(directory)
        actions, states = numerical_rows(rows)
        language = contexts(episode["instruction"], rows)
        name = f"episode_{number:06d}.hdf5"
        with h5py.File(output / name, "w") as f:
            f.attrs.update(
                sim=True,
                compress=False,
                schema_version=1,
                action_contract=episode["action_contract"],
                prediction_horizon=horizon,
                source_episode=directory.name,
            )
            f.create_dataset("action", data=actions)
            f.create_dataset("observations/qpos", data=states)
            # Required by upstream loaders; not a policy input in this benchmark.
            f.create_dataset("observations/qvel", data=np.zeros_like(states))
            width, height = episode["image_size"]
            images = f.create_dataset(
                "observations/images/screen",
                shape=(len(rows), height, width, 3),
                dtype="u1",
                chunks=(1, height, width, 3),
                compression="gzip",
            )
            for t, row in enumerate(rows):
                with Image.open(directory / row["image"]) as image:
                    images[t] = np.asarray(image.convert("RGB"))
            string = h5py.string_dtype("utf-8")
            f.create_dataset("language_raw", data=[episode["instruction"]], dtype=string)
            f.create_dataset("language_per_timestep", data=language, dtype=string)
        item = {"file": name, "length": len(rows), "source_episode": directory.name}
        # Sampling/split provenance lives only in the index, never in policy inputs.
        instance = json.loads((directory / "private.json").read_text())["instance"]
        for key in ("task", "order", "wording", "values_seed", "layout_seed", "order_seed"):
            if key in instance:
                item[key] = instance[key]
        if "id" in instance:
            item["instance_id"] = instance["id"]
        if instance.get("task") in FAMILIES:
            item["component_count"] = len(FAMILIES[instance["task"]])
        index.append(item)
    metadata = {
        "version": 1,
        "split": split,
        "prediction_horizon": horizon,
        "normalization_version": stats["version"],
        "episodes": index,
        "primitive_transitions": sum(item["length"] for item in index),
        "chunk_examples": sum(item["length"] for item in index),
    }
    write_json(output / "index.json", metadata)
    return metadata


class ChunkDataset:
    """NumPy dataset: one current image/context/state and one masked future action chunk.

    Sampling never crosses an episode; terminal actions are valid, positions AFTER
    them are padding. Future screenshots and evaluator records aren't loaded.
    """

    def __init__(self, root: str | Path, horizon: int | None = None):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "index.json").read_text())
        self.horizon = horizon or self.metadata["prediction_horizon"]
        if self.horizon < 1:
            raise ValueError("Horizon must be positive")
        self.normalizer = Normalizer(json.loads((self.root / "normalization.json").read_text()))
        self.index = [
            (episode["file"], t, episode["length"])
            for episode in self.metadata["episodes"]
            for t in range(episode["length"])
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        filename, start, end = self.index[index]
        valid = min(self.horizon, end - start)
        action = np.zeros((self.horizon, 3), dtype=np.float32)
        with h5py.File(self.root / filename, "r") as f:
            action[:valid] = self.normalizer.actions(f["action"][start : start + valid])
            image = f["observations/images/screen"][start]
            state = self.normalizer.states(f["observations/qpos"][start])
            language = f["language_per_timestep"].asstr()[start]
        return {
            "image": image,
            "state": state,
            "action": action,
            "is_pad": np.arange(self.horizon) >= valid,
            "raw_lang": language,
        }


class TinyVLADataset:
    """Torch-facing adapter. A supplied processor receives one-view RGB CHW data.

    Do not use upstream's unmodified two-camera LlavaPythiaProcess: it indexes a
    second view unconditionally. See docs/tinyvla.md for the audited integration.
    """

    def __init__(self, dataset: ChunkDataset, processor=None):
        self.dataset, self.processor = dataset, processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch

        sample = self.dataset[index]
        sample["image"] = torch.from_numpy(sample["image"].copy()).permute(2, 0, 1)[None] / 255.0
        for key in ("state", "action", "is_pad"):
            sample[key] = torch.from_numpy(sample[key])
        return self.processor(sample) if self.processor else sample


def check_context(text: str, tokenizer, max_tokens: int, reserved_image_tokens: int) -> int:
    """Preflight with actual tokenizer; include wrapper tokens in text, never truncate."""
    count = len(tokenizer.encode(text, add_special_tokens=True)) + reserved_image_tokens
    if count > max_tokens:
        raise ValueError(f"Full procedure/history needs {count} tokens, capacity is {max_tokens}")
    return count


def masked_diffusion_loss(prediction, target, is_pad):
    """Epsilon MSE averaged per example over valid positions, then across examples."""
    import torch

    if prediction.shape != target.shape or is_pad.shape != prediction.shape[:2]:
        raise ValueError("Expected B x H x A predictions/targets and B x H padding mask")
    mask = (~is_pad.bool()).unsqueeze(-1)
    counts = mask.sum((1, 2)) * prediction.shape[-1]
    if torch.any(counts == 0):
        raise ValueError("Every diffusion example needs at least one valid action")
    return (((prediction - target).square() * mask).sum((1, 2)) / counts).mean()
