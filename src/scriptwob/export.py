"""Export validated successful expert trajectories without private evaluator information."""

import hashlib
import json
import shutil
from pathlib import Path

from .contracts import ACTION_VERSION, ClickAction, HistoryEntry, format_context
from .generation import read_manifest
from .seeclick_prompt import PROMPT_VERSION, build_assistant_answer, build_user_prompt


def _json(path: Path):
    return json.loads(path.read_text())


def _contained(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe trajectory image: {name}")
    return path


def _verify_image(directory: Path, row: dict, key: str) -> Path:
    path = _contained(directory, row[key])
    expected = row.get(f"{key}_sha256")
    if expected is None:
        raise ValueError(f"Missing integrity digest for {key}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Image integrity check failed: {path}")
    return path


def export_dataset(
    run_dir: Path,
    output: Path,
    *,
    format: str = "messages",
    allow_oracle: bool = False,
) -> dict:
    """Copy training images and one next-action example per accepted expert step.

    `messages` is portable JSONL with image paths relative to the dataset directory.
    `seeclick` is the upstream Qwen-VL conversation JSON format with absolute image paths.
    Never export validation/test trajectories, unsuccessful attempts or mixed teachers.
    """
    if format not in {"messages", "seeclick"}:
        raise ValueError("format must be messages or seeclick")
    run_dir, output = Path(run_dir), Path(output)
    run = _json(run_dir / "run.json")
    # Collection fingerprints retain public provenance. No AST is copied into samples.
    config = run.get("config", run)
    if config.get("mode") != "collect" or config.get("split") != "train":
        raise ValueError("Only a train collection run can become a fine-tuning dataset")
    policy = config.get("policy", {})
    supervision = policy.get("supervision")
    if supervision != "frontier" and not (allow_oracle and supervision == "privileged_oracle"):
        raise ValueError("Export requires frontier demonstrations; oracle needs --allow-oracle")
    manifest = read_manifest(run_dir / "manifest.json")
    if manifest.digest != config.get("manifest_digest"):
        raise ValueError("Manifest integrity check failed")
    if config.get("action_contract") != ACTION_VERSION:
        raise ValueError("Unsupported action contract")
    source_episodes = {episode.id: episode for episode in manifest.episodes}
    if output.exists():
        raise ValueError("Export directory already exists; choose a new output directory")
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()
    samples, episodes, seen = [], [], set()
    image_hashes = {}
    try:
        for result_path in sorted((run_dir / "attempts").glob("*/private.json")):
            if result_path.parent.name.startswith("."):
                continue  # Only atomic directory commits publish an accepted attempt.
            result = _json(result_path)
            if not result.get("accepted"):
                continue
            if not result.get("success") or result.get("infrastructure_failure"):
                raise ValueError("Accepted attempt does not have a successful verdict")
            directory = result_path.parent
            episode = _json(directory / "episode.json")
            if episode["split"] != "train":
                raise ValueError("Non-training episode in collection")
            episode_id = episode["episode_id"]
            source = source_episodes.get(episode_id)
            if source is None or source.split != "train":
                raise ValueError("Accepted episode is not in the training manifest")
            if (
                episode["instruction"] != source.instruction
                or episode["level"] != source.level
                or episode["family"] != source.family
            ):
                raise ValueError("Episode metadata differs from the frozen manifest")
            if (
                episode.get("supervision") != supervision
                or result.get("supervision") != supervision
            ):
                raise ValueError("Episode supervision differs from the run policy")
            if (
                episode.get("action_contract") != ACTION_VERSION
                or result.get("episode_id") != episode_id
            ):
                raise ValueError("Episode identity or action contract is inconsistent")
            if episode_id in seen:
                continue
            rows = [
                json.loads(line)
                for line in (directory / "transitions.jsonl").read_text().splitlines()
            ]
            if not rows or not rows[-1]["terminal"]:
                raise ValueError("Accepted episode lacks a terminal transition")
            history: list[HistoryEntry] = []
            image_prefix = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
            previous_next = None
            for index, row in enumerate(rows):
                if row["step"] != index + 1:
                    raise ValueError("Noncontiguous transition sequence")
                if index < len(rows) - 1 and row["terminal"]:
                    raise ValueError("Actions appear after a terminal transition")
                if row.get("instruction", episode["instruction"]) != episode["instruction"]:
                    raise ValueError("Instruction changed during the trajectory")
                image = _verify_image(directory, row, "image")
                _verify_image(directory, row, "next_image")
                if previous_next is not None and previous_next != row["image_sha256"]:
                    raise ValueError("Discontinuous screenshot history")
                previous_next = row["next_image_sha256"]
                action = ClickAction.model_validate(row["action"])
                relative = f"images/{image_prefix}-{index:05d}.png"
                shutil.copyfile(image, output / relative)
                image_hashes[relative] = row["image_sha256"]
                context = format_context(episode["instruction"], tuple(history))
                if format == "messages":
                    samples.append(
                        {
                            "id": f"{episode_id}/{index}",
                            "episode_id": episode_id,
                            "step": index,
                            "split": "train",
                            "level": episode["level"],
                            "images": [relative],
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "<image>\n"
                                    + context
                                    + "\nReturn one click as JSON with x and y in [0,1].",
                                },
                                {"role": "assistant", "content": action.model_dump_json()},
                            ],
                        }
                    )
                else:
                    samples.append(
                        {
                            "id": f"{episode_id}/{index}",
                            "conversations": [
                                {
                                    "from": "user",
                                    "value": build_user_prompt(
                                        episode["instruction"], tuple(history), output / relative
                                    ),
                                },
                                {"from": "assistant", "value": build_assistant_answer(action)},
                            ],
                        }
                    )
                history.append(
                    HistoryEntry(step=index + 1, action=action, feedback=row["feedback"])
                )
            seen.add(episode_id)
            episodes.append({"id": episode_id, "level": episode["level"], "steps": len(rows)})
        if not samples:
            raise ValueError("No accepted training demonstrations to export")
        filename = "train.jsonl" if format == "messages" else "train.json"
        if format == "messages":
            (output / filename).write_text("".join(json.dumps(s) + "\n" for s in samples))
        else:
            (output / filename).write_text(json.dumps(samples, indent=2) + "\n")
        summary = {
            "schema_version": "scriptwob-sft-v1",
            "format": format,
            "split": "train",
            "prompt_version": PROMPT_VERSION if format == "seeclick" else "scriptwob-messages-v1",
            "image_sha256": image_hashes,
            "supervision": supervision,
            "source_run_sha256": hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest(),
            "manifest_digest": manifest.digest,
            "policy": policy,
            "episodes": len(episodes),
            "examples": len(samples),
            "data_file": filename,
            "image_paths": "relative to dataset"
            if format == "messages"
            else "absolute; re-export after moving",
        }
        (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
        return summary
    except BaseException:
        # This directory was created exclusively by this invocation.
        shutil.rmtree(output)
        raise
