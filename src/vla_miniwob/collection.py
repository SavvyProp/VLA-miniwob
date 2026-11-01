"""Bounded, resumable collection/evaluation; public transitions and private verdicts."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from openai import OpenAIError
from pydantic import ValidationError
from selenium.common.exceptions import WebDriverException

from .contracts import CONTRACT_VERSION, MouseAction, Policy
from .environment import BrowserEnvironment
from .policies import PolicyOutputError
from .tasks import Instance, Manifest, stable_seed


def write_json(path: Path, value) -> None:
    """Atomic metadata update, so interruption cannot create a valid-looking result."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def attempt_path(root: Path, instance: Instance, attempt: int) -> Path:
    key = hashlib.sha256(instance.id.encode()).hexdigest()[:24]
    return root / "episodes" / f"{key}-{attempt:03d}"


def implementation_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*")):
        if path.suffix in (".py", ".html"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_episode(
    env: BrowserEnvironment,
    policy: Policy,
    instance: Instance,
    directory: Path,
    attempt: int,
    rollout_seed: int,
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    write_json(
        directory / "attempt.json",
        {
            "instance": instance.model_dump(),
            "attempt": attempt,
            "rollout_seed": rollout_seed,
            "started_utc": started,
        },
    )
    reason, error, infrastructure = "", None, False
    rows = []
    reset_ok = False
    expected_horizon = policy.provenance().get(
        "prediction_horizon", env.manifest.prediction_horizon
    )
    try:
        policy.reset(rollout_seed)
        observation = env.reset(instance)
        reset_ok = True
        observation.image.save(directory / "000000.png")
        write_json(
            directory / "episode.json",
            {
                "schema_version": 1,
                "action_contract": CONTRACT_VERSION,
                "instruction": observation.instruction,
                "image_size": observation.image.size,
                "initial_state": observation.state,
            },
        )
        with (directory / "transitions.jsonl").open("w") as stream:
            while not env.terminated and env.steps < env.manifest.max_steps:
                prediction = policy.predict(observation)
                try:
                    chunk = np.asarray(prediction, dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise PolicyOutputError("Action chunk must be a numerical H x 3 array") from exc
                if (
                    chunk.ndim != 2
                    or chunk.shape[1] != 3
                    or not np.isfinite(chunk).all()
                    or len(chunk) != expected_horizon
                    or len(chunk) < env.manifest.execution_prefix
                ):
                    raise PolicyOutputError(f"Invalid H x 3 action chunk: shape={chunk.shape}")
                # One chunk, one prefix: no validator-based resampling or temporal averaging.
                for vector in chunk[: env.manifest.execution_prefix]:
                    before = observation
                    action = MouseAction(x=vector[0], y=vector[1], click=vector[2])
                    observation, executed, feedback = env.step(action)
                    t = len(rows)
                    observation.image.save(directory / f"{t + 1:06d}.png")
                    row = {
                        "step": t,
                        "image": f"{t:06d}.png",
                        "state": before.state,
                        "requested_action": action.model_dump(),
                        "executed": executed.model_dump(),
                        "feedback": feedback,
                        "next_image": f"{t + 1:06d}.png",
                        "next_state": observation.state,
                        "terminal": env.terminated,
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if env.terminated or env.steps >= env.manifest.max_steps:
                        break
    except (PolicyOutputError, ValidationError) as exc:
        reason, error = "invalid_policy_output", str(exc)
    except OpenAIError as exc:
        reason, error, infrastructure = "provider_error", type(exc).__name__, True
    except (WebDriverException, OSError, TimeoutError) as exc:
        reason, error, infrastructure = "infrastructure_error", str(exc), True
    except KeyboardInterrupt:
        reason, infrastructure = "interrupted", True
        raise
    except Exception as exc:
        # Programming errors must not masquerade as benchmark model failures.
        reason, error, infrastructure = "runner_error", f"{type(exc).__name__}: {exc}", True
        raise
    finally:
        try:
            result = env.private_result(reason) if reset_ok else None
        except (WebDriverException, OSError):
            reason, infrastructure, result = "infrastructure_error", True, None
        if result is None:
            result = {
                "required_order": instance.required_order,
                "events": [],
                "native_reward": None,
                "verdict": {
                    "success": False,
                    "goals_valid": False,
                    "order_valid": False,
                    "budget_valid": False,
                    "completed": [],
                    "first_violation": None,
                    "reason": reason or "interrupted",
                },
            }
        result.update(
            instance=instance.model_dump(),
            attempt=attempt,
            rollout_seed=rollout_seed,
            started_utc=started,
            finished_utc=datetime.now(UTC).isoformat(),
            primitive_transitions=len(rows),
            infrastructure_failure=infrastructure,
            error=error,
            browser_version=env.browser_version,
            teacher_calls=getattr(policy, "calls", []),
        )
        write_json(directory / "private.json", result)
    return result


def run(
    manifest: Manifest,
    output: Path,
    policy: Policy,
    *,
    mode: str,
    split: str,
    max_episodes: int,
    attempts: int = 2,
    headless: bool = True,
    env_factory=BrowserEnvironment,
) -> dict:
    if mode not in ("collect", "evaluate") or max_episodes < 1 or attempts < 1:
        raise ValueError("Invalid run limits or mode")
    if mode == "collect" and split != "train":
        raise ValueError("Demonstration collection is restricted to training instances")
    if mode == "evaluate" and split not in ("dev", "eval"):
        raise ValueError("Reference/student evaluation requires dev or eval instances")
    if mode == "evaluate" and attempts != 1:
        raise ValueError("Initial evaluation protocol requires exactly one attempt per instance")
    selected = [item for item in manifest.instances if item.split == split]
    if not selected:
        raise ValueError(f"No {split} instances in manifest")
    provenance = policy.provenance()
    horizon = provenance.get("prediction_horizon", manifest.prediction_horizon)
    if not manifest.execution_prefix <= horizon <= manifest.prediction_horizon:
        raise ValueError("Declared policy horizon must cover K and fit the manifest horizon")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "implementation_digest": implementation_digest(),
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("miniwob", "selenium", "openai", "numpy", "pillow")
            },
        },
        "mode": mode,
        "split": split,
        "attempts_per_instance": attempts,
        "policy": provenance,
        "action_contract": CONTRACT_VERSION,
        "prediction_horizon": horizon,
        "training_chunk_horizon": manifest.prediction_horizon,
        "execution_prefix": manifest.execution_prefix,
        "infrastructure_policy": "retain and exclude from success denominator; report separately",
    }
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != config:
            raise ValueError("Resume configuration differs; use a new output directory")
    else:
        write_json(run_path, config)
        write_json(output / "manifest.json", manifest.model_dump())
    count = 0
    with env_factory(manifest, headless=headless) as env:
        for instance in selected:
            for attempt in range(attempts):
                directory = attempt_path(output, instance, attempt)
                result_file = directory / "private.json"
                if result_file.exists():
                    previous = json.loads(result_file.read_text())
                    if previous["verdict"]["success"]:
                        break
                    continue
                if directory.exists():
                    # Never silently overwrite interrupted collection or erase its cost.
                    transition_file = directory / "transitions.jsonl"
                    saved_steps = (
                        len(transition_file.read_text().splitlines())
                        if transition_file.exists()
                        else 0
                    )
                    write_json(
                        result_file,
                        {
                            "instance": instance.model_dump(),
                            "attempt": attempt,
                            "primitive_transitions": saved_steps,
                            "infrastructure_failure": True,
                            "verdict": {
                                "success": False,
                                "goals_valid": False,
                                "order_valid": False,
                                "reason": "interrupted",
                            },
                        },
                    )
                    continue
                if count >= max_episodes:
                    return {"new_attempts": count}
                result = run_episode(
                    env,
                    policy,
                    instance,
                    directory,
                    attempt,
                    stable_seed(instance.id, attempt, "rollout"),
                )
                count += 1
                if result["verdict"]["success"]:
                    break
    return {"new_attempts": count}
