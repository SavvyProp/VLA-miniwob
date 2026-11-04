"""Generate scriptWoB tasks, collect experts, and export supervised examples."""

import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .collection import ResumeConfigurationError, run, write_json
from .export import export_dataset
from .generation import FAMILIES, generate_manifest, read_manifest, task_fingerprint
from .policies import OpenAITeacher, OraclePolicy, TeacherConfig, load_policy

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
LEVELS = {
    0: "Atomic operations",
    1: "Ordered sequences",
    2: "Conditional branches",
    3: "Repetition and stopping conditions",
    4: "Branches composed with loops",
    5: "Nested control flow",
}


def _print(value):
    typer.echo(json.dumps(value, indent=2, allow_nan=False))


def _run_rollouts(manifest, output: Path, policy, *, fresh: bool, **kwargs):
    if fresh:
        # Reserve a unique sibling atomically, without reading or modifying the
        # old run. mkdtemp adds a random suffix even for concurrent invocations.
        output.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(tempfile.mkdtemp(prefix=f"{output.name}-{timestamp}-", dir=output.parent))
        typer.echo(f"Fresh run directory: {output.resolve()}", err=True)
    try:
        return run(manifest, output, policy, **kwargs)
    except ResumeConfigurationError as exc:
        typer.echo(f"Error: {exc} Use --fresh to start a separate run.", err=True)
        raise typer.Exit(1) from None


def _policy(teacher: str, model: str, reasoning: str, max_output_tokens: int):
    if teacher == "oracle":
        return OraclePolicy()
    if teacher != "openai":
        raise typer.BadParameter("teacher must be openai or oracle")
    if not os.environ.get("OPENAI_API_KEY"):
        raise typer.BadParameter(
            "Set OPENAI_API_KEY in the environment (or use uv run --env-file .env)."
        )
    return OpenAITeacher(
        TeacherConfig(model=model, reasoning_effort=reasoning, max_output_tokens=max_output_tokens)
    )


@app.command()
def catalog():
    """List all available script families, grouped by structural difficulty."""
    _print(
        {
            str(level): {"description": LEVELS[level], "families": families}
            for level, families in FAMILIES.items()
        }
    )


@app.command()
def generate(
    output: Path,
    train: Annotated[int, typer.Option(min=0)] = 600,
    dev: Annotated[int, typer.Option(min=0)] = 120,
    test: Annotated[int, typer.Option(min=0)] = 120,
    seed: int = 2026,
    levels: str | None = None,
    split_strategy: str = "seed",
):
    """Create reproducible tasks. Counts are totals per split, balanced across levels."""
    if output.exists():
        raise typer.BadParameter("Output already exists; choose a new manifest path")
    try:
        selected = (
            [int(s.strip()) for s in levels.split(",")]
            if levels
            else (list(range(2, 6)) if split_strategy == "structure" else list(range(6)))
        )
        manifest = generate_manifest(
            train=train,
            dev=dev,
            test=test,
            seed=seed,
            levels=selected,
            split_strategy=split_strategy,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest.model_dump(mode="json"))
    _print(
        {
            "manifest": str(output),
            "digest": manifest.digest,
            "episodes": len(manifest.episodes),
            "split_strategy": split_strategy,
        }
    )


@app.command()
def inspect(manifest_path: Path, index: int | None = None):
    """Describe task difficulty and splits, or print one episode's instruction."""
    manifest = read_manifest(manifest_path)
    if index is not None:
        if not 0 <= index < len(manifest.episodes):
            raise typer.BadParameter("Episode index is out of range")
        episode = manifest.episodes[index]
        _print(
            {
                "id": episode.id,
                "level": episode.level,
                "family": episode.family,
                "split": episode.split,
                "instruction": episode.instruction,
                "complexity": episode.model_dump(mode="json")["complexity"],
            }
        )
        return
    _print(
        {
            "version": __version__,
            "digest": manifest.digest,
            "episodes": len(manifest.episodes),
            "splits": dict(Counter(e.split for e in manifest.episodes)),
            "unique_tasks_by_split": {
                split: len({task_fingerprint(e) for e in manifest.episodes if e.split == split})
                for split in ("train", "dev", "test")
            },
            "levels": {
                str(k): {"description": v, "episodes": sum(e.level == k for e in manifest.episodes)}
                for k, v in LEVELS.items()
            },
            "families": dict(Counter(e.family for e in manifest.episodes)),
        }
    )


@app.command()
def validate(manifest_path: Path):
    """Validate every program with its reference executor, without launching a browser."""
    from .engine import Engine

    manifest = read_manifest(manifest_path)
    actions = 0
    for episode in manifest.episodes:
        engine = Engine(episode)
        steps = 0
        while not engine.done and steps < episode.max_steps:
            action = engine.expected_action()
            if action is None:
                raise ValueError(f"No reference action for {episode.id}")
            engine.step(action)
            steps += 1
        if not engine.success:
            raise ValueError(f"Invalid reference execution: {episode.id}: {engine.error}")
        actions += steps
    _print({"validated_episodes": len(manifest.episodes), "reference_clicks": actions})


@app.command()
def preview(manifest_path: Path, output: Path, index: int = 0, headed: bool = False):
    """Render one task as a PNG plus its public natural-language instruction."""
    from .environment import BrowserEnvironment

    manifest = read_manifest(manifest_path)
    if not 0 <= index < len(manifest.episodes):
        raise typer.BadParameter("Episode index is out of range")
    if output.exists():
        raise typer.BadParameter("Preview directory already exists")
    with BrowserEnvironment(headless=not headed) as env:
        episode = manifest.episodes[index]
        observation = env.reset(episode)
        output.mkdir(parents=True)
        observation.image.save(output / "screen.png")
        (output / "instruction.txt").write_text(observation.instruction + "\n")
        write_json(
            output / "task.json",
            {
                "id": episode.id,
                "level": episode.level,
                "family": episode.family,
                "instruction": observation.instruction,
            },
        )
    _print({"preview": str(output.resolve())})


@app.command()
def collect(
    manifest_path: Path,
    output: Path,
    teacher: str = "openai",
    model: str = "gpt-6-astra",
    reasoning: str = "medium",
    attempts: Annotated[int, typer.Option(min=1)] = 3,
    max_episodes: Annotated[int | None, typer.Option(min=1)] = None,
    max_total_steps: Annotated[int, typer.Option(min=1)] = 10000,
    max_output_tokens: Annotated[int, typer.Option(min=1)] = 4096,
    headed: bool = False,
    fresh: Annotated[bool, typer.Option(
        help="Start from the beginning in a new, uniquely named sibling output directory."
    )] = False,
):
    """Collect successful train demonstrations. Use --fresh for a separate new run.

    Without --fresh, re-running resumes and limits may increase.
    max-episodes selects a prefix of train instances. max-total-steps bounds NEW
    model calls this invocation. A budget-cut attempt is retained as unsuccessful.
    Oracle is privileged reference supervision, never frontier-model supervision.
    """
    manifest = read_manifest(manifest_path)
    result = _run_rollouts(
        manifest,
        output,
        _policy(teacher, model, reasoning, max_output_tokens),
        fresh=fresh,
        mode="collect",
        split="train",
        attempts=attempts,
        max_episodes=max_episodes,
        max_total_steps=max_total_steps,
        headless=not headed,
    )
    _print(result)
    if result["stop_reason"] in {"provider_error", "infrastructure_error", "runner_error"}:
        raise typer.Exit(1)


@app.command()
def evaluate(
    manifest_path: Path,
    output: Path,
    split: str = "test",
    teacher: str = "openai",
    model: str = "gpt-6-astra",
    reasoning: str = "medium",
    policy_factory: str | None = None,
    policy_config: Path | None = None,
    max_episodes: Annotated[int | None, typer.Option(min=1)] = None,
    max_total_steps: Annotated[int, typer.Option(min=1)] = 10000,
    headed: bool = False,
    fresh: Annotated[bool, typer.Option(
        help="Start from the beginning in a new, uniquely named sibling output directory."
    )] = False,
):
    """Evaluate dev/test once per episode with a frontier expert or local policy plugin."""
    if split not in {"dev", "test"}:
        raise typer.BadParameter("Evaluation split must be dev or test")
    manifest = read_manifest(manifest_path)
    if policy_config and not policy_factory:
        raise typer.BadParameter("policy-config requires policy-factory")
    policy = (
        load_policy(policy_factory, json.loads(policy_config.read_text()) if policy_config else {})
        if policy_factory
        else _policy(teacher, model, reasoning, 4096)
    )
    result = _run_rollouts(
        manifest,
        output,
        policy,
        fresh=fresh,
        mode="evaluate",
        split=split,
        attempts=1,
        max_episodes=max_episodes,
        max_total_steps=max_total_steps,
        headless=not headed,
    )
    _print(result)
    if result["stop_reason"] in {"provider_error", "infrastructure_error", "runner_error"}:
        raise typer.Exit(1)


@app.command("export")
def export_command(
    run_dir: Path, output: Path, format: str = "messages", allow_oracle: bool = False
):
    """Export accepted train examples; formats: messages JSONL or upstream seeclick JSON."""
    _print(export_dataset(run_dir, output, format=format, allow_oracle=allow_oracle))


@app.command("train-seeclick")
def train_seeclick_command(
    source: Path,
    output: Path,
    config: Path | None = None,
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    resume: bool = False,
    prepare_only: bool = False,
    allow_oracle: bool = False,
):
    """Fine-tune pretrained SeeClick from a collection run or SeeClick export.

    Run in integrations/seeclick's uv runtime. prepare-only validates images and
    token lengths without loading weights. Resume restores optimizer/RNG state.
    """
    from .seeclick_training import SeeClickTrainingConfig, train_seeclick

    values = json.loads(config.read_text()) if config else {}
    if max_steps is not None:
        values["max_steps"] = max_steps
    settings = SeeClickTrainingConfig.model_validate(values)
    _print(
        train_seeclick(
            source,
            output,
            settings,
            resume=resume,
            prepare_only=prepare_only,
            allow_oracle=allow_oracle,
        )
    )


@app.command("evaluate-seeclick")
def evaluate_seeclick_command(
    checkpoint: str,
    manifest_path: Path,
    output: Path,
    split: str = "test",
    max_episodes: Annotated[int | None, typer.Option(min=1)] = None,
    max_total_steps: Annotated[int, typer.Option(min=1)] = 10000,
    device: str | None = None,
    quantization: str | None = None,
    precision: str | None = None,
    max_context_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 64,
    allow_different_manifest: bool = False,
    headed: bool = False,
    fresh: Annotated[bool, typer.Option(
        help="Start from the beginning in a new, uniquely named sibling output directory."
    )] = False,
):
    """Evaluate a trained SeeClick run/checkpoint, or 'base' for the pretrained baseline."""
    from .seeclick_policy import SeeClickPolicy
    from .seeclick_runtime import resolve_checkpoint

    if split not in {"dev", "test"}:
        raise typer.BadParameter("Evaluation split must be dev or test")
    manifest = read_manifest(manifest_path)
    _, _, metadata = resolve_checkpoint(checkpoint)
    training_manifest = metadata.get("dataset", {}).get("metadata", {}).get("manifest_digest")
    if training_manifest and training_manifest != manifest.digest and not allow_different_manifest:
        raise typer.BadParameter(
            "Evaluation manifest differs from the training manifest. Use the original manifest, "
            "or --allow-different-manifest for a deliberate transfer evaluation after checking "
            "that its dev/test tasks were not used for training."
        )
    policy = SeeClickPolicy(
        checkpoint,
        device=device,
        quantization=quantization,
        precision=precision,
        max_context_tokens=max_context_tokens,
        max_new_tokens=max_new_tokens,
    )
    try:
        result = _run_rollouts(
            manifest,
            output,
            policy,
            fresh=fresh,
            mode="evaluate",
            split=split,
            attempts=1,
            max_episodes=max_episodes,
            max_total_steps=max_total_steps,
            headless=not headed,
        )
    finally:
        policy.close()
    _print(result)
    if result["stop_reason"] in {"provider_error", "infrastructure_error", "runner_error"}:
        raise typer.Exit(1)


@app.command()
def report(run_dir: Path):
    """Print the latest collection/evaluation report."""
    _print(json.loads((run_dir / "report.json").read_text()))


@app.command()
def doctor():
    """Check local prerequisites without making an API request or starting Chrome."""
    from importlib.metadata import version

    _print(
        {
            "scriptwob": __version__,
            "chrome": os.environ.get("MINIWOB_CHROME_BINARY")
            or shutil.which("google-chrome")
            or shutil.which("chromium"),
            "driver": os.environ.get("MINIWOB_CHROMEDRIVER")
            or shutil.which("chromedriver")
            or "Selenium Manager will locate/download a matching driver",
            "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "packages": {name: version(name) for name in ("miniwob", "selenium", "openai")},
        }
    )


if __name__ == "__main__":
    app()
