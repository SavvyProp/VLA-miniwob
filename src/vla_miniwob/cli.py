"""Command-line entry points. All provider runs have explicit episode limits."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .collection import run, write_json
from .contracts import MouseAction
from .dataset import export_hdf5
from .environment import BrowserEnvironment
from .policies import OpenAITeacher, RandomPolicy, TeacherConfig, load_policy
from .reporting import report
from .tasks import FAMILIES, make_manifest, read_manifest

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Order-controlled MiniWoB++ benchmark and VLA demonstration pipeline.",
)


@app.command()
def tasks():
    """List the supported mouse-only pilot task families."""
    for name, parts in FAMILIES.items():
        typer.echo(f"{len(parts)}\t{name}")


@app.command()
def manifest(
    output: Path,
    train: Annotated[int, typer.Option(min=0)] = 10,
    dev: Annotated[int, typer.Option(min=0)] = 2,
    evaluation: Annotated[int, typer.Option(min=0)] = 2,
    seed: int = 2026,
    task: Annotated[list[str] | None, typer.Option()] = None,
    max_steps: Annotated[int, typer.Option(min=1)] = 64,
):
    """Freeze paired values/layouts, semantic orders, wording, and disjoint splits."""
    if output.exists():
        raise typer.BadParameter("Manifest exists; choose a new filename")
    value = make_manifest(train, dev, evaluation, seed, task)
    value.max_steps = max_steps
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value.model_dump())
    typer.echo(f"Wrote {len(value.instances)} instances; sha256={value.digest}")


def _teacher(model: str, reasoning: str):
    if not os.environ.get("OPENAI_API_KEY"):
        raise typer.BadParameter("Set OPENAI_API_KEY in your shell to use GPT-6 Astra")
    if reasoning not in ("low", "medium", "high", "xhigh", "max"):
        raise typer.BadParameter("Reasoning must be low, medium, high, xhigh, or max")
    return OpenAITeacher(TeacherConfig(model=model, reasoning_effort=reasoning))


@app.command()
def collect(
    manifest_path: Path,
    output: Path,
    max_episodes: Annotated[int, typer.Option(min=1)] = 10,
    attempts: Annotated[int, typer.Option(min=1)] = 2,
    model: str = "gpt-6-astra",
    reasoning: str = "medium",
    headless: bool = True,
):
    """Collect training demonstrations with Astra; keep all rejected attempts too.

    max-episodes limits NEW attempts this invocation, including failures. Resume
    with the same output directory/configuration. Calls incur OpenAI API usage.
    """
    value = read_manifest(manifest_path)
    policy = _teacher(model, reasoning)
    result = run(
        value,
        output,
        policy,
        mode="collect",
        split="train",
        max_episodes=max_episodes,
        attempts=attempts,
        headless=headless,
    )
    write_json(output / "report.json", report(output))
    typer.echo(json.dumps(result))


@app.command()
def evaluate(
    manifest_path: Path,
    output: Path,
    policy: str = "random",
    split: str = "eval",
    max_episodes: Annotated[int, typer.Option(min=1)] = 10,
    policy_config: Path | None = None,
    model: str = "gpt-6-astra",
    reasoning: str = "medium",
    headless: bool = True,
):
    """Evaluate random, astra, or a module:factory policy; count every model attempt."""
    value = read_manifest(manifest_path)
    if policy == "astra":
        agent = _teacher(model, reasoning)
    elif policy == "random":
        agent = RandomPolicy(value.prediction_horizon)
    else:
        config = json.loads(policy_config.read_text()) if policy_config else {}
        agent = load_policy(policy, config)
    result = run(
        value,
        output,
        agent,
        mode="evaluate",
        split=split,
        max_episodes=max_episodes,
        attempts=1,
        headless=headless,
    )
    write_json(output / "report.json", report(output))
    typer.echo(json.dumps(result))


@app.command("report")
def report_command(run_directory: Path):
    """Report coverage, success, chronology, and clustered uncertainty."""
    value = report(run_directory)
    write_json(run_directory / "report.json", value)
    typer.echo(json.dumps(value, indent=2))


@app.command()
def export(
    run_directory: Path,
    output: Path,
    split: str = "train",
    normalization: Path | None = None,
    horizon: Annotated[int, typer.Option(min=1)] = 16,
):
    """Export accepted trajectories only, with numerical actions and literal history."""
    stats = json.loads(normalization.read_text()) if normalization else None
    result = export_hdf5(run_directory, output, split=split, stats=stats, horizon=horizon)
    typer.echo(json.dumps({key: value for key, value in result.items() if key != "episodes"}))


@app.command()
def smoke(output: Path = Path("runs/smoke"), headless: bool = True):
    """Open a native composed task and execute a neutral move without an API call."""
    value = make_manifest(1, 0, 0, tasks=["click-button_click-link"])
    output.mkdir(parents=True, exist_ok=True)
    with BrowserEnvironment(value, headless=headless) as env:
        observation = env.reset(value.instances[0])
        observation.image.save(output / "initial.png")
        next_observation, action, _ = env.step(MouseAction(x=0.99, y=0.99, click=0))
        next_observation.image.save(output / "after.png")
        write_json(
            output / "smoke.json",
            {
                "instruction": observation.instruction,
                "browser": env.browser_version,
                "executed": action.model_dump(),
            },
        )
    typer.echo(f"Browser smoke check passed; screenshots in {output}")


@app.command()
def replay(episode_directory: Path, manifest_path: Path, output: Path):
    """Replay saved numerical primitives against the same instance and revalidate."""
    from .dataset import read_rows
    from .tasks import Instance

    if output.exists():
        raise typer.BadParameter("Replay output already exists")
    metadata = json.loads((episode_directory / "attempt.json").read_text())
    instance = Instance.model_validate(metadata["instance"])
    value = read_manifest(manifest_path)
    if instance not in value.instances:
        raise typer.BadParameter("Replay instance does not belong to this manifest")
    output.mkdir(parents=True)
    with BrowserEnvironment(value) as env:
        observation = env.reset(instance)
        observation.image.save(output / "000000.png")
        for index, row in enumerate(read_rows(episode_directory)):
            observation, _, _ = env.step(MouseAction.model_validate(row["executed"]["action"]))
            observation.image.save(output / f"{index + 1:06d}.png")
            if env.terminated:
                break
        result = env.private_result()
        write_json(output / "private.json", result)
    typer.echo(json.dumps(result["verdict"]))


if __name__ == "__main__":
    app()
