import json

import pytest
from typer.testing import CliRunner

from scriptwob import cli, seeclick_policy, seeclick_runtime, seeclick_training
from scriptwob.collection import ResumeConfigurationError
from scriptwob.generation import generate_manifest, write_manifest

runner = CliRunner()


def test_training_cli_configuration_overrides_and_resume(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_steps": 5, "learning_rate": 2e-4}))
    received = {}

    def train(source, output, settings, **kwargs):
        received.update(source=source, output=output, settings=settings, **kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(seeclick_training, "train_seeclick", train)
    result = runner.invoke(
        cli.app,
        [
            "train-seeclick",
            str(tmp_path / "source"),
            str(tmp_path / "training"),
            "--config",
            str(config),
            "--max-steps",
            "9",
            "--resume",
            "--prepare-only",
            "--allow-oracle",
            "--prompt-style",
            "code",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received["settings"].max_steps == 9
    assert received["settings"].learning_rate == 2e-4
    assert received["resume"] and received["prepare_only"] and received["allow_oracle"]
    assert received["prompt_style"] == "code"


@pytest.fixture
def evaluation_cli(tmp_path, monkeypatch):
    manifest = generate_manifest(train=1, dev=1, test=1, seed=77)
    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, manifest)
    observed = {"constructed": False, "closed": False, "metadata": {}}

    class Policy:
        def __init__(self, checkpoint, **kwargs):
            observed.update(constructed=True, checkpoint=checkpoint, policy_options=kwargs)

        def close(self):
            observed["closed"] = True

    def resolve(*args, **kwargs):
        return None, None, observed["metadata"]

    def run(value, output, policy, **kwargs):
        observed.update(run_options=kwargs, manifest=value, output=output)
        return {"stop_reason": "complete", "successful_episodes": 1}

    monkeypatch.setattr(seeclick_runtime, "resolve_checkpoint", resolve)
    monkeypatch.setattr(seeclick_policy, "SeeClickPolicy", Policy)
    monkeypatch.setattr(cli, "run", run)
    return ["evaluate-seeclick", "base", str(manifest_path), str(tmp_path / "eval")], observed


def test_evaluation_cli_uses_one_attempt_and_closes_policy(evaluation_cli):
    args, observed = evaluation_cli
    result = runner.invoke(cli.app, args + ["--split", "dev", "--max-episodes", "1"])
    assert result.exit_code == 0, result.output
    assert observed["constructed"] and observed["closed"]
    assert observed["run_options"]["split"] == "dev"
    assert observed["run_options"]["attempts"] == 1
    assert observed["run_options"]["mode"] == "evaluate"
    assert observed["policy_options"]["prompt_style"] == "natural"
    assert observed["policy_options"]["instruction_map"] is None


def test_evaluation_inherits_code_style_and_only_passes_static_text(evaluation_cli):
    from scriptwob.prompt_styles import build_instruction_map

    args, observed = evaluation_cli
    observed["metadata"] = {"dataset": {"metadata": {"prompt_style": "code"}}}
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    options = observed["policy_options"]
    assert options["prompt_style"] == "code"
    assert options["instruction_map"] == build_instruction_map(observed["manifest"], "code")
    assert all(isinstance(text, str) for text in options["instruction_map"].values())
    result = runner.invoke(cli.app, args + ["--prompt-style", "natural"])
    assert result.exit_code == 0, result.output
    assert observed["policy_options"]["prompt_style"] == "natural"
    assert observed["policy_options"]["instruction_map"] is None


def test_evaluation_rejects_invalid_prompt_style_before_loading(evaluation_cli):
    args, observed = evaluation_cli
    result = runner.invoke(cli.app, args + ["--prompt-style", "typo"])
    assert result.exit_code != 0
    assert not observed["constructed"]


def test_prompt_preview_and_existing_export_conversion(tmp_path):
    from test_script_export import fixture_run

    from scriptwob.seeclick_data import load_training_export

    source, _ = fixture_run(tmp_path)
    manifest_path = source / "manifest.json"
    preview = tmp_path / "prompts.jsonl"
    before = manifest_path.read_bytes()
    result = runner.invoke(cli.app, ["render-prompts", str(manifest_path), str(preview)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in preview.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["prompt_style"] == "code"
    assert rows[0]["instruction"] != rows[0]["original_instruction"]
    exported = tmp_path / "data"
    result = runner.invoke(cli.app, [
        "export", str(source), str(exported), "--format", "seeclick",
    ])
    assert result.exit_code == 0, result.output
    converted = tmp_path / "code-data"
    result = runner.invoke(cli.app, [
        "reformat-seeclick", str(exported), str(manifest_path), str(converted),
    ])
    assert result.exit_code == 0, result.output
    loaded = load_training_export(converted)
    assert loaded.metadata["prompt_style"] == "code"
    assert rows[0]["instruction"] in loaded.samples[0]["conversations"][0]["value"]
    assert manifest_path.read_bytes() == before


def test_evaluation_cli_rejects_train_before_loading_model(evaluation_cli):
    args, observed = evaluation_cli
    result = runner.invoke(cli.app, args + ["--split", "train"])
    assert result.exit_code != 0
    assert not observed["constructed"]


def test_evaluation_manifest_mismatch_needs_explicit_transfer(evaluation_cli):
    args, observed = evaluation_cli
    observed["metadata"] = {"dataset": {"metadata": {"manifest_digest": "different"}}}
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0 and "manifest differs" in result.output
    assert not observed["constructed"]
    result = runner.invoke(cli.app, args + ["--allow-different-manifest"])
    assert result.exit_code == 0 and observed["closed"]


def test_evaluation_failure_still_closes_model(evaluation_cli, monkeypatch):
    args, observed = evaluation_cli

    def run(*args, **kwargs):
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr(cli, "run", run)
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0 and observed["closed"]


def test_fresh_evaluation_preserves_existing_run_and_closes_policy(evaluation_cli, tmp_path):
    args, observed = evaluation_cli
    original = tmp_path / "eval"
    original.mkdir()
    metadata = b'{"manifest_digest": "legacy"}\n'
    (original / "run.json").write_bytes(metadata)
    result = runner.invoke(cli.app, args + ["--fresh"])
    assert result.exit_code == 0, result.output
    assert observed["constructed"] and observed["closed"]
    actual = observed["output"]
    assert actual != original and actual.parent == original.parent
    assert actual.is_dir()
    assert f"Fresh run directory: {actual.resolve()}" in result.stderr
    assert "fresh" not in observed["run_options"]
    assert (original / "run.json").read_bytes() == metadata


def test_evaluation_resume_error_is_clean_and_still_closes_policy(evaluation_cli, monkeypatch):
    args, observed = evaluation_cli

    def run(*args, **kwargs):
        raise ResumeConfigurationError("Resume configuration differs: policy")

    monkeypatch.setattr(cli, "run", run)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1 and observed["closed"]
    assert isinstance(result.exception, SystemExit)
    assert "Resume configuration differs: policy" in result.stderr
    assert "Traceback" not in result.output
