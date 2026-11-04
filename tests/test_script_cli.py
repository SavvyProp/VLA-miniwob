import json
from pathlib import Path

import pytest
from test_script_collection import FakeEnvironment, Policy
from typer.testing import CliRunner

from scriptwob import cli
from scriptwob.cli import app
from scriptwob.collection import ResumeConfigurationError
from scriptwob.collection import run as collection_run
from scriptwob.export import export_dataset
from scriptwob.generation import FAMILIES, generate_manifest, read_manifest, write_manifest

runner = CliRunner()


def test_generate_inspect_validate_and_refuse_overwrite(tmp_path):
    path = tmp_path / "tasks.json"
    result = runner.invoke(
        app, ["generate", str(path), "--train", "12", "--dev", "6", "--test", "6", "--seed", "61"]
    )
    assert result.exit_code == 0, result.output
    manifest = read_manifest(path)
    assert len(manifest.episodes) == 24
    result = runner.invoke(app, ["inspect", str(path)])
    assert json.loads(result.output)["splits"] == {"train": 12, "dev": 6, "test": 6}
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["validated_episodes"] == 24
    original = path.read_bytes()
    assert runner.invoke(app, ["generate", str(path)]).exit_code != 0
    assert path.read_bytes() == original


def test_catalog_and_structure_default_levels(tmp_path):
    result = runner.invoke(app, ["catalog"])
    catalog = json.loads(result.output)
    assert sum(len(row["families"]) for row in catalog.values()) == sum(map(len, FAMILIES.values()))
    path = tmp_path / "tasks.json"
    result = runner.invoke(
        app,
        [
            "generate",
            str(path),
            "--train",
            "8",
            "--dev",
            "4",
            "--test",
            "4",
            "--split-strategy",
            "structure",
        ],
    )
    assert result.exit_code == 0, result.output
    assert {episode.level for episode in read_manifest(path).episodes} == {2, 3, 4, 5}


def test_cli_rejects_tampered_instructions_and_missing_api_key(tmp_path, monkeypatch):
    path = tmp_path / "tasks.json"
    assert (
        runner.invoke(
            app, ["generate", str(path), "--train", "1", "--dev", "0", "--test", "0"]
        ).exit_code
        == 0
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["collect", str(path), str(tmp_path / "run")])
    assert result.exit_code != 0 and "OPENAI_API_KEY" in result.output
    assert not (tmp_path / "run").exists()
    content = json.loads(path.read_text())
    content["episodes"][0]["instruction"] = "Do something different."
    path.write_text(json.dumps(content))
    assert runner.invoke(app, ["validate", str(path)]).exit_code != 0


@pytest.mark.parametrize("command", ["collect", "evaluate"])
def test_fresh_rollouts_create_distinct_runs_and_preserve_legacy_data(
    tmp_path, monkeypatch, command
):
    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=1, dev=1, test=1, seed=77))
    original = tmp_path / "astra"
    original.mkdir()
    legacy_metadata = b'{"mode": "collect", "manifest_digest": "legacy"}\n'
    (original / "run.json").write_bytes(legacy_metadata)
    (original / "episodes").mkdir()
    (original / "episodes" / "old.json").write_text("old demonstration\n")
    outputs = []

    def collect_run(manifest, output, policy, **kwargs):
        # The CLI must reserve a new directory before entering the collector.
        assert output.is_dir()
        assert not any(output.iterdir())
        assert "fresh" not in kwargs
        outputs.append(output)
        return collection_run(manifest, output, policy, env_factory=FakeEnvironment, **kwargs)

    monkeypatch.setattr(cli, "run", collect_run)
    monkeypatch.setattr(cli, "_policy", lambda *args: Policy())
    args = [command, str(manifest_path), str(original), "--teacher", "oracle", "--fresh"]
    for _ in range(2):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        report = json.loads(result.stdout)
        actual = Path(report["output_dir"])
        assert actual == outputs[-1].resolve()
        assert actual.parent == original.parent
        assert actual != original
        assert f"Fresh run directory: {actual}" in result.stderr
        assert (actual / "report.json").is_file()
        assert report["new_attempts"] == report["total_attempts"] == 1
        assert report["successful_episodes"] == 1
        assert report["new_predict_calls"] == 2
        assert len(list((actual / "attempts").glob("*/private.json"))) == 1

    assert outputs[0] != outputs[1]
    assert (original / "run.json").read_bytes() == legacy_metadata
    assert (original / "episodes" / "old.json").read_text() == "old demonstration\n"
    assert {path.name for path in original.iterdir()} == {"run.json", "episodes"}


@pytest.mark.parametrize("command", ["collect", "evaluate"])
def test_rollout_resume_configuration_error_is_a_clean_cli_error(tmp_path, monkeypatch, command):
    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=1, dev=1, test=1, seed=77))
    output = tmp_path / "run"

    def reject_resume(manifest, actual_output, policy, **kwargs):
        assert actual_output == output
        raise ResumeConfigurationError("Resume configuration differs: manifest_digest")

    monkeypatch.setattr(cli, "run", reject_resume)
    result = runner.invoke(app, [command, str(manifest_path), str(output), "--teacher", "oracle"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Resume configuration differs: manifest_digest" in result.stderr
    assert "Traceback" not in result.output
    assert not output.exists()


def test_all_splits_cli_collects_and_preserves_training_boundary(tmp_path, monkeypatch):
    from scriptwob import suite

    manifest_path = tmp_path / "tasks.json"
    manifest = generate_manifest(train=2, dev=1, test=1, seed=77)
    write_manifest(manifest_path, manifest)
    original = tmp_path / "all-data"
    original.mkdir()
    sentinel = original / "run.json"
    sentinel.write_text("legacy run\n")
    collect_suite = suite.collect_suite
    policy = Policy()

    def run_suite(*args, **kwargs):
        return collect_suite(*args, env_factory=FakeEnvironment, **kwargs)

    monkeypatch.setattr(FakeEnvironment, "fail", False)
    monkeypatch.setattr(suite, "collect_suite", run_suite)
    monkeypatch.setattr(cli, "_policy", lambda *args: policy)
    result = runner.invoke(app, [
        "collect", str(manifest_path), str(original), "--all-splits", "--fresh",
        "--attempts", "3", "--max-total-steps", "20",
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    root = Path(report["output_dir"])
    assert root != original and root.parent == original.parent
    assert f"Fresh run directory: {root}" in result.stderr
    assert sentinel.read_text() == "legacy run\n"
    assert report["stop_reason"] == "complete"
    assert report["new_predict_calls"] == policy.total_calls == 8
    assert json.loads((root / "report.json").read_text()) == report
    for split, count in (("train", 2), ("dev", 1), ("test", 1)):
        directory = root / split
        metadata = json.loads((directory / "run.json").read_text())
        assert metadata["config"]["split"] == split
        assert metadata["config"]["mode"] == ("collect" if split == "train" else "evaluate")
        assert metadata["limits"]["attempts"] == (3 if split == "train" else 1)
        verdicts = list((directory / "attempts").glob("*/private.json"))
        assert len(verdicts) == count
        assert all(json.loads(path.read_text())["accepted"] == (split == "train")
                   for path in verdicts)

    exported = export_dataset(root / "train", tmp_path / "training", format="seeclick")
    assert exported["episodes"] == 2
    for split in ("dev", "test"):
        with pytest.raises(ValueError, match="train"):
            export_dataset(root / split, tmp_path / f"export-{split}")
