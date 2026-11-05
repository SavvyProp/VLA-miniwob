import errno
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


@pytest.mark.parametrize("command", ["collect", "evaluate"])
@pytest.mark.parametrize("retries", [None, 0])
def test_cli_passes_transport_retry_configuration_to_teacher(
    tmp_path, monkeypatch, command, retries
):
    path = tmp_path / "tasks.json"
    write_manifest(path, generate_manifest(train=1, dev=1, test=1, seed=77))
    captured = []

    def teacher(config):
        captured.append(config)
        return Policy()

    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setattr(cli, "OpenAITeacher", teacher)
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: {"stop_reason": "complete"})
    args = [command, str(path), str(tmp_path / "run")]
    if retries is not None:
        args += ["--transport-retries", str(retries)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert captured[0].transport_retries == (3 if retries is None else retries)


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


def test_disk_full_while_reserving_fresh_output_is_reported_without_model_calls(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=1, dev=0, test=0, seed=77))

    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "secret-exception", "secret-filename")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("No rollouts should start")

    monkeypatch.setattr(cli.tempfile, "mkdtemp", disk_full)
    monkeypatch.setattr(cli, "run", unexpected_run)
    result = runner.invoke(app, ["collect", str(manifest_path), str(tmp_path / "output"),
                                 "--fresh", "--teacher", "oracle"])
    assert result.exit_code == 1
    assert not result.stdout
    assert "No space left on device" in result.stderr
    assert "stage=prepare_output" in result.stderr
    assert "secret-" not in result.output


@pytest.mark.parametrize("command", ["collect", "evaluate", "evaluate-seeclick", "all-splits"])
def test_aborted_rollout_prints_error_and_actual_fresh_path(tmp_path, monkeypatch, command):
    from scriptwob import seeclick_policy, seeclick_runtime, suite

    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=1, dev=1, test=1, seed=77))
    original = tmp_path / "run"
    observed = {}
    failure = {
        "episode_id": "L5-example", "reason": "infrastructure_error", "error_type": "OSError",
        "error": {
            "stage": "environment_reset", "exception_type": "OSError", "errno": 28,
            "errno_name": "ENOSPC", "os_message": "No space left on device", "traceback": [],
        },
    }

    def stopped_run(manifest, output, policy, **kwargs):
        observed["output"] = output
        child = {"stop_reason": "infrastructure_error", "failures": {"last_error": failure}}
        report = child
        if command == "all-splits":
            report = {
                "stop_reason": "infrastructure_error",
                "splits": {
                    "train": {"ran_this_invocation": True, "report": child},
                    "dev": {"ran_this_invocation": False, "report": None},
                },
            }
        observed["report"] = report
        return report

    class StubSeeClick:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(cli, "run", stopped_run)
    monkeypatch.setattr(suite, "collect_suite", stopped_run)
    monkeypatch.setattr(seeclick_policy, "SeeClickPolicy", StubSeeClick)
    monkeypatch.setattr(seeclick_runtime, "resolve_checkpoint", lambda *args: (None, None, {}))
    if command == "evaluate-seeclick":
        args = [command, "base", str(manifest_path), str(original)]
    else:
        args = ["collect" if command == "all-splits" else command,
                str(manifest_path), str(original), "--teacher", "oracle"]
        if command == "all-splits":
            args.append("--all-splits")
    result = runner.invoke(app, [*args, "--fresh"])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout) == observed["report"]
    label = "Evaluation" if command.startswith("evaluate") else "Collection"
    assert f"{label} stopped: infrastructure_error" in result.stderr
    for detail in ("episode=L5-example", "stage=environment_reset", "OSError", "errno=28 ENOSPC",
                   "No space left on device"):
        assert detail in result.stderr
    assert "Free space or choose an output directory on another filesystem." in result.stderr
    assert f"Report: {(observed['output'] / 'report.json').resolve()}" in result.stderr
    assert observed["output"] != original
    assert result.output.rfind(f"{label} stopped:") > result.output.rfind('"stop_reason"')
    if command == "all-splits":
        assert "split=train" in result.stderr
    if command == "evaluate-seeclick":
        assert observed["closed"]


@pytest.mark.parametrize("reason", ["provider_error", "infrastructure_error", "runner_error"])
def test_stop_summary_handles_older_reports_without_details(tmp_path, monkeypatch, reason):
    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=1, dev=0, test=0, seed=77))
    report = {"stop_reason": reason, "failures": {"by_reason": {reason: 1}}}
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: report)
    output = tmp_path / "run"
    result = runner.invoke(app, ["collect", str(manifest_path), str(output), "--teacher", "oracle"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == report
    assert f"Collection stopped: {reason}\n" in result.stderr
    assert f"Report: {(output / 'report.json').resolve()}" in result.stderr


def test_stop_summary_does_not_print_raw_provider_messages_or_tracebacks(tmp_path, capsys):
    secret = "credential-from-provider-payload"
    cli._print_stop_summary({
        "stop_reason": "provider_error",
        "failures": {"last_error": {
            "episode_id": "episode-one", "error_type": "APIConnectionError",
            "error": {
                "stage": "policy_predict", "exception_type": "APIConnectionError",
                "message": secret, "filename": secret, "traceback": [{"filename": secret}],
            },
        }},
        "policy": {"api_key": secret},
    }, tmp_path, mode="collect")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "APIConnectionError" in captured.err
    assert "stage=policy_predict" in captured.err
    assert secret not in captured.err


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_disk_full_preventing_report_write_is_a_clean_cli_error(tmp_path, monkeypatch, code):
    from scriptwob import collection

    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=1, dev=0, test=0, seed=77))
    original = tmp_path / "run"
    secret = "private-credential-in-error-message-and-filename"
    actual_outputs = []
    write_json = collection.write_json

    def cannot_write_report(path, value):
        if path.name == "report.json":
            raise OSError(code, secret, f"/private/{secret}")
        write_json(path, value)

    def run_with_mock_environment(manifest, output, policy, **kwargs):
        actual_outputs.append(output)
        return collection_run(manifest, output, policy, env_factory=FakeEnvironment, **kwargs)

    monkeypatch.setattr(collection, "write_json", cannot_write_report)
    monkeypatch.setattr(cli, "run", run_with_mock_environment)
    monkeypatch.setattr(cli, "_policy", lambda *args: Policy())
    result = runner.invoke(app, ["collect", str(manifest_path), str(original), "--fresh"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""
    assert "Collection stopped: infrastructure_error" in result.stderr
    assert f"errno={code} {errno.errorcode[code]}" in result.stderr
    assert "Free space or choose an output directory on another filesystem." in result.stderr
    assert "may be incomplete or missing" in result.stderr
    assert f"Report: {(actual_outputs[0] / 'report.json').resolve()}" in result.stderr
    assert actual_outputs[0] != original
    assert "Traceback" not in result.output
    assert secret not in result.output


def test_fatal_os_error_without_errno_omits_raw_exception_message(tmp_path, monkeypatch):
    manifest_path = tmp_path / "tasks.json"
    write_manifest(manifest_path, generate_manifest(train=0, dev=1, test=0, seed=77))
    secret = "private-credential-in-os-error"

    def fail(*args, **kwargs):
        raise OSError(secret)

    monkeypatch.setattr(cli, "run", fail)
    result = runner.invoke(app, ["evaluate", str(manifest_path), str(tmp_path / "run"),
                                 "--split", "dev", "--teacher", "oracle"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Evaluation stopped: infrastructure_error" in result.stderr
    assert "OSError" in result.stderr
    assert "errno=" not in result.stderr
    assert "may be incomplete or missing" in result.stderr
    assert secret not in result.output
