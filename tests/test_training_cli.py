"""CLI wiring tests run in the base environment without torch or a browser."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vla_miniwob import cli, tinyvla_policy, training
from vla_miniwob.collection import write_json
from vla_miniwob.tasks import make_manifest


def test_train_command_maps_overrides_resume_and_development_source(tmp_path, monkeypatch):
    configuration = {"chunk_size": 16, "training": {"max_steps": 10, "seed": 9}}
    config = tmp_path / "recipe.json"
    write_json(config, configuration)
    captured = {}

    def fake_train(source, output, recipe, **kwargs):
        captured.update(source=source, output=output, recipe=recipe, **kwargs)
        return {"step": 20, "trained": True}

    monkeypatch.setattr(training, "train", fake_train)
    result = CliRunner().invoke(
        cli.app,
        [
            "train-tinyvla",
            str(tmp_path / "train"),
            str(tmp_path / "model"),
            "--config",
            str(config),
            "--dev-source",
            str(tmp_path / "dev"),
            "--max-steps",
            "20",
            "--batch-size",
            "2",
            "--gradient-accumulation",
            "3",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--resume",
            "--preflight-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"step": 20, "trained": True}
    assert captured == {
        "source": tmp_path / "train",
        "output": tmp_path / "model",
        "recipe": {
            "chunk_size": 16,
            "training": {
                "max_steps": 20,
                "seed": 9,
                "batch_size": 2,
                "gradient_accumulation": 3,
            },
        },
        "dev_source": tmp_path / "dev",
        "device": "cpu",
        "dtype": "float32",
        "resume": True,
        "preflight_only": True,
    }
    assert json.loads(config.read_text()) == configuration


@pytest.mark.parametrize(
    "error, expected",
    [
        (ValueError("Resume settings differ"), "Resume settings differ"),
        (ImportError("No torch"), "scripts/setup_tinyvla.sh cuda"),
    ],
)
def test_training_setup_errors_are_actionable(tmp_path, monkeypatch, error, expected):
    config = tmp_path / "config.json"
    write_json(config, {"chunk_size": 16})

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(training, "train", fail)
    result = CliRunner().invoke(
        cli.app,
        [
            "train-tinyvla",
            "data",
            "model",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 2
    message = " ".join(result.output.replace("│", " ").split())
    assert expected in message
    assert "Traceback" not in result.output


@pytest.mark.parametrize("split, limit", [("eval", None), ("dev", None), ("eval", 2)])
def test_evaluation_runs_selected_split_once_and_saves_report(tmp_path, monkeypatch, split, limit):
    manifest = make_manifest(1, 2, 3, tasks=["click-button"])
    path = tmp_path / "manifest.json"
    write_json(path, manifest.model_dump())
    captured = {}
    policy = object()

    def fake_policy(checkpoint, **kwargs):
        captured["policy"] = {"checkpoint": checkpoint, **kwargs}
        return policy

    def fake_run(value, output, agent, **kwargs):
        assert value == manifest
        assert agent is policy
        captured["run"] = kwargs
        output.mkdir()
        return {"new_attempts": kwargs["max_episodes"]}

    def fake_report(output):
        assert output == tmp_path / "evaluation"
        return {"attempts": captured["run"]["max_episodes"], "split": split}

    monkeypatch.setattr(tinyvla_policy, "TinyVLAPolicy", fake_policy)
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "report", fake_report)
    arguments = [
        "evaluate-tinyvla",
        "checkpoint",
        str(path),
        str(tmp_path / "evaluation"),
        "--split",
        split,
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--no-headless",
    ]
    if limit is not None:
        arguments.extend(["--max-episodes", str(limit)])
    result = CliRunner().invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    expected = limit or sum(item.split == split for item in manifest.instances)
    assert captured["policy"] == {
        "checkpoint": Path("checkpoint"),
        "device": "cpu",
        "dtype": "float32",
    }
    assert captured["run"] == {
        "mode": "evaluate",
        "split": split,
        "max_episodes": expected,
        "attempts": 1,
        "headless": False,
    }
    assert json.loads(result.output) == {"new_attempts": expected}
    assert json.loads((tmp_path / "evaluation" / "report.json").read_text()) == {
        "attempts": expected,
        "split": split,
    }


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("train_split", "split must be dev or eval"),
        ("wrong_horizon", "H=16 and K=1"),
        ("wrong_prefix", "H=16 and K=1"),
        ("empty_split", "no dev instances"),
    ],
)
def test_evaluation_rejects_invalid_protocol_before_model_loading(
    tmp_path,
    monkeypatch,
    condition,
    expected,
):
    manifest = make_manifest(1, 0, 1, tasks=["click-button"])
    split = "eval"
    if condition == "train_split":
        split = "train"
    elif condition == "wrong_horizon":
        manifest.prediction_horizon = 8
    elif condition == "wrong_prefix":
        manifest.execution_prefix = 2
    else:
        split = "dev"
    path = tmp_path / "manifest.json"
    write_json(path, manifest.model_dump())

    def should_not_load(*_args, **_kwargs):
        pytest.fail("Invalid benchmark protocol reached model loading")

    monkeypatch.setattr(tinyvla_policy, "TinyVLAPolicy", should_not_load)
    result = CliRunner().invoke(
        cli.app,
        [
            "evaluate-tinyvla",
            "checkpoint",
            str(path),
            str(tmp_path / "evaluation"),
            "--split",
            split,
        ],
    )
    assert result.exit_code == 2
    assert expected in " ".join(result.output.replace("│", " ").split())
    assert not (tmp_path / "evaluation").exists()
