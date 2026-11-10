import pytest
from typer.testing import CliRunner

from scriptwob import cli, gui_owl_policy, gui_owl_runtime, gui_owl_training
from scriptwob.generation import generate_manifest, write_manifest

runner = CliRunner()


def test_training_passes_history_config_and_resume(tmp_path, monkeypatch):
    received = {}

    def train(source, output, config, **kwargs):
        received.update(config=config, **kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(gui_owl_training, "train_gui_owl", train)
    result = runner.invoke(cli.app, [
        "train-gui-owl", str(tmp_path / "source"), str(tmp_path / "out"),
        "--config", "configs/gui_owl_4b_3090.json", "--max-steps", "9",
        "--history-images", "0", "--resume", "--prepare-only", "--prompt-style", "code",
    ])
    assert result.exit_code == 0, result.output
    assert received["config"].model.model_id.endswith("4B-Instruct")
    assert received["config"].max_steps == 9 and received["config"].history_images == 0
    assert received["resume"] and received["prepare_only"]
    assert received["prompt_style"] == "code"


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    manifest = generate_manifest(train=1, dev=1, test=1, seed=77)
    path = tmp_path / "tasks.json"
    write_manifest(path, manifest)
    state = {"constructed": False, "closed": False, "metadata": {
        "dataset": {"metadata": {"prompt_style": "code", "manifest_digest": manifest.digest}},
    }}

    class Policy:
        def __init__(self, checkpoint, **kwargs):
            state.update(constructed=True, policy_options=kwargs)

        def close(self):
            state["closed"] = True

    def run(*args, **kwargs):
        state["run_options"] = kwargs
        return {"stop_reason": "complete"}

    monkeypatch.setattr(gui_owl_runtime, "resolve_checkpoint", lambda *a, **k: (
        None, None, state["metadata"],
    ))
    monkeypatch.setattr(gui_owl_policy, "GUIOwlPolicy", Policy)
    monkeypatch.setattr(cli, "run", run)
    return ["evaluate-gui-owl", "base-2b", str(path), str(tmp_path / "eval")], state


def test_evaluation_uses_code_one_attempt_and_history_override(evaluation):
    args, state = evaluation
    result = runner.invoke(cli.app, args + ["--split", "dev", "--history-images", "0"])
    assert result.exit_code == 0, result.output
    assert state["closed"] and state["constructed"]
    assert state["run_options"]["attempts"] == 1
    assert state["run_options"]["split"] == "dev"
    assert state["policy_options"]["history_images"] == 0
    assert state["policy_options"]["prompt_style"] == "code"
    assert state["policy_options"]["instruction_map"]
    assert all(isinstance(text, str)
               for text in state["policy_options"]["instruction_map"].values())


def test_evaluation_rejects_training_split_and_manifest_mismatch(evaluation):
    args, state = evaluation
    result = runner.invoke(cli.app, args + ["--split", "train"])
    assert result.exit_code != 0 and not state["constructed"]
    state["metadata"]["dataset"]["metadata"]["manifest_digest"] = "different"
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0 and not state["constructed"]
    result = runner.invoke(cli.app, args + ["--allow-different-manifest"])
    assert result.exit_code == 0 and state["closed"]


def test_evaluation_closes_on_browser_error(evaluation, monkeypatch):
    args, state = evaluation

    def fail(*args, **kwargs):
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr(cli, "run", fail)
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0 and state["closed"]


def test_help_needs_no_model_runtime():
    for command in ("train-gui-owl", "evaluate-gui-owl"):
        result = runner.invoke(cli.app, [command, "--help"])
        assert result.exit_code == 0 and "history-images" in result.output
