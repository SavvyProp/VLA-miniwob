from typer.testing import CliRunner

from vla_miniwob import cli
from vla_miniwob.collection import ResumeConfigurationError
from vla_miniwob.tasks import make_manifest


def test_collect_explains_resume_mismatch_without_traceback(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(make_manifest(1, 0, 0, tasks=["click-button"]).model_dump_json())
    monkeypatch.setattr(cli, "_teacher", lambda *_: object())

    def incompatible_run(*args, **kwargs):
        raise ResumeConfigurationError(
            "Resume configuration differs: attempts_per_instance "
            "(saved --attempts 3, requested --attempts 2)."
        )

    monkeypatch.setattr(cli, "run", incompatible_run)
    result = CliRunner().invoke(cli.app, ["collect", str(manifest), str(tmp_path / "run")])
    assert result.exit_code == 2
    # Rich wraps the error box to the terminal width.
    message = " ".join(result.output.replace("│", " ").split())
    assert "saved --attempts 3" in message
    assert "requested --attempts 2" in message
    assert "Traceback" not in result.output
