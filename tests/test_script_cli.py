import json

from typer.testing import CliRunner

from scriptwob.cli import app
from scriptwob.generation import FAMILIES, read_manifest

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
