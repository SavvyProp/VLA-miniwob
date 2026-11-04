"""Real Chrome integration from generated task through accepted SFT dataset."""

import json

import pytest
from PIL import Image

from scriptwob.collection import run
from scriptwob.export import export_dataset
from scriptwob.generation import generate_manifest
from scriptwob.policies import OraclePolicy


@pytest.mark.browser
def test_collect_resume_export_and_evaluate(tmp_path):
    manifest = generate_manifest(train=6, dev=6, test=0, seed=88)
    root = tmp_path / "run"
    # A small prefix can be extended without changing the run fingerprint.
    first = run(manifest, root, OraclePolicy(), attempts=1, max_episodes=2)
    assert first["accepted_episodes"] == 2
    complete = run(manifest, root, OraclePolicy(), attempts=2)
    assert complete["accepted_episodes"] == 6 and complete["new_attempts"] == 4
    resumed = run(manifest, root, OraclePolicy(), attempts=2)
    assert resumed["new_attempts"] == 0
    assert json.loads((root / "run.json").read_text())["limit_history"]

    with pytest.raises(ValueError, match="oracle"):
        export_dataset(root, tmp_path / "forbidden")
    output = tmp_path / "messages"
    summary = export_dataset(root, output, allow_oracle=True)
    expected = sum(e.complexity.reference_actions for e in manifest.episodes if e.split == "train")
    assert summary["episodes"] == 6 and summary["examples"] == expected
    samples = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    for sample in samples:
        with Image.open(output / sample["images"][0]) as image:
            assert image.size == (1000, 760)
        answer = json.loads(sample["messages"][1]["content"])
        assert 0 <= answer["x"] <= 1 and 0 <= answer["y"] <= 1
    export_dataset(root, tmp_path / "seeclick", format="seeclick", allow_oracle=True)
    assert len(json.loads((tmp_path / "seeclick" / "train.json").read_text())) == expected

    evaluation = run(
        manifest, tmp_path / "evaluation", OraclePolicy(), mode="evaluate", split="dev", attempts=1
    )
    assert evaluation["successful_episodes"] == 6 and evaluation["accepted_episodes"] == 0
    with pytest.raises(ValueError, match="train"):
        export_dataset(tmp_path / "evaluation", tmp_path / "forbidden-eval", allow_oracle=True)
