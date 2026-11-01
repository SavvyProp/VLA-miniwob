"""Exercise the real optimizer/checkpoint loop using a tiny injected torch backend."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_dataset import fixture_run

torch = pytest.importorskip("torch")

from vla_miniwob import training  # noqa: E402
from vla_miniwob.collection import write_json  # noqa: E402
from vla_miniwob.dataset import export_hdf5, fit_normalization, masked_diffusion_loss  # noqa: E402
from vla_miniwob.training import train, verify_checkpoint  # noqa: E402


class FakeProcessor:
    max_length = 2048
    image_processor = SimpleNamespace(image_mean=[0, 0, 0], image_std=[1, 1, 1])

    def text_ids(self, text):
        return list(range(len(text.split())))

    def image(self, image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) / 255

    def __call__(self, sample):
        return {key: torch.from_numpy(sample[key].copy()) for key in ("state", "action", "is_pad")}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.randn(3, 3) * 0.1)
        self.projection = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, state):
        return self.dropout(state @ self.lora_weight) + self.projection(state)


class FakeBackend:
    image_tokens = 4

    def __init__(self, recipe):
        self.recipe = recipe
        self.model = FakeModel()
        self.processor = FakeProcessor()

    @classmethod
    def from_pretrained(cls, recipe, **_kwargs):
        return cls(recipe)

    @classmethod
    def from_checkpoint(cls, checkpoint, **_kwargs):
        result = cls(json.loads((checkpoint / "recipe.json").read_text()))
        result.model.load_state_dict(torch.load(checkpoint / "weights.pt", weights_only=True))
        return result

    @staticmethod
    def collate(samples):
        return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}

    def loss(self, batch, *, generator):
        target = batch["action"] + torch.randn(batch["action"].shape, generator=generator) * 0.05
        # Exercise restoration of independent torch noise, global torch dropout,
        # Python, and NumPy random states in addition to the sample generator.
        target = target + random.random() * 0.03 + np.random.random() * 0.03
        prediction = self.model(batch["state"])[:, None].expand_as(target)
        return masked_diffusion_loss(prediction, target, batch["is_pad"])

    def save(self, output):
        torch.save(self.model.state_dict(), output / "weights.pt")
        write_json(output / "recipe.json", self.recipe)


@pytest.fixture(autouse=True)
def stable_fake_runtime(monkeypatch):
    # These tests require only torch, not installation of the actual VLM stack.
    monkeypatch.setattr(training, "version", lambda name: f"test-{name}")
    monkeypatch.setattr(training, "implementation_digest", lambda: "test-implementation")


def recipe(steps=4, **changes):
    return {
        "chunk_size": 4,
        "training": {
            "max_steps": steps,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "learning_rate": 0.01,
            "non_lora_lr": 0.005,
            "warmup_steps": 3,
            "save_every": 2,
            "gradient_checkpointing": False,
            **changes,
        },
    }


def run(source, output, configuration=None, **kwargs):
    return train(
        source,
        output,
        configuration or recipe(),
        device="cpu",
        dtype="float32",
        backend_class=FakeBackend,
        progress=lambda *_args, **_kwargs: None,
        **kwargs,
    )


def metrics(output):
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def test_uninterrupted_and_resumed_training_are_identical(tmp_path):
    source = fixture_run(tmp_path)
    full = run(source, tmp_path / "full")
    run(source, tmp_path / "resumed", recipe(2))
    resumed = run(source, tmp_path / "resumed", recipe(4), resume=True)
    uninterrupted_weights = torch.load(Path(full["checkpoint"]) / "weights.pt", weights_only=True)
    resumed_weights = torch.load(Path(resumed["checkpoint"]) / "weights.pt", weights_only=True)
    for name in uninterrupted_weights:
        torch.testing.assert_close(
            uninterrupted_weights[name], resumed_weights[name], rtol=0, atol=0
        )
    expected = metrics(tmp_path / "full")
    actual = metrics(tmp_path / "resumed")
    for left, right in zip(expected, actual, strict=True):
        assert {key: value for key, value in left.items() if key != "seconds"} == {
            key: value for key, value in right.items() if key != "seconds"
        }
    assert [item["step"] for item in actual] == [1, 2, 3, 4]
    assert expected[0]["learning_rates"] == pytest.approx([0.01 / 3, 0.005 / 3])


def test_checkpoint_integrity_existing_output_and_completed_resume(tmp_path):
    source = fixture_run(tmp_path)
    output = tmp_path / "trained"
    result = run(source, output, recipe(2))
    checkpoint = Path(result["checkpoint"])
    metadata = verify_checkpoint(checkpoint)
    assert metadata["step"] == 2
    assert "training_state.pt" in metadata["files"]
    saved = (checkpoint / "weights.pt").read_bytes()
    with pytest.raises(ValueError, match="Training output exists"):
        run(source, output, recipe(2))
    finished = run(source, output, recipe(2), resume=True)
    assert finished["already_complete"] is True
    assert (checkpoint / "weights.pt").read_bytes() == saved
    with (checkpoint / "weights.pt").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum differs"):
        run(source, output, recipe(4), resume=True)


@pytest.mark.parametrize("change", ["omitted_checksum", "extra_payload"])
def test_checkpoint_rejects_unchecksummed_payloads(tmp_path, change):
    source = fixture_run(tmp_path)
    result = run(source, tmp_path / "trained", recipe(1))
    checkpoint = Path(result["checkpoint"])
    if change == "omitted_checksum":
        metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        del metadata["files"]["normalization.json"]
        write_json(checkpoint / "checkpoint.json", metadata)
    else:
        (checkpoint / "untracked_weights.pt").write_bytes(b"untracked payload")
    with pytest.raises(ValueError):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("change", ["data", "recipe", "code", "decreased_steps"])
def test_resume_rejects_changed_training_contract(tmp_path, monkeypatch, change):
    source = fixture_run(tmp_path)
    exported = tmp_path / "export"
    export_hdf5(source, exported, horizon=4)
    output = tmp_path / "trained"
    run(exported, output, recipe(2))
    updated_recipe = recipe(4)
    if change == "data":
        with h5py.File(exported / "episode_000000.hdf5", "r+") as file:
            file["action"][0] = [0.4, 0.3, 1]
    elif change == "recipe":
        updated_recipe = recipe(4, gradient_accumulation=1)
    elif change == "code":
        monkeypatch.setattr(training, "implementation_digest", lambda: "new-implementation")
    else:
        updated_recipe = recipe(1)
    with pytest.raises(ValueError, match="Resume settings/data/code differ"):
        run(exported, output, updated_recipe, resume=True)


def test_eval_examples_cannot_enter_training(tmp_path):
    source = fixture_run(tmp_path)
    exported = tmp_path / "eval"
    export_hdf5(source, exported, split="eval", stats=fit_normalization(source))
    with pytest.raises(ValueError, match="train export"):
        run(exported, tmp_path / "bad")
    assert not list((tmp_path / "bad").glob("checkpoint-*"))


def test_preflight_has_no_optimization_or_checkpoint(tmp_path):
    source = fixture_run(tmp_path)
    output = tmp_path / "checked"
    result = run(source, output, preflight_only=True)
    assert result["trained"] is False
    assert (output / "preflight.json").is_file()
    assert list(output.glob("train-vision-*.png"))
    assert not (output / "metrics.jsonl").exists()
    assert not list(output.glob("checkpoint-*"))


def test_preflight_can_resume_into_training(tmp_path):
    source = fixture_run(tmp_path)
    output = tmp_path / "checked"
    run(source, output, recipe(2), preflight_only=True)
    # A failed run before its first checkpoint can have incomplete metrics too.
    (output / "metrics.jsonl").write_text('{"step":1,"epsilon_loss":999}\n{"step":')
    resumed = run(source, output, recipe(2), resume=True)
    assert resumed["step"] == 2
    assert [row["step"] for row in metrics(output)] == [1, 2]
    assert all(row["epsilon_loss"] != 999 for row in metrics(output))


@pytest.mark.parametrize("pointer", ["missing", "stale"])
def test_resume_recovers_published_checkpoint_and_trims_lost_metrics(tmp_path, pointer):
    source = fixture_run(tmp_path)
    output = tmp_path / "interrupted"
    result = run(source, output, recipe(4))
    published = Path(result["checkpoint"])
    original_weights = (published / "weights.pt").read_bytes()
    if pointer == "missing":
        (output / "latest.json").unlink()
    else:
        write_json(output / "latest.json", {"checkpoint": "checkpoint-0000002", "step": 2})
    with (output / "metrics.jsonl").open("a") as stream:
        stream.write('{"step":5,"epsilon_loss":999}\n{"step":')
    resumed = run(source, output, recipe(6), resume=True)
    assert resumed["step"] == 6
    assert (published / "weights.pt").read_bytes() == original_weights
    rows = metrics(output)
    assert [row["step"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert all(row["epsilon_loss"] != 999 for row in rows)
