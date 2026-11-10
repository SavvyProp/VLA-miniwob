"""Training lifecycle checks without downloading pretrained model weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_seeclick_data import make_export
from test_seeclick_training import FakeModel

from scriptwob import gui_owl_training as training
from scriptwob.gui_owl_runtime import GUIOwlModelConfig
from scriptwob.gui_owl_training import GUIOwlTrainingConfig


def config(**updates):
    return GUIOwlTrainingConfig(
        model=GUIOwlModelConfig(
            device="cpu", precision="fp32", quantization="none", gradient_checkpointing=False,
        ),
        max_length=1024, max_steps=2, gradient_accumulation_steps=1,
        save_steps=1, logging_steps=1,
    ).model_copy(update=updates)


@pytest.fixture
def stack(monkeypatch):
    observed = SimpleNamespace(models=[], trainers=[], datasets=[])

    class Dataset:
        def __init__(self, exported, processor, max_length, **kwargs):
            observed.datasets.append(kwargs)
            self.exported = exported

        def preflight(self):
            return {"examples": len(self.exported.samples), "tokens_max": 100,
                    "truncated_examples": 0}

    class Trainer:
        def __init__(self, model, args, **kwargs):
            self.model, self.args, self.kwargs = model, args, kwargs
            self.state = SimpleNamespace(global_step=0, epoch=0)
            self.resumed = None
            observed.trainers.append(self)

        def train(self, resume_from_checkpoint=None):
            self.resumed = resume_from_checkpoint
            self.state.global_step, self.state.epoch = self.args.max_steps, 1.0
            self._save_checkpoint(self.model, trial=None)
            for callback in self.kwargs["callbacks"]:
                callback.on_save(self.args, self.state, None)
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def _save_checkpoint(self, model, trial):
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            model.save_pretrained(path)
            (path / "trainer_state.json").write_text(json.dumps({
                "global_step": self.state.global_step,
            }))
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (path / name).write_bytes(b"test")

    def load(*args, **kwargs):
        observed.models.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(training, "load_processor", lambda config: object())
    monkeypatch.setattr(training, "load_model", load)
    monkeypatch.setattr(training, "GUIOwlDataset", Dataset)
    monkeypatch.setattr(training, "GUIOwlCollator", lambda processor: object())
    monkeypatch.setattr(training, "_training_stack", lambda: (
        Trainer, object, lambda **kwargs: SimpleNamespace(**kwargs), lambda seed: None,
    ))
    monkeypatch.setattr(training, "runtime_versions", lambda: {"test": "pinned"})
    monkeypatch.setattr(training, "implementation_digest", lambda: "pinned")
    return observed


def test_prepare_train_resume_and_checkpoint_metadata(stack, tmp_path):
    source, output = make_export(tmp_path / "data"), tmp_path / "train"
    prepared = training.train_gui_owl(source, output, config(), prepare_only=True)
    assert prepared["history_images"] == 3 and not stack.models
    result = training.train_gui_owl(source, output, config(), resume=True)
    assert result["global_step"] == 2
    assert stack.datasets[-1] == {"history_images": 3, "reserve_tokens": 64}
    metadata = json.loads((output / "adapter" / training.METADATA_FILE).read_text())
    assert metadata["schema_version"] == "scriptwob-gui-owl-training-v1"
    assert metadata["history_images"] == 3 and metadata["epoch"] == 1
    assert metadata["adapter_hashes"]
    training.train_gui_owl(source, output, config(max_steps=4), resume=True)
    assert stack.trainers[-1].resumed.endswith("checkpoint-2")
    assert training.latest_checkpoint(output).name == "checkpoint-4"


@pytest.mark.parametrize("update", [
    {"history_images": 0}, {"learning_rate": 0.002}, {"max_length": 2048},
])
def test_resume_rejects_changed_experiment_before_model_load(stack, tmp_path, update):
    source, output = make_export(tmp_path / "data"), tmp_path / "train"
    training.train_gui_owl(source, output, config(), prepare_only=True)
    before = (output / training.METADATA_FILE).read_bytes()
    with pytest.raises(ValueError, match="differs"):
        training.train_gui_owl(source, output, config(**update), resume=True)
    assert not stack.models
    assert (output / training.METADATA_FILE).read_bytes() == before


def test_resume_requires_optimizer_integrity_and_increasing_target(stack, tmp_path):
    source, output = make_export(tmp_path / "data"), tmp_path / "train"
    training.train_gui_owl(source, output, config())
    with pytest.raises(ValueError, match="already reached"):
        training.train_gui_owl(source, output, config(), resume=True)
    with pytest.raises(ValueError, match="only increase"):
        training.train_gui_owl(source, output, config(max_steps=1), resume=True)
    checkpoint = training.latest_checkpoint(output)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        training.train_gui_owl(source, output, config(max_steps=4), resume=True)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="No complete optimizer"):
        training.train_gui_owl(source, output, config(max_steps=4), resume=True)
    assert len(stack.models) == 1


def test_invalid_export_style_does_not_create_output(stack, tmp_path):
    source, output = make_export(tmp_path / "data"), tmp_path / "train"
    with pytest.raises(ValueError, match="reformat-seeclick"):
        training.train_gui_owl(source, output, config(), prompt_style="code")
    assert not output.exists() and not stack.models


def test_quantized_parameter_counts_use_peft_unpacked_size():
    model = SimpleNamespace(get_nb_trainable_parameters=lambda: (100, 2000))
    assert training.parameter_counts(model) == {
        "trainable_parameters": 100, "total_parameters_including_adapter": 2000,
    }


def test_all_size_configs_share_experiment_settings():
    configs = [GUIOwlTrainingConfig.model_validate_json(
        Path(f"configs/gui_owl_{size}_3090.json").read_text()
    ) for size in ("2b", "4b", "8b")]
    serialized = [config.model_dump() for config in configs]
    for item in serialized:
        assert item["model"]["model_revision"] != "main"
        item["model"].pop("model_id")
        item["model"].pop("model_revision")
    assert serialized[0] == serialized[1] == serialized[2]
