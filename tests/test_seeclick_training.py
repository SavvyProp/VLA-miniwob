"""Exercise resumable training orchestration without downloading model weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_seeclick_data import CharacterTokenizer, make_export

from scriptwob import seeclick_training as training
from scriptwob.seeclick_prompt import build_user_prompt
from scriptwob.seeclick_runtime import SeeClickModelConfig
from scriptwob.seeclick_training import SeeClickTrainingConfig


def config(**updates):
    return SeeClickTrainingConfig(
        model=SeeClickModelConfig(
            device="cpu", precision="fp32", quantization="none", gradient_checkpointing=False,
        ),
        max_length=1024, max_steps=2, gradient_accumulation_steps=1,
        save_steps=1, logging_steps=1,
    ).model_copy(update=updates)


def read_json(path):
    return json.loads(path.read_text())


class FakeModel:
    def parameters(self):
        return [SimpleNamespace(requires_grad=True, numel=lambda: 16),
                SimpleNamespace(requires_grad=False, numel=lambda: 100)]

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
        (path / "adapter_model.safetensors").write_bytes(b"fake adapter")


@pytest.fixture
def stack(monkeypatch):
    models, trainers, seeds = [], [], []

    def load_model(*args, **kwargs):
        models.append((args, kwargs))
        return FakeModel()

    class Trainer:
        def __init__(self, model, args, **kwargs):
            self.model, self.args = model, args
            self.state = SimpleNamespace(global_step=0)
            self.kwargs = kwargs
            self.resumed = None
            self.checkpoint_saves = []
            trainers.append(self)

        def train(self, resume_from_checkpoint=None):
            self.resumed = resume_from_checkpoint
            self.state.global_step = self.args.max_steps
            if self.state.global_step % self.args.save_steps == 0:
                self._save_checkpoint(self.model, trial=None)
                for callback in self.kwargs["callbacks"]:
                    callback.on_save(self.args, self.state, None)
            return SimpleNamespace(metrics={"train_loss": 0.75})

        def _save_checkpoint(self, model, trial):
            self.checkpoint_saves.append(self.state.global_step)
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            model.save_pretrained(path)
            (path / "trainer_state.json").write_text(json.dumps({
                "global_step": self.state.global_step,
            }))
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (path / name).write_bytes(b"fake optimizer state")

    monkeypatch.setattr(training, "load_tokenizer", lambda _: CharacterTokenizer())
    monkeypatch.setattr(training, "load_model", load_model)
    monkeypatch.setattr(training, "_training_stack", lambda: (
        Trainer, object, lambda **kwargs: SimpleNamespace(**kwargs), seeds.append,
    ))
    monkeypatch.setattr(training, "runtime_versions", lambda: {"test_runtime": "pinned"})
    monkeypatch.setattr(training, "implementation_digest", lambda: "fixed-code-digest")
    return SimpleNamespace(models=models, trainers=trainers, seeds=seeds, trainer_type=Trainer)


def test_prepare_only_validates_data_without_loading_weights_then_resumes_training(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    result = training.train_seeclick(source, output, config(), prepare_only=True)
    assert result["status"] == "prepared"
    assert result["examples"] == 1 and result["truncated_examples"] == 0
    assert result["tokens_min"] == result["tokens_max"] > 0
    assert stack.models == []
    before = read_json(output / training.METADATA_FILE)
    assert before["model"]["model_id"] == "cckevinn/SeeClick"
    assert before["status"] == "prepared"
    result = training.train_seeclick(source, output, config(), resume=True)
    assert result["status"] == "complete"
    assert result["global_step"] == 2
    assert result["trainable_parameters"] == 16
    assert stack.models[0][1] == {"training": True}
    assert stack.trainers[0].resumed is None
    assert stack.seeds == [2026]
    adapter = output / "adapter"
    saved = read_json(adapter / training.METADATA_FILE)
    assert saved["adapter_hashes"] == training.checkpoint_hashes(adapter)
    assert saved["training_fingerprint"] == before["training_fingerprint"]
    checkpoint = training.latest_checkpoint(output)
    assert checkpoint.name == "checkpoint-2"
    assert read_json(checkpoint / training.METADATA_FILE)["adapter_hashes"]
    assert read_json(output / "train_results.json")["global_step"] == 2


def test_resume_increases_step_target_preserves_identity_and_restores_checkpoint(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config())
    before = read_json(output / training.METADATA_FILE)
    result = training.train_seeclick(source, output, config(max_steps=4), resume=True)
    after = read_json(output / training.METADATA_FILE)
    assert result["global_step"] == 4
    assert stack.trainers[-1].resumed == str(output / "checkpoints" / "checkpoint-2")
    assert after["training_fingerprint"] == before["training_fingerprint"]
    assert after["created_utc"] == before["created_utc"]
    assert after["limit_history"][-1]["old_max_steps"] == 2
    assert after["limit_history"][-1]["new_max_steps"] == 4


@pytest.mark.parametrize("mutation", ["dataset", "config", "runtime", "source", "code"])
def test_resume_rejects_changed_identity_before_loading_model(
    stack, tmp_path, monkeypatch, mutation,
):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config(), prepare_only=True)
    before = (output / training.METADATA_FILE).read_text()
    settings = config()
    if mutation == "dataset":
        data = read_json(source / "train.json")
        data[0]["id"] = "changed-id"
        (source / "train.json").write_text(json.dumps(data))
    elif mutation == "config":
        settings = config(learning_rate=2e-4)
    elif mutation == "runtime":
        monkeypatch.setattr(training, "runtime_versions", lambda: {"test_runtime": "changed"})
    elif mutation == "source":
        source = make_export(tmp_path / "other-data")
    elif mutation == "code":
        monkeypatch.setattr(training, "implementation_digest", lambda: "changed-code")
    with pytest.raises(ValueError, match="differs"):
        training.train_seeclick(source, output, settings, resume=True)
    assert stack.models == []
    assert (output / training.METADATA_FILE).read_text() == before


@pytest.mark.parametrize("new_limit", [1, -1])
def test_resume_rejects_smaller_or_unbounded_target(stack, tmp_path, new_limit):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config(), prepare_only=True)
    with pytest.raises(ValueError, match="only increase"):
        training.train_seeclick(source, output, config(max_steps=new_limit), resume=True)
    assert stack.models == []


def test_reached_target_does_not_accidentally_train_extra_steps(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config())
    with pytest.raises(ValueError, match="already reached"):
        training.train_seeclick(source, output, config(), resume=True)
    assert len(stack.models) == 1


def test_tampered_adapter_cannot_resume(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config())
    checkpoint = training.latest_checkpoint(output)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"corrupt adapter")
    with pytest.raises(ValueError, match="integrity"):
        training.train_seeclick(source, output, config(max_steps=4), resume=True)
    assert len(stack.models) == 1


def test_partial_newer_checkpoint_is_ignored_and_complete_one_restored(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config())
    partial = output / "checkpoints" / "checkpoint-3"
    partial.mkdir()
    (partial / "trainer_state.json").write_text('{"global_step":3}')
    assert training.latest_checkpoint(output).name == "checkpoint-2"
    training.train_seeclick(source, output, config(max_steps=4), resume=True)
    assert stack.trainers[-1].resumed.endswith("checkpoint-2")


def test_completed_run_missing_optimizer_checkpoint_cannot_claim_resume(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config())
    checkpoint = training.latest_checkpoint(output)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="No complete optimizer checkpoint"):
        training.train_seeclick(source, output, config(max_steps=4), resume=True)
    assert len(stack.models) == 1


def test_collection_source_exports_once_and_resume_keeps_snapshot(stack, tmp_path, monkeypatch):
    source = tmp_path / "collection"
    source.mkdir()
    (source / "run.json").write_text("{}")
    output = tmp_path / "training"
    calls = []

    def export(source_path, destination, **kwargs):
        calls.append((source_path, destination, kwargs))
        make_export(destination)

    monkeypatch.setattr(training, "export_dataset", export)
    training.train_seeclick(source, output, config(), prepare_only=True)
    training.train_seeclick(source, output, config(), resume=True)
    assert len(calls) == 1
    assert calls[0][2] == {
        "format": "seeclick", "allow_oracle": False, "prompt_style": "natural",
    }
    assert read_json(output / training.METADATA_FILE)["dataset_path"] == str(output / "dataset")


def test_code_collection_export_records_style_and_resume_inherits_it(stack, tmp_path):
    from test_script_export import fixture_run

    source, _ = fixture_run(tmp_path)
    output = tmp_path / "training"
    training.train_seeclick(
        source, output, config(max_length=2048), prepare_only=True, prompt_style="code",
    )
    metadata_path = output / training.METADATA_FILE
    before = metadata_path.read_bytes()
    metadata = read_json(metadata_path)
    assert metadata["prompt_style"] == "code"
    assert metadata["dataset"]["metadata"]["prompt_style"] == "code"
    assert metadata["identity"]["instruction_version"] == "scriptwob-code-v1"
    assert stack.models == []
    with pytest.raises(ValueError, match="Frozen dataset prompt style differs"):
        training.train_seeclick(
            source, output, config(max_length=2048), resume=True, prompt_style="natural",
        )
    assert metadata_path.read_bytes() == before
    training.train_seeclick(source, output, config(max_length=2048), resume=True)
    assert read_json(output / "adapter" / training.METADATA_FILE)["prompt_style"] == "code"


def test_export_style_override_requires_reformat_before_creating_training_output(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="reformat-seeclick"):
        training.train_seeclick(source, output, config(), prompt_style="code")
    assert not output.exists()
    assert stack.models == []


def test_training_records_failed_preflight_without_loading_weights(stack, tmp_path):
    source = make_export(tmp_path / "data")
    data = read_json(source / "train.json")
    data[0]["conversations"][0]["value"] = build_user_prompt(
        "Very long instruction. " * 200, (), source / "images" / "step.png",
    )
    (source / "train.json").write_text(json.dumps(data))
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="never truncated"):
        training.train_seeclick(source, output, config(), prepare_only=True)
    assert stack.models == []
    saved = read_json(output / training.METADATA_FILE)
    assert saved["status"] == "failed" and saved["error_type"] == "ValueError"


def test_preflight_reserves_generation_budget_even_when_gold_answer_fits(stack, tmp_path):
    source = make_export(tmp_path / "data")
    data = read_json(source / "train.json")
    data[0]["conversations"][0]["value"] = build_user_prompt(
        "Click red. " * 30, (), source / "images" / "step.png",
    )
    (source / "train.json").write_text(json.dumps(data))
    dataset = training.SeeClickDataset(
        training.load_training_export(source), CharacterTokenizer(), 4096,
    )
    sample = dataset.samples[0]
    prompt_length = next(index for index, label in enumerate(sample["labels"]) if label != -100)
    insufficient_limit = prompt_length + 63
    assert 512 <= len(sample["input_ids"]) <= insufficient_limit
    with pytest.raises(ValueError, match="needs 64 output tokens"):
        training.train_seeclick(
            source, tmp_path / "insufficient", config(max_length=insufficient_limit),
            prepare_only=True,
        )
    assert stack.models == []
    exact = training.train_seeclick(
        source, tmp_path / "exact", config(max_length=prompt_length + 64), prepare_only=True,
    )
    assert exact["prompt_tokens_max"] == prompt_length
    assert exact["reserved_output_tokens"] == 64
    assert stack.models == []


def test_oracle_data_requires_explicit_training_flag(stack, tmp_path):
    source = make_export(tmp_path / "data", supervision="privileged_oracle")
    with pytest.raises(ValueError, match="allow-oracle"):
        training.train_seeclick(source, tmp_path / "rejected", config(), prepare_only=True)
    result = training.train_seeclick(
        source, tmp_path / "allowed", config(), prepare_only=True, allow_oracle=True,
    )
    assert result["examples"] == 1
    metadata = read_json(tmp_path / "allowed" / training.METADATA_FILE)
    assert metadata["identity"]["allow_oracle"] is True
    assert metadata["dataset"]["metadata"]["supervision"] == "privileged_oracle"


def test_output_safety_and_existing_run_require_explicit_resume(stack, tmp_path):
    source = make_export(tmp_path / "data")
    with pytest.raises(ValueError, match="must not contain"):
        training.train_seeclick(source, tmp_path, config(), prepare_only=True)
    with pytest.raises(ValueError, match="must not contain"):
        training.train_seeclick(source, source, config(), prepare_only=True)
    with pytest.raises(ValueError, match="No training metadata"):
        training.train_seeclick(source, tmp_path / "unknown", config(), resume=True)
    output = tmp_path / "training"
    training.train_seeclick(source, output, config(), prepare_only=True)
    with pytest.raises(ValueError, match="already exists"):
        training.train_seeclick(source, output, config())


def test_training_config_rejects_zero_step_target():
    with pytest.raises(ValueError, match="max_steps"):
        SeeClickTrainingConfig(max_steps=0)


def test_invalid_source_does_not_create_output_and_can_be_fixed_and_retried(stack, tmp_path):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    metadata = read_json(source / "dataset.json")
    (source / "dataset.json").write_text(json.dumps({**metadata, "examples": 99}))
    with pytest.raises(ValueError, match="count"):
        training.train_seeclick(source, output, config(), prepare_only=True)
    assert not output.exists()
    (source / "dataset.json").write_text(json.dumps(metadata))
    result = training.train_seeclick(source, output, config(), prepare_only=True)
    assert result["status"] == "prepared"


def test_collection_export_failure_can_retry_a_directory_containing_only_lock(
    stack, tmp_path, monkeypatch,
):
    source = tmp_path / "collection"
    source.mkdir()
    (source / "run.json").write_text("{}")
    output = tmp_path / "training"

    def failing_export(*args, **kwargs):
        raise ValueError("No accepted demonstration yet")

    monkeypatch.setattr(training, "export_dataset", failing_export)
    with pytest.raises(ValueError, match="No accepted"):
        training.train_seeclick(source, output, config(), prepare_only=True)
    assert [path.name for path in output.iterdir()] == [".writer.lock"]
    monkeypatch.setattr(training, "export_dataset", lambda src, dst, **kwargs: make_export(dst))
    result = training.train_seeclick(source, output, config(), prepare_only=True)
    assert result["status"] == "prepared"


def test_tokenizer_download_failure_can_resume_before_optimization_started(
    stack, tmp_path, monkeypatch,
):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"

    def failed_tokenizer(_):
        raise OSError("Temporary tokenizer download failure")

    monkeypatch.setattr(training, "load_tokenizer", failed_tokenizer)
    with pytest.raises(OSError, match="Temporary tokenizer"):
        training.train_seeclick(source, output, config())
    saved = read_json(output / training.METADATA_FILE)
    assert saved["status"] == "failed"
    assert saved["optimization_started"] is False
    assert training.latest_checkpoint(output) is None
    monkeypatch.setattr(training, "load_tokenizer", lambda _: CharacterTokenizer())
    assert training.train_seeclick(source, output, config(), resume=True)["status"] == "complete"
    assert stack.trainers[-1].resumed is None


def test_prepare_retry_cannot_erase_prior_optimization_and_restart_without_checkpoint(
    stack, tmp_path, monkeypatch,
):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    original_train = stack.trainer_type.train

    def interrupted_train(self, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(stack.trainer_type, "train", interrupted_train)
    with pytest.raises(KeyboardInterrupt):
        training.train_seeclick(source, output, config())
    before = read_json(output / training.METADATA_FILE)
    assert before["status"] == "interrupted"
    assert before["optimization_started"] is True
    assert training.latest_checkpoint(output) is None
    prepared = training.train_seeclick(source, output, config(), resume=True, prepare_only=True)
    assert prepared["status"] == "prepared"
    assert read_json(output / training.METADATA_FILE)["optimization_started"] is True
    monkeypatch.setattr(stack.trainer_type, "train", original_train)
    with pytest.raises(ValueError, match="No complete optimizer checkpoint"):
        training.train_seeclick(source, output, config(), resume=True)
    assert len(stack.models) == 1


@pytest.mark.parametrize("save_steps", [1, 100])
def test_final_resumable_checkpoint_is_saved_once_even_when_periodic_save_matches(
    stack, tmp_path, save_steps,
):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    training.train_seeclick(source, output, config(save_steps=save_steps))
    assert stack.trainers[-1].checkpoint_saves == [2]
    assert training.latest_checkpoint(output).name == "checkpoint-2"


def test_failure_to_record_error_metadata_preserves_original_error(stack, tmp_path, monkeypatch):
    source = make_export(tmp_path / "data")
    original_write = training.write_json
    original_error = RuntimeError("Original tokenizer error")

    def failed_tokenizer(_):
        raise original_error

    def failed_error_metadata(path, value):
        if value.get("status") == "failed":
            raise OSError("Disk full while writing failure metadata")
        original_write(path, value)

    monkeypatch.setattr(training, "load_tokenizer", failed_tokenizer)
    monkeypatch.setattr(training, "write_json", failed_error_metadata)
    with pytest.raises(RuntimeError, match="Original tokenizer") as raised:
        training.train_seeclick(source, tmp_path / "training", config())
    assert raised.value is original_error
    assert "Could not update training metadata: OSError" in raised.value.__notes__


def test_training_requires_visible_gpu_zero_without_restricting_evaluation_config():
    model = SeeClickModelConfig(device="cuda:1")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        SeeClickTrainingConfig(model=model)


@pytest.mark.parametrize("world_size", ["0", "2"])
def test_training_rejects_distributed_launch_before_creating_outputs(
    stack, tmp_path, monkeypatch, world_size,
):
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    monkeypatch.setenv("WORLD_SIZE", world_size)
    with pytest.raises(ValueError, match="one process and one GPU"):
        training.train_seeclick(source, output, config())
    assert not output.exists()
    assert stack.models == []


def test_native_cpu_trainer_lora_save_reload_and_resume(tmp_path, monkeypatch):
    """Real optimization/checkpoint smoke in the optional runtime; no remote model.

    This verifies Trainer/PEFT persistence, not SeeClick's visual grounding quality
    or the CUDA/4-bit kernels. The tokenizer and tiny random language model are local.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    torch.set_num_threads(1)
    models = []

    def base_model():
        # Recreate the same frozen pretrained stand-in for adapter reload checks.
        torch.manual_seed(42)
        return transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=2048, n_positions=1024, n_embd=8, n_layer=1, n_head=1,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, use_cache=False,
        ))

    def load_tiny_model(*args, **kwargs):
        model = peft.get_peft_model(base_model(), peft.LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0, target_modules=["c_attn"],
            task_type="CAUSAL_LM", bias="none",
        ))
        models.append(model)
        return model

    monkeypatch.setattr(training, "load_tokenizer", lambda _: CharacterTokenizer())
    monkeypatch.setattr(training, "load_model", load_tiny_model)
    monkeypatch.setattr(training, "implementation_digest", lambda: "native-test-fixed-code")
    source = make_export(tmp_path / "data")
    output = tmp_path / "training"
    result = training.train_seeclick(source, output, config(learning_rate=0.01))
    assert result["global_step"] == 2
    checkpoint = training.latest_checkpoint(output)
    assert checkpoint.name == "checkpoint-2"
    assert (checkpoint / "optimizer.pt").stat().st_size > 0
    trained = models[-1]
    trained.eval()
    assert any(torch.count_nonzero(parameter).item() > 0
               for name, parameter in trained.named_parameters() if "lora_B" in name)
    loaded = peft.PeftModel.from_pretrained(base_model(), output / "adapter")
    loaded.eval()
    batch = {"input_ids": torch.tensor([[1000, 1001, 1002, 1003]])}
    with torch.inference_mode():
        assert torch.allclose(trained(**batch).logits, loaded(**batch).logits)
    result = training.train_seeclick(
        source, output, config(max_steps=3, learning_rate=0.01), resume=True,
    )
    assert result["global_step"] == 3
    assert read_json(training.latest_checkpoint(output) / "trainer_state.json")["global_step"] == 3
    # Optimizer and scheduler restore must be possible; at the old cosine endpoint
    # the first resumed LR can be zero, so do not assert weights changed immediately.
    assert all(torch.isfinite(parameter).all() for parameter in models[-1].parameters())
