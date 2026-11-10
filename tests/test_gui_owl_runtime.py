"""Loader/device isolation and adapter provenance checks without model downloads."""

import json
import re
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from scriptwob import gui_owl_runtime as runtime


def write_adapter(path, config=None):
    path.mkdir(parents=True)
    config = config or runtime.GUIOwlModelConfig()
    (path / "adapter_model.safetensors").write_bytes(b"test mock adapter")
    (path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": config.model_id,
        "revision": config.model_revision, "target_modules": runtime.LORA_TARGETS,
        "modules_to_save": None, "r": config.lora_rank, "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout, "bias": "none", "task_type": "CAUSAL_LM",
    }))
    metadata = {
        "schema_version": runtime.TRAINING_SCHEMA, "model": config.model_dump(),
        "max_length": 8192, "history_images": 3, "adapter_hashes": runtime.adapter_hashes(path),
    }
    (path / runtime.TRAINING_METADATA).write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize("size", ["2b", "4b", "8b"])
def test_each_size_resolves_to_its_own_pinned_model_and_processor(size):
    config, path, metadata = runtime.resolve_checkpoint(f"base-{size}")
    assert (config.model_id, config.model_revision) == runtime.GUI_OWL_MODELS[size]
    assert re.fullmatch(r"[0-9a-f]{40}", config.model_revision)
    selected = runtime.GUIOwlModelConfig(model_id=config.model_id)
    assert selected.model_revision == config.model_revision
    assert path is None and metadata == {}


@pytest.mark.parametrize("changes", [
    {"model_revision": "main"}, {"model_id": "Qwen/Qwen3-VL-2B-Instruct"},
    {"device": "cpu"}, {"device": "cpu", "quantization": "none"},
    {"precision": "fp32"}, {"device": "auto"}, {"lora_rank": 0},
    {"lora_dropout": 1}, {"min_pixels": 500000, "max_pixels": 262144},
])
def test_configuration_rejects_unsafe_or_incompatible_settings(changes):
    with pytest.raises(ValidationError):
        runtime.GUIOwlModelConfig(**changes)


def test_lora_scope_excludes_vision_embeddings_and_head():
    for layer in [0, 1, 32]:
        for kind, names in {
            "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
        }.items():
            for name in names:
                assert re.fullmatch(
                    runtime.LORA_TARGETS, f"model.language_model.layers.{layer}.{kind}.{name}"
                )
    for name in [
        "model.visual.blocks.0.attn.q_proj", "model.visual.blocks.0.mlp.up_proj",
        "model.language_model.embed_tokens", "lm_head",
    ]:
        assert not re.fullmatch(runtime.LORA_TARGETS, name)


def test_completed_and_explicit_intermediate_checkpoints_resolve(tmp_path):
    expected = write_adapter(tmp_path / "run" / "adapter")
    config, adapter, metadata = runtime.resolve_checkpoint(tmp_path / "run", precision="fp16")
    assert adapter == tmp_path / "run" / "adapter"
    assert config.precision == "fp16"
    assert metadata == expected and metadata["model"]["precision"] == "bf16"
    intermediate = tmp_path / "other" / "checkpoints" / "checkpoint-100"
    write_adapter(intermediate)
    with pytest.raises(ValueError, match="explicit"):
        runtime.resolve_checkpoint(tmp_path / "other")
    assert runtime.resolve_checkpoint(intermediate)[1] == intermediate


@pytest.mark.parametrize("artifact", runtime.ADAPTER_FILES)
def test_modified_artifacts_are_rejected_before_loading(tmp_path, artifact):
    path = tmp_path / "adapter"
    write_adapter(path)
    with (path / artifact).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        runtime.resolve_checkpoint(path)


@pytest.mark.parametrize("change", ["missing_metadata", "wrong_schema", "partial_hashes", "pickle"])
def test_partial_or_foreign_checkpoint_is_rejected(tmp_path, change):
    path = tmp_path / "adapter"
    metadata = write_adapter(path)
    if change == "missing_metadata":
        (path / runtime.TRAINING_METADATA).unlink()
    elif change == "wrong_schema":
        metadata["schema_version"] = "scriptwob-seeclick-training-v1"
    elif change == "partial_hashes":
        metadata["adapter_hashes"].pop("adapter_model.safetensors")
    else:
        (path / "adapter_model.safetensors").rename(path / "adapter_model.bin")
    if change in {"wrong_schema", "partial_hashes"}:
        (path / runtime.TRAINING_METADATA).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        runtime.resolve_checkpoint(path)


@pytest.mark.parametrize("field,value", [
    ("revision", "f" * 40), ("target_modules", ["q_proj"]),
    ("base_model_name_or_path", "cckevinn/SeeClick"), ("r", 32), ("bias", "all"),
])
def test_even_rehashed_incompatible_adapters_are_rejected(tmp_path, field, value):
    path = tmp_path / "adapter"
    metadata = write_adapter(path)
    data = json.loads((path / "adapter_config.json").read_text())
    data[field] = value
    (path / "adapter_config.json").write_text(json.dumps(data))
    metadata["adapter_hashes"] = runtime.adapter_hashes(path)
    (path / runtime.TRAINING_METADATA).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs|language-only"):
        runtime.resolve_checkpoint(path)


class FakeParameter:
    def __init__(self):
        self.requires_grad = True
        self.dtype = "fp32"
        self.data = self

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def to(self, *, dtype):
        self.dtype = dtype
        return self


class FakeModel:
    def __init__(self):
        self.visual = FakeParameter()
        self.embedding = FakeParameter()
        self.head = FakeParameter()
        self.language = FakeParameter()
        self.model = SimpleNamespace(visual=self.visual)
        self.config = SimpleNamespace(use_cache=True, text_config=SimpleNamespace(use_cache=True))
        self.all_parameters = {
            "model.visual.weight": self.visual,
            "model.language_model.embed_tokens.weight": self.embedding,
            "lm_head.weight": self.head,
            "model.language_model.layers.0.self_attn.q_proj.weight": self.language,
        }
        self.mode = None
        self.checkpointing = None

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def requires_grad_(self, value):
        for parameter in self.parameters():
            parameter.requires_grad = value
        return self

    def named_parameters(self):
        return self.all_parameters.items()

    def parameters(self):
        return self.all_parameters.values()

    def named_modules(self):
        return [("model.language_model.layers.0.self_attn.q_proj", self.language)]

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = kwargs

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


@pytest.fixture
def fake_dependencies(monkeypatch):
    model, calls = FakeModel(), {}
    torch = SimpleNamespace(
        bfloat16="bf16", float16="fp16", float32="fp32",
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            device=lambda _: nullcontext(), is_bf16_supported=lambda: True,
        ),
    )

    def from_pretrained(identifier, **kwargs):
        calls.update(model_id=identifier, loader=kwargs)
        return model

    def quantization(**kwargs):
        calls["quantization"] = kwargs
        return kwargs

    def prepare(value, **kwargs):
        calls["prepare"] = kwargs
        value.requires_grad_(False)
        for parameter in value.parameters():
            parameter.dtype = "fp32"
        return value

    def add_lora(value, config):
        calls["lora"] = config
        value.requires_grad_(False)
        value.all_parameters[
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        ] = FakeParameter()
        return value

    def load_lora(value, path, *, is_trainable):
        calls["adapter"] = (path, is_trainable)
        return add_lora(value, {}) if is_trainable else value

    transformers = SimpleNamespace(
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=from_pretrained),
        BitsAndBytesConfig=quantization,
    )
    peft = SimpleNamespace(
        prepare_model_for_kbit_training=prepare, LoraConfig=lambda **kwargs: kwargs,
        get_peft_model=add_lora, PeftModel=SimpleNamespace(from_pretrained=load_lora),
    )
    monkeypatch.setattr(runtime, "_dependencies", lambda: (torch, transformers, peft))
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))
    return model, calls, torch


def test_nf4_training_freezes_vision_uses_sdpa_and_nonreentrant_checkpointing(fake_dependencies):
    model, calls, _ = fake_dependencies
    config = runtime.GUIOwlModelConfig()
    assert runtime.load_model(config, training=True) is model
    assert calls["loader"]["revision"] == config.model_revision
    assert calls["loader"]["trust_remote_code"] is False
    assert calls["loader"]["use_safetensors"] is True
    assert calls["loader"]["attn_implementation"] == "sdpa"
    assert calls["loader"]["device_map"] == {"": 0}
    assert calls["quantization"]["llm_int8_skip_modules"] == ["model.visual", "lm_head"]
    assert calls["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert calls["prepare"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert calls["lora"]["target_modules"] == runtime.LORA_TARGETS
    assert model.visual.dtype == model.embedding.dtype == model.head.dtype == "bf16"
    assert not model.visual.requires_grad
    assert not model.config.use_cache and not model.config.text_config.use_cache
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 1 and trainable[0].dtype == "fp32"
    assert model.mode == "train"


def test_base_inference_does_not_attach_random_lora(fake_dependencies):
    model, calls, _ = fake_dependencies
    runtime.load_model(runtime.GUIOwlModelConfig())
    assert "lora" not in calls and "prepare" not in calls
    assert model.config.use_cache and model.mode == "eval"


@pytest.mark.parametrize("device,index", [("cuda", 0), ("cuda:0", 0), ("cuda:2", 2)])
def test_cuda_map_has_integer_index(fake_dependencies, device, index):
    _, calls, torch = fake_dependencies
    torch.cuda.device_count = lambda: 3
    runtime.load_model(runtime.GUIOwlModelConfig(device=device), training=True)
    assert type(calls["loader"]["device_map"][""]) is int
    assert calls["loader"]["device_map"][""] == index


def test_unquantized_cpu_training_also_freezes_base_and_uses_nonreentrant(fake_dependencies):
    model, calls, _ = fake_dependencies
    config = runtime.GUIOwlModelConfig(device="cpu", quantization="none", precision="fp32")
    runtime.load_model(config, training=True)
    assert "quantization" not in calls
    assert model.checkpointing == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert not model.visual.requires_grad


@pytest.mark.parametrize("training", [False, True])
def test_saved_adapter_attached_with_intended_trainability(fake_dependencies, tmp_path, training):
    _, calls, _ = fake_dependencies
    path = tmp_path / "adapter"
    write_adapter(path)
    runtime.load_model(runtime.GUIOwlModelConfig(), training=training, adapter_path=path)
    assert calls["adapter"] == (str(path), training)


def test_processor_shares_model_revision_and_resolution(monkeypatch):
    calls = {}
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=42))

    def load(identifier, **kwargs):
        calls.update(identifier=identifier, **kwargs)
        return processor

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load)
    ))
    config = runtime.GUIOwlModelConfig(model_id=runtime.GUI_OWL_MODELS["8b"][0])
    assert runtime.load_processor(config) is processor
    assert calls["identifier"] == config.model_id
    assert calls["revision"] == config.model_revision
    assert calls["trust_remote_code"] is False
    assert calls["min_pixels"] == config.min_pixels and calls["max_pixels"] == config.max_pixels
    assert processor.tokenizer.pad_token_id == 42
    assert processor.tokenizer.padding_side == "right"


@pytest.mark.parametrize("failure", ["cuda_missing", "device_missing", "bf16_missing"])
def test_hardware_failures_precede_weight_loading(fake_dependencies, failure):
    _, calls, torch = fake_dependencies
    if failure == "cuda_missing":
        torch.cuda.is_available = lambda: False
    elif failure == "device_missing":
        torch.cuda.device_count = lambda: 0
    else:
        torch.cuda.is_bf16_supported = lambda: False
    with pytest.raises(ValueError if failure != "cuda_missing" else RuntimeError):
        runtime.load_model(runtime.GUIOwlModelConfig())
    assert "loader" not in calls


def test_native_bnb_replacement_excludes_visual_subtree_and_output_head():
    """Exercise the real pinned converter's prefix matching without loading weights."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.6":
        pytest.skip("requires the isolated GUI-Owl runtime")
    bnb = pytest.importorskip("bitsandbytes")
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.model.language_model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.lm_head = torch.nn.Linear(4, 4)
    skip = ["model.visual", "lm_head"]
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=skip,
    )
    result = replace_with_bnb_linear(
        model, modules_to_not_convert=skip, quantization_config=quantization,
    )
    assert type(result.model.visual[0]) is torch.nn.Linear
    assert type(result.lm_head) is torch.nn.Linear
    assert isinstance(result.model.language_model[0], bnb.nn.Linear4bit)
