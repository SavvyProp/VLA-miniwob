import json
import os
import re
import shutil
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from scriptwob import seeclick_runtime as runtime


def write_adapter(path, config=None):
    path.mkdir(parents=True)
    config = config or runtime.SeeClickModelConfig()
    (path / "adapter_model.safetensors").write_bytes(b"test mock adapter")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": config.model_id,
                "revision": config.model_revision,
                "target_modules": runtime.LORA_TARGETS,
                "modules_to_save": None,
            }
        )
    )
    metadata = {"model": config.model_dump(), "max_length": 2048}
    (path / runtime.TRAINING_METADATA).write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize(
    "changes",
    [
        {"model_revision": "main"},
        {"tokenizer_revision": "latest"},
        {"device": "cpu"},
        {"device": "cpu", "quantization": "none"},
        {"precision": "fp32"},
        {"device": "auto"},
        {"lora_rank": 0},
        {"lora_dropout": 1},
    ],
)
def test_configuration_rejects_unpinned_or_invalid_settings(changes):
    with pytest.raises(ValidationError):
        runtime.SeeClickModelConfig(**changes)


def test_lora_regex_excludes_vision_encoder():
    for name in (
        "transformer.h.0.attn.c_attn",
        "transformer.h.31.attn.c_proj",
        "transformer.h.3.mlp.w1",
        "transformer.h.3.mlp.w2",
        "transformer.h.3.mlp.c_proj",
    ):
        assert re.fullmatch(runtime.LORA_TARGETS, name)
    for name in ("transformer.visual.transformer.resblocks.0.mlp.c_proj", "lm_head"):
        assert not re.fullmatch(runtime.LORA_TARGETS, name)


def test_checkpoint_resolves_completed_run_and_preserves_metadata(tmp_path):
    expected = write_adapter(tmp_path / "run" / "adapter")
    config, adapter, metadata = runtime.resolve_checkpoint(tmp_path / "run", precision="fp16")
    assert adapter == tmp_path / "run" / "adapter"
    assert config.precision == "fp16"
    assert metadata == expected
    assert metadata["model"]["precision"] == "bf16"


def test_checkpoint_requires_explicit_intermediate_path(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint-100"
    write_adapter(checkpoint)
    with pytest.raises(ValueError, match="explicit"):
        runtime.resolve_checkpoint(tmp_path / "run")
    assert runtime.resolve_checkpoint(checkpoint)[1] == checkpoint


@pytest.mark.parametrize("field,value", [("revision", "f" * 40), ("target_modules", ["c_proj"])])
def test_adapter_architecture_and_revision_are_validated(tmp_path, field, value):
    path = tmp_path / "adapter"
    write_adapter(path)
    config_path = path / "adapter_config.json"
    data = json.loads(config_path.read_text())
    data[field] = value
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runtime.resolve_checkpoint(path)


def test_adapter_without_own_provenance_is_rejected(tmp_path):
    path = tmp_path / "adapter"
    write_adapter(path)
    (path / runtime.TRAINING_METADATA).unlink()
    (tmp_path / runtime.TRAINING_METADATA).write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        runtime.resolve_checkpoint(path)


def test_pickle_adapter_is_not_loaded(tmp_path):
    path = tmp_path / "adapter"
    write_adapter(path)
    (path / "adapter_model.safetensors").rename(path / "adapter_model.bin")
    with pytest.raises(ValueError, match="safe LoRA"):
        runtime.resolve_checkpoint(path)


class FakeModule:
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
        self.visual = FakeModule()
        self.embedding = FakeModule()
        self.head = FakeModule()
        self.language = FakeModule()
        self.transformer = SimpleNamespace(visual=self.visual)
        self.config = SimpleNamespace(use_cache=True)
        self.all_parameters = {
            "transformer.visual.weight": self.visual,
            "transformer.wte.weight": self.embedding,
            "lm_head.weight": self.head,
            "transformer.h.0.attn.c_attn.weight": self.language,
        }
        self.mode = None
        self.checkpointing = False
        self.input_grads = False

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
        return [("transformer.h.0.attn.c_attn", self.language)]

    def enable_input_require_grads(self):
        self.input_grads = True

    def gradient_checkpointing_enable(self):
        self.checkpointing = True

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


@pytest.fixture
def fake_dependencies(monkeypatch):
    model = FakeModel()
    calls = {}
    torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            device=lambda _: nullcontext(),
            is_bf16_supported=lambda: True,
        ),
    )

    def from_pretrained(identifier, **kwargs):
        calls["model_id"] = identifier
        calls["loader"] = kwargs
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
        value.all_parameters["base_model.model.transformer.h.0.attn.c_attn.lora_A.weight"] = (
            FakeModule()
        )
        return value

    def load_lora(value, path, *, is_trainable):
        calls["adapter"] = (path, is_trainable)
        if is_trainable:
            return add_lora(value, {})
        return value

    transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
        BitsAndBytesConfig=quantization,
    )
    peft = SimpleNamespace(
        prepare_model_for_kbit_training=prepare,
        LoraConfig=lambda **kwargs: kwargs,
        get_peft_model=add_lora,
        PeftModel=SimpleNamespace(from_pretrained=load_lora),
    )
    monkeypatch.setattr(runtime, "_dependencies", lambda: (torch, transformers, peft))
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))
    return model, calls, torch


def test_4bit_training_restores_frozen_vision_precision_and_limits_lora(fake_dependencies):
    model, calls, _ = fake_dependencies
    loaded = runtime.load_model(runtime.SeeClickModelConfig(), training=True)
    assert loaded is model
    assert calls["loader"]["revision"] == runtime.SEECLICK_REVISION
    assert calls["loader"]["code_revision"] == runtime.SEECLICK_REVISION
    assert calls["loader"]["use_safetensors"] is True
    assert calls["loader"]["use_flash_attn"] is False
    assert calls["loader"]["device_map"] == {"": 0}
    assert calls["quantization"]["llm_int8_skip_modules"] == ["transformer.visual", "lm_head"]
    assert calls["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert calls["prepare"]["use_gradient_checkpointing"] is True
    assert calls["lora"]["target_modules"] == runtime.LORA_TARGETS
    assert model.visual.dtype == model.embedding.dtype == model.head.dtype == "bf16"
    assert model.visual.requires_grad is False
    assert not model.config.use_cache
    assert model.mode == "train"


def test_baseline_inference_does_not_create_random_adapter(fake_dependencies):
    model, calls, _ = fake_dependencies
    runtime.load_model(runtime.SeeClickModelConfig())
    assert "lora" not in calls and "prepare" not in calls
    assert model.config.use_cache is True
    assert model.mode == "eval"


@pytest.mark.parametrize(
    "configured_device,current_device,expected_index",
    [("cuda", 2, 0), ("cuda:0", 2, 0), ("cuda:2", 0, 2)],
)
@pytest.mark.parametrize("training", [False, True])
def test_cuda_loader_uses_explicit_index_even_when_another_gpu_is_current(
    fake_dependencies, configured_device, current_device, expected_index, training
):
    _, calls, torch = fake_dependencies
    torch.cuda.device_count = lambda: 3
    torch.cuda.current_device = lambda: current_device
    runtime.load_model(runtime.SeeClickModelConfig(device=configured_device), training=training)
    # Accelerate 0.27.2 distinguishes the string "cuda:0" from integer 0 when its
    # training device is the unindexed CUDA alias. Always pass the integer form.
    device = calls["loader"]["device_map"][""]
    assert type(device) is int
    assert device == expected_index


def test_cpu_lora_training_keeps_base_frozen_and_enables_checkpoint_inputs(fake_dependencies):
    model, calls, _ = fake_dependencies
    config = runtime.SeeClickModelConfig(device="cpu", quantization="none", precision="fp32")
    runtime.load_model(config, training=True)
    assert "quantization" not in calls
    assert calls["loader"]["device_map"] == {"": "cpu"}
    assert model.input_grads and model.checkpointing
    assert not model.visual.requires_grad


def test_pinned_accelerate_accepts_quantized_loader_map_on_unindexed_cuda(
    monkeypatch, request
):
    """Exercise Accelerate's real device check on a CPU host, with no model download."""
    torch = pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    if accelerate.__version__ != "0.27.2":
        pytest.skip("regression for the isolated runtime's pinned Accelerate 0.27.2")
    import accelerate.accelerator as accelerator_module
    from accelerate.utils import DistributedType, DynamoBackend

    _, calls, _ = request.getfixturevalue("fake_dependencies")
    runtime.load_model(runtime.SeeClickModelConfig(), training=True)

    # torch.device(int) queries a real accelerator. Resolve just that constructor
    # on this CPU host; all equality checks and prepare_model logic remain real.
    class DeviceMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, torch.device)

    class Device(metaclass=DeviceMeta):
        def __new__(cls, value):
            return torch.device("cuda", value) if isinstance(value, int) else torch.device(value)

    torch_proxy = SimpleNamespace(**vars(torch))
    torch_proxy.device = Device
    monkeypatch.setattr(accelerator_module, "torch", torch_proxy)

    # Construct only the preparation state, avoiding shared Accelerate state or
    # CUDA initialization. The tiny CPU module is never moved or quantized.
    accelerator = object.__new__(accelerate.Accelerator)
    accelerator.device_placement = False
    accelerator.native_amp = False
    accelerator._models = []
    accelerator.state = SimpleNamespace(
        device=torch.device("cuda"),
        distributed_type=DistributedType.NO,
        mixed_precision="no",
        dynamo_plugin=SimpleNamespace(backend=DynamoBackend.NO),
    )
    model = torch.nn.Linear(1, 1)
    model.is_loaded_in_4bit = True
    model.hf_device_map = {"": "cuda:0"}
    with pytest.raises(ValueError, match="different device"):
        accelerator.prepare_model(model)

    model.hf_device_map = calls["loader"]["device_map"]
    assert accelerator.prepare_model(model) is model

    # The corrected map still rejects a genuinely different training GPU.
    accelerator.state.device = torch.device("cuda:1")
    with pytest.raises(ValueError, match="different device"):
        accelerator.prepare_model(model)


def test_existing_adapter_loaded_trainable_for_training(tmp_path, fake_dependencies):
    _, calls, _ = fake_dependencies
    path = tmp_path / "adapter"
    write_adapter(path)
    runtime.load_model(runtime.SeeClickModelConfig(), training=True, adapter_path=path)
    assert calls["adapter"] == (str(path), True)


def test_missing_gpu_fails_before_checkpoint_download(fake_dependencies):
    _, calls, torch = fake_dependencies
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="cannot access a GPU"):
        runtime.load_model(runtime.SeeClickModelConfig())
    assert "loader" not in calls


def test_tokenizer_pins_font_and_code_without_changing_lookup_afterward(monkeypatch):
    utils = ModuleType("transformers.utils")
    original = lambda *args, **kwargs: "other-cache-result"  # noqa: E731
    utils.try_to_load_from_cache = original
    transformer_module = ModuleType("transformers")
    transformer_module.utils = utils
    tokenizer = SimpleNamespace(eod_id=151643, pad_token_id=None)
    calls = {}

    def from_pretrained(identifier, **kwargs):
        calls["identifier"] = identifier
        calls["kwargs"] = kwargs
        assert utils.try_to_load_from_cache(runtime.QWEN_TOKENIZER, "SimSun.ttf") == "pinned.ttf"
        assert utils.try_to_load_from_cache("other", "other.txt") == "other-cache-result"
        return tokenizer

    transformer_module.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    hub_module = ModuleType("huggingface_hub")

    def download(identifier, filename, **kwargs):
        assert identifier == runtime.QWEN_TOKENIZER
        assert filename == "SimSun.ttf"
        assert kwargs["revision"] == runtime.QWEN_TOKENIZER_REVISION
        return "pinned.ttf"

    hub_module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "transformers", transformer_module)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    assert runtime.load_tokenizer(runtime.SeeClickModelConfig()) is tokenizer
    assert tokenizer.pad_token_id == 151643
    assert calls["kwargs"]["revision"] == calls["kwargs"]["code_revision"]
    assert utils.try_to_load_from_cache is original


@pytest.mark.skipif(
    os.environ.get("SCRIPTWOB_TEST_NATIVE_SEECLICK") != "1",
    reason="opt-in pinned remote-code smoke; downloads source files only, no pretrained weights",
)
def test_native_qwen_image_forward_lora_gradients_save_reload_and_generate(tmp_path):
    """Exercise the real pinned architecture with tiny randomly initialized weights.

    Run in the isolated CPU environment with SCRIPTWOB_TEST_NATIVE_SEECLICK=1.
    This verifies compatibility and gradient flow, not pretrained model accuracy or CUDA.
    """
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoConfig, AutoModelForCausalLM

    names = [
        "config.json",
        "configuration_qwen.py",
        "modeling_qwen.py",
        "qwen_generation_utils.py",
        "visual.py",
    ]
    source = snapshot_download(
        runtime.SEECLICK_MODEL, revision=runtime.SEECLICK_REVISION, allow_patterns=names
    )
    from pathlib import Path

    directory = tmp_path / "tiny-model"
    directory.mkdir()
    for name in names:
        shutil.copyfile(Path(source) / name, directory / name)
    values = json.loads((directory / "config.json").read_text())
    values.update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        kv_channels=32,
        vocab_size=512,
        bf16=False,
        fp16=False,
        fp32=True,
        max_position_embeddings=512,
        seq_length=512,
        use_cache=False,
    )
    values["visual"].update(
        heads=4,
        image_size=28,
        layers=1,
        width=32,
        patch_size=14,
        mlp_ratio=2,
        output_dim=128,
        image_start_id=300,
    )
    (directory / "config.json").write_text(json.dumps(values))
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
        base = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        base.save_pretrained(directory, safe_serialization=True)
        del base
        model_config = runtime.SeeClickModelConfig(
            model_id=str(directory), device="cpu", quantization="none", precision="fp32"
        )
        model = runtime.load_model(model_config, training=True)
        image_path = tmp_path / "image.png"
        Image.new("RGB", (80, 80), "red").save(image_path)
        path_bytes = list(str(image_path).encode())
        assert len(path_bytes) < 255
        # Original Qwen-VL embeds a byte-encoded path in a fixed 256-token image span.
        ids = [1, 300] + path_bytes + [302] * (256 - len(path_bytes)) + [301, 2, 3]
        inputs = torch.tensor([ids])
        labels = torch.full_like(inputs, -100)
        labels[:, -2:] = inputs[:, -2:]
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
        output.loss.backward()
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        assert torch.isfinite(output.loss)
        assert trainable and all(
            "lora_" in name and ".visual." not in name for name, _ in trainable
        )
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for _, parameter in trainable
        )
        assert all(parameter.dtype == torch.float32 for _, parameter in trainable)
        adapter = tmp_path / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        del model
        loaded = runtime.load_model(model_config, adapter_path=adapter)
        with torch.inference_mode():
            output = loaded.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=511,
            )
        assert inputs.shape[1] < output.shape[1] <= inputs.shape[1] + 2
    finally:
        torch.set_num_threads(original_threads)
