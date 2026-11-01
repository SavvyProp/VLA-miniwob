"""Exercise the real upstream architecture at tiny width, without network/GPU."""

import copy
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("llava_pythia")
pytest.importorskip("peft")

from vla_miniwob.tinyvla_model import TinyVLABackend, _upstream  # noqa: E402


@pytest.fixture
def backend(tmp_path, monkeypatch):
    import policy_heads.models
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import CLIPImageProcessor, PreTrainedTokenizerFast

    # Keep native modules, diffusion operations and serialization; reduce only
    # dimensions so this integration test runs on CPU without released weights.
    monkeypatch.setattr(
        policy_heads.models, "ConditionalUnet1D",
        partial(policy_heads.models.ConditionalUnet1D,
                down_dims=(8, 16, 32), diffusion_step_embed_dim=16, kernel_size=3),
    )
    torch.set_num_threads(1)
    config_class, model_class = _upstream()
    config = config_class(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, vocab_size=8, max_position_embeddings=128,
        action_head_type="droid_diffusion", action_dim=3, state_dim=3, chunk_size=16,
        image_aspect_ratio="pad", mm_use_im_start_end=False,
        vision_config={
            "vision_tower": {
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 2, "image_size": 8, "patch_size": 4,
                "vision_model_name_or_path": "clip-tiny-test",
                "mm_vision_select_feature": "patch", "mm_vision_select_layer": -2,
            },
            "mm_projector": {
                "mm_hidden_size": 16, "hidden_size": 32,
                "mm_projector_type": "mlp2x_gelu",
            },
        },
    )
    source = tmp_path / "backbone"
    source.mkdir()
    # Released TinyVLA-B weights are a VLM, without a trained diffusion head.
    vlm = {name: value for name, value in model_class(config).state_dict().items()
           if not name.startswith("embed_out.")}
    vlm["embed_out.weight"] = torch.zeros(8, 32)
    save_file(vlm, source / "model.safetensors", metadata={"format": "pt"})
    config.save_pretrained(source)
    tokenizer = Tokenizer(WordLevel({
        "<unk>": 0, "<pad>": 1, "<eos>": 2, "click": 3,
        "button": 4, "history": 5, "move": 6, "test": 7,
    }, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
    ).save_pretrained(source)
    CLIPImageProcessor(
        size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
    ).save_pretrained(source)
    recipe = {
        "backbone": str(source), "backbone_revision": "local-test",
        "action_head_type": "droid_diffusion", "action_dim": 3, "state_dim": 3,
        "chunk_size": 16, "model_max_length": 128,
        "scheduler": {"inference_steps": 2},
        "adaptation_starting_point": {"lora_r": 2, "lora_alpha": 4, "lora_module": "vit llm"},
    }
    return TinyVLABackend.from_pretrained(recipe, device="cpu", dtype="float32")


def sample(text="click button", valid=1):
    return {
        "image": np.zeros((12, 16, 3), dtype=np.uint8),
        "state": np.array([0.1, 0.2, 0], dtype=np.float32),
        "raw_lang": text,
        "action": np.zeros((16, 3), dtype=np.float32),
        "is_pad": np.arange(16) >= valid,
    }


def test_native_training_masks_padding_and_trains_all_adapters(backend):
    batch = backend.collate([backend.processor(sample()), backend.processor(sample(valid=3))])
    altered = copy.deepcopy(batch)
    altered["action"][altered["is_pad"]] = 1e4
    before = backend.loss(batch, generator=torch.Generator().manual_seed(19))
    after = backend.loss(altered, generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(before, after)
    before.backward()
    parameters = dict(backend.model.named_parameters())
    for group in ("embed_out", "mm_projector", "layers.0.attention.query_key_value.lora_B",
                  "vision_model.encoder.layers.0.self_attn.q_proj.lora_B"):
        assert any(group in name and value.grad is not None and torch.any(value.grad != 0)
                   for name, value in parameters.items())
    assert all(not value.requires_grad for name, value in parameters.items()
               if "lora_" not in name and "embed_out" not in name and "mm_projector" not in name)


def test_native_padding_does_not_change_conditioning(backend):
    backend.model.eval()
    first = backend.processor(sample())
    second = backend.processor(sample("click button " + "history " * 12))
    with torch.no_grad():
        single = backend._conditioning(backend.collate([first]))[0]
        together = backend._conditioning(backend.collate([first, second]))[0]
    torch.testing.assert_close(single, together, atol=1e-6, rtol=1e-5)


def test_native_checkpoint_roundtrip_preserves_diffusion_projector_and_lora(backend, tmp_path):
    batch = backend.collate([backend.processor(sample())])
    optimizer = torch.optim.AdamW(
        [value for value in backend.model.parameters() if value.requires_grad], lr=0.001
    )
    backend.loss(batch, generator=torch.Generator().manual_seed(5)).backward()
    optimizer.step()
    backend.model.eval()
    expected = backend.sample(batch, generator=torch.Generator().manual_seed(7))
    checkpoint = tmp_path / "checkpoint"
    backend.save(checkpoint)
    restored = TinyVLABackend.from_checkpoint(checkpoint, device="cpu", dtype="float32")
    actual = restored.sample(batch, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (1, 16, 3)
    assert not any(value.requires_grad for value in restored.model.parameters())
    trainable = TinyVLABackend.from_checkpoint(
        checkpoint, device="cpu", dtype="float32", training=True,
    )
    assert {item["name"] for item in trainable.trainable_parameters()} == {
        item["name"] for item in backend.trainable_parameters()
    }


def test_native_context_preflight_counts_actual_image_tokens(backend):
    assert backend.image_tokens == 4
    with pytest.raises(ValueError, match="not truncated"):
        backend.processor.text_ids("history " * 128)
    with pytest.raises(ValueError, match="reserved"):
        backend.processor.text_ids("click <image>")


def test_bfloat16_compute_keeps_exact_float32_optimizer_weights(backend, tmp_path):
    recipe = copy.deepcopy(backend.recipe)
    recipe["training"] = {"gradient_checkpointing": True}
    mixed = TinyVLABackend.from_pretrained(recipe, device="cpu", dtype="bfloat16")
    native = mixed.native_model
    assert native.get_model().gradient_checkpointing
    assert native.get_vision_tower().vision_model.encoder.gradient_checkpointing
    assert all(parameter.dtype == torch.float32 for parameter in mixed.model.parameters()
               if parameter.requires_grad)
    assert native.get_model().embed_in.weight.dtype == torch.bfloat16
    batch = mixed.collate([mixed.processor(sample())])
    optimizer = torch.optim.AdamW(
        [value for value in mixed.model.parameters() if value.requires_grad], lr=2e-5,
    )
    loss = mixed.loss(batch, generator=torch.Generator().manual_seed(3))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "mixed-checkpoint"
    mixed.save(checkpoint)
    resumed = TinyVLABackend.from_checkpoint(
        checkpoint, device="cpu", dtype="bfloat16", training=True,
    )
    restored_weights = dict(resumed.model.named_parameters())
    for name, value in mixed.model.named_parameters():
        if value.requires_grad:
            assert restored_weights[name].dtype == torch.float32
            torch.testing.assert_close(value, restored_weights[name], atol=0, rtol=0)
    resumed.model.eval()
    actions = resumed.sample(batch, generator=torch.Generator().manual_seed(4))
    assert actions.shape == (1, 16, 3) and torch.isfinite(actions).all()
    inference = TinyVLABackend.from_checkpoint(checkpoint, device="cpu", dtype="bfloat16")
    reloaded = inference.sample(batch, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(reloaded, actions, atol=0, rtol=0)


def test_native_training_checkpoint_loads_in_benchmark_policy(backend, tmp_path):
    from PIL import Image
    from test_dataset import fixture_run

    from vla_miniwob.contracts import PolicyObservation
    from vla_miniwob.tinyvla_policy import TinyVLAPolicy
    from vla_miniwob.training import train

    source = fixture_run(tmp_path)
    recipe = copy.deepcopy(backend.recipe)
    recipe["training"] = {
        "max_steps": 1, "batch_size": 1, "gradient_accumulation": 1,
        "warmup_steps": 0, "save_every": 1, "gradient_checkpointing": True,
    }
    output = tmp_path / "trained"
    trained = train(
        source, output, recipe, device="cpu", dtype="float32",
        progress=lambda *_args, **_kwargs: None,
    )
    assert trained["step"] == 1 and trained["trained"]
    observation = PolicyObservation(
        instruction="Click A, then B.", image=Image.new("RGB", (20, 10)),
        state=(0, 0, 0), history=(),
    )
    policy = TinyVLAPolicy(output, device="cpu", dtype="float32")
    policy.reset(33)
    expected = policy.predict(observation)
    assert expected.shape == (16, 3) and np.isfinite(expected).all()
    restored = TinyVLAPolicy(output, device="cpu", dtype="float32")
    restored.reset(33)
    np.testing.assert_array_equal(restored.predict(observation), expected)
    assert restored.provenance()["training_step"] == 1
