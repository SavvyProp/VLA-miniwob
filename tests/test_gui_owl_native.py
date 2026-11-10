"""Native Qwen3-VL checks with a tiny local tokenizer; no pretrained downloads."""

import json

import pytest


def make_tiny_processor():
    """Build the actual HF processor around a <512-token local byte-level BPE."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    backend = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [
            "system user assistant PROCEDURE EXECUTED CLICK HISTORY screenshot current previous",
            'Click the red button. {"action_type": 4, "click_point": [0.5, 0.25]}',
        ],
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special,
        ),
    )
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=special[1:],
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
        "{% endfor %}{% endif %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            min_pixels=1024, max_pixels=1024, patch_size=16, temporal_patch_size=2, merge_size=2
        ),
        video_processor=Qwen3VLVideoProcessor(
            size={"shortest_edge": 1024, "longest_edge": 1024},
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
        ),
        chat_template=template,
    )


def test_native_qwen3_lora_training_reload_and_resume(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("Requires the isolated GUI-Owl runtime")
    from test_gui_owl_data import make_trajectories

    from scriptwob import gui_owl_runtime as runtime
    from scriptwob import gui_owl_training as training
    from scriptwob.gui_owl_data import GUIOwlCollator, GUIOwlDataset
    from scriptwob.seeclick_data import load_training_export

    processor = make_tiny_processor()
    tokenizer = processor.tokenizer
    text_config = {
        "vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "max_position_embeddings": 4096,
        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
    }
    vision_config = {
        "hidden_size": 32, "intermediate_size": 64, "depth": 2, "num_heads": 4,
        "out_hidden_size": 32, "patch_size": 16, "temporal_patch_size": 2,
        "spatial_merge_size": 2, "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0, 1],
    }
    native_config = transformers.Qwen3VLConfig(
        text_config=text_config, vision_config=vision_config,
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    native_config._name_or_path = runtime.GUI_OWL_MODELS["2b"][0]

    def base_model(*args, **kwargs):
        torch.manual_seed(11)
        model = transformers.Qwen3VLForConditionalGeneration(native_config)
        model.name_or_path = native_config._name_or_path
        return model

    # Replace only the weight download, keeping the actual native model,
    # processor, PEFT adapter, Trainer, optimizer and checkpoint serialization.
    monkeypatch.setattr(
        transformers.Qwen3VLForConditionalGeneration, "from_pretrained", base_model,
    )
    monkeypatch.setattr(
        "peft.utils.save_and_load.check_file_exists_on_hf_hub", lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(training, "load_processor", lambda config: processor)
    monkeypatch.setattr(training, "implementation_digest", lambda: "native-test-code")
    models = []

    def load_model(*args, **kwargs):
        model = runtime.load_model(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(training, "load_model", load_model)
    settings = training.GUIOwlTrainingConfig(
        model=runtime.GUIOwlModelConfig(
            device="cpu", quantization="none", precision="fp32", lora_rank=2,
            lora_alpha=4, lora_dropout=0, gradient_checkpointing=True,
            min_pixels=1024, max_pixels=1024,
        ),
        max_length=2048, max_steps=2, batch_size=2, gradient_accumulation_steps=1,
        learning_rate=0.01, warmup_ratio=0, save_steps=100, logging_steps=1,
    )
    source = make_trajectories(tmp_path / "data").directory
    output = tmp_path / "training"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = training.train_gui_owl(source, output, settings)
        assert result["global_step"] == 2
        checkpoint = training.latest_checkpoint(output)
        assert checkpoint.name == "checkpoint-2"
        assert (checkpoint / "optimizer.pt").stat().st_size > 0
        trained = models[-1]
        assert all(not p.requires_grad for p in trained.base_model.model.model.visual.parameters())
        assert any(torch.count_nonzero(p).item() > 0
                   for name, p in trained.named_parameters() if "lora_B" in name)
        resolved, adapter, metadata = runtime.resolve_checkpoint(output)
        assert metadata["history_images"] == 3
        loaded = runtime.load_model(resolved, adapter_path=adapter)
        dataset = GUIOwlDataset(load_training_export(source), processor, 2048)
        batch = GUIOwlCollator(processor)([dataset[0], dataset[2]])
        trained.eval()
        with torch.inference_mode():
            original = trained(**batch).logits
            restored = loaded(**batch).logits
            assert torch.allclose(original, restored, atol=1e-6)
            assert torch.isfinite(restored).all()
            single = GUIOwlCollator(processor)([dataset[2]])
            single.pop("labels")
            generated = loaded.generate(**single, max_new_tokens=2, do_sample=False)
            assert generated.shape[1] > single["input_ids"].shape[1]
        resumed = training.train_gui_owl(
            source, output, settings.model_copy(update={"max_steps": 3}), resume=True,
        )
        assert resumed["global_step"] == 3
        state = json.loads((training.latest_checkpoint(output) / "trainer_state.json").read_text())
        assert state["global_step"] == 3
    finally:
        torch.set_num_threads(old_threads)
