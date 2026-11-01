"""Native TinyVLA-B backbone, LoRA adapters and GUI diffusion policy.

The architecture comes from the pinned TinyVLA source. This module changes its
robot data plumbing: one image, three-dimensional state/actions, padding-aware
pooling and epsilon loss, and device-independent, seeded chunk sampling.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import masked_diffusion_loss

IMAGE_TOKEN_INDEX = -200
BACKEND_FORMAT = 1


def _torch_dtype(name):
    import torch

    values = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in values:
        raise ValueError(f"Unsupported model dtype: {name}")
    return values[name]


def _upstream():
    try:
        from llava_pythia.model.language_model.pythia.configuration_llava_pythia import (
            LlavaPythiaConfig,
        )
        from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "TinyVLA training dependencies/source are missing. Follow docs/tinyvla.md "
            "to prepare the pinned training environment."
        ) from error
    return LlavaPythiaConfig, LlavaPythiaForCausalLM


def _validate_recipe(recipe):
    if (recipe["action_dim"], recipe["state_dim"], recipe["chunk_size"]) != (3, 3, 16):
        raise ValueError("The initial GUI TinyVLA backend requires action/state dims 3 and H=16")
    if recipe["action_head_type"] != "droid_diffusion":
        raise ValueError("TinyVLA-B requires the native droid_diffusion action head")
    if recipe.get("temporal_aggregation", False):
        raise ValueError("GUI inference does not use robot temporal aggregation")
    if recipe.get("camera_names", ["screen"]) != ["screen"]:
        raise ValueError("The GUI backend requires exactly one screen camera")
    adaptation = recipe["adaptation_starting_point"]
    if not adaptation.get("lora_enable", True):
        raise ValueError("This backend implements the LoRA TinyVLA-B recipe")


class TinyVLAProcessor:
    """The same untruncated prompt and native image pipeline for train and rollout.

    Input is a raw ``ChunkDataset`` sample (uint8 HWC RGB image, normalized state,
    full ``raw_lang``, and optionally normalized action/is_pad arrays).
    """

    def __init__(self, tokenizer, image_processor, model_config, image_tokens, max_length):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.image_tokens = int(image_tokens)
        self.max_length = int(max_length)
        if self.image_tokens < 1 or self.max_length < self.image_tokens:
            raise ValueError("Invalid multimodal context capacity")

    def text_ids(self, text):
        from llava_pythia.conversation import conv_templates
        from llava_pythia.mm_utils import tokenizer_image_token

        # A literal image marker in a task is data, never a second camera slot.
        if "<image>" in text:
            raise ValueError("Task context contains the reserved <image> marker")
        conv = conv_templates["pythia"].copy()
        conv.append_message(conv.roles[0], "<image>\n" + text)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors="pt")
        total = len(ids) + self.image_tokens - 1
        if total > self.max_length:
            raise ValueError(
                f"Full GUI procedure/history needs {total} multimodal tokens; "
                f"capacity is {self.max_length}. Context was not truncated."
            )
        if int((ids == IMAGE_TOKEN_INDEX).sum()) != 1:
            raise ValueError("The GUI prompt must contain exactly one image token")
        return ids

    def image(self, value):
        from llava_pythia.mm_utils import process_images

        if not isinstance(value, Image.Image):
            value = Image.fromarray(np.asarray(value, dtype=np.uint8))
        return process_images([value.convert("RGB")], self.image_processor, self.model_config)[0]

    def __call__(self, sample):
        import torch

        result = {
            "input_ids": self.text_ids(sample["raw_lang"]),
            "image": self.image(sample["image"]),
            "state": torch.as_tensor(sample["state"], dtype=torch.float32),
        }
        if result["state"].shape != (3,):
            raise ValueError("Expected normalized cursor/button state with shape (3,)")
        if "action" in sample:
            result["action"] = torch.as_tensor(sample["action"], dtype=torch.float32)
            result["is_pad"] = torch.as_tensor(sample["is_pad"], dtype=torch.bool)
        return result

    def collate(self, samples):
        import torch

        if not samples:
            raise ValueError("Cannot collate an empty batch")
        result = {
            # Retaining lengths avoids interpreting a real EOS token as padding.
            "input_ids": [sample["input_ids"] for sample in samples],
            "image": torch.stack([sample["image"] for sample in samples]),
            "state": torch.stack([sample["state"] for sample in samples]),
        }
        if "action" in samples[0]:
            result["action"] = torch.stack([sample["action"] for sample in samples])
            result["is_pad"] = torch.stack([sample["is_pad"] for sample in samples])
        return result


class TinyVLABackend:
    """A native TinyVLA model and the shared training/inference operations.

    ``model`` is a PEFT model; optimize its parameters and set train/eval on it.
    ``loss`` and ``sample`` accept ``collate`` output and move tensors to device.
    ``sample`` returns normalized B x H x 3 actions, without action clipping or
    temporal averaging. The benchmark adapter undoes dataset normalization.
    """

    def __init__(self, model, tokenizer, image_processor, recipe, *, dtype="float32"):
        import torch

        self.model = model
        self.recipe = copy.deepcopy(recipe)
        self.dtype_name = dtype
        # AdamW must update float32 master parameters: bf16 weights can round
        # away the small projector/decoder learning-rate updates entirely.
        # Frozen backbone weights keep the requested storage/compute precision.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        native = self.native_model
        # Upstream hardcodes bf16 in its sinusoidal embedding, including on CPU.
        native.embed_out.diffusion_step_encoder[0].dtype = _torch_dtype(dtype)
        native.config.use_cache = False
        maximum = min(int(recipe["model_max_length"]), native.config.max_position_embeddings)
        tokenizer.model_max_length = maximum
        native.eval()
        with torch.no_grad(), self._autocast():
            provisional = TinyVLAProcessor(tokenizer, image_processor, native.config, 1, maximum)
            pixels = provisional.image(Image.new("RGB", (720, 540)))
            encoded = native.encode_images(pixels[None].to(self.device, dtype=self.dtype))
            visual_tokens = encoded.shape[1]
        self.processor = TinyVLAProcessor(
            tokenizer, image_processor, native.config, visual_tokens, maximum
        )
        self.collate = self.processor.collate
        self.image_tokens = visual_tokens
        settings = recipe.get("scheduler", {})
        for key in (
            "prediction_type", "num_train_timesteps", "beta_schedule", "clip_sample",
            "set_alpha_to_one", "steps_offset",
        ):
            if key in settings and settings[key] != getattr(native.noise_scheduler.config, key):
                raise ValueError(f"Recipe differs from native TinyVLA diffusion setting: {key}")
        self.inference_steps = int(settings.get("inference_steps", 10))
        if not 1 <= self.inference_steps <= native.noise_scheduler.config.num_train_timesteps:
            raise ValueError("Invalid DDIM inference timestep count")
        if recipe.get("training", {}).get("gradient_checkpointing", False):
            native.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def native_model(self):
        return self.model.get_base_model()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return _torch_dtype(self.dtype_name)

    def _autocast(self):
        import torch

        return torch.autocast(
            device_type=self.device.type, dtype=self.dtype,
            enabled=self.dtype_name != "float32",
        )

    @classmethod
    def _load_base(cls, recipe, device, dtype, *, architecture_path=None):
        from transformers import AutoTokenizer, CLIPImageProcessor, SiglipImageProcessor

        config_class, model_class = _upstream()
        _validate_recipe(recipe)
        source = recipe["backbone"]
        revision = recipe["backbone_revision"]
        if architecture_path is None:
            config = config_class.from_pretrained(source, revision=revision)
            config.action_head_type = "droid_diffusion"
            config.action_dim = recipe["action_dim"]
            config.state_dim = recipe["state_dim"]
            config.chunk_size = recipe["chunk_size"]
            config.concat = "token_cat"
        else:
            config = config_class.from_pretrained(architecture_path)
            if (config.action_dim, config.state_dim, config.chunk_size) != (3, 3, 16):
                raise ValueError("Checkpoint architecture does not match the GUI action contract")
        if getattr(config, "mm_use_im_start_end", False):
            raise ValueError("This pinned GUI recipe expects the backbone's plain image marker")
        model, info = model_class.from_pretrained(
            source,
            revision=revision,
            config=config,
            torch_dtype=_torch_dtype(dtype),
            low_cpu_mem_usage=True,
            output_loading_info=True,
            attn_implementation="eager",
        )
        # Only the robot decoder is newly initialized. Do not silently accept a
        # changed/mismatched VLM or projector from a different checkpoint.
        missing = [key for key in info["missing_keys"] if not key.startswith("embed_out.")]
        unexpected = [key for key in info["unexpected_keys"] if key != "embed_out.weight"]
        if missing or unexpected or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError(f"Backbone weights do not match TinyVLA-B: {info}")
        if any(key.startswith("embed_out.") for key in info["missing_keys"]):
            from policy_heads.models import ConditionalUnet1D

            # Transformers' low-memory loader cannot materialize every custom
            # U-Net tensor from a VLM-only checkpoint. Build its native head
            # outside the meta-device context, retaining native initialization.
            model.embed_out = ConditionalUnet1D(
                input_dim=config.action_dim, global_cond_dim=config.hidden_size,
                state_dim=config.state_dim,
            )
            model.embed_out.apply(model._init_weights)
        if any(value.is_meta for value in model.state_dict().values()):
            raise ValueError("TinyVLA loading left uninitialized meta-device tensors")
        tokenizer_source = architecture_path or source
        options = {} if architecture_path else {"revision": revision}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, **options)
        vision_name = config.vision_config["vision_tower"]["vision_model_name_or_path"]
        processor_class = CLIPImageProcessor if "clip" in vision_name else SiglipImageProcessor
        image_processor = processor_class.from_pretrained(tokenizer_source, **options)
        return model.to(device), tokenizer, image_processor

    @classmethod
    def from_pretrained(cls, recipe, *, device="cuda", dtype="bfloat16", training=True):
        import torch
        from peft import LoraConfig, get_peft_model

        native, tokenizer, image_processor = cls._load_base(recipe, device, dtype)
        adaptation = recipe["adaptation_starting_point"]
        requested = set(adaptation.get("lora_module", "vit llm").split())
        if not requested or requested - {"vit", "llm"}:
            raise ValueError("LoRA modules must be selected from 'vit llm'")
        targets = [
            name for name, module in native.named_modules()
            if isinstance(module, torch.nn.Linear)
            and (("llm" in requested and name.startswith("gpt_neox.layers."))
                 or ("vit" in requested and ".vision_model.encoder.layers." in name))
        ]
        if not targets:
            raise ValueError("No native transformer layers matched the LoRA recipe")
        native.requires_grad_(False)
        model = get_peft_model(native, LoraConfig(
            r=int(adaptation["lora_r"]), lora_alpha=int(adaptation["lora_alpha"]),
            lora_dropout=float(adaptation.get("lora_dropout", 0.0)),
            target_modules=targets, bias="none",
        ))
        # These are saved separately so PEFT does not duplicate the large U-Net
        # in a ModulesToSaveWrapper or omit it from the adapter checkpoint.
        native.embed_out.requires_grad_(True)
        native.get_model().mm_projector.requires_grad_(True)
        backend = cls(model, tokenizer, image_processor, recipe, dtype=dtype)
        model.train(training)
        if not training:
            model.requires_grad_(False)
        return backend

    def trainable_parameters(self):
        return [
            {"name": name, "numel": parameter.numel(), "shape": list(parameter.shape)}
            for name, parameter in self.model.named_parameters() if parameter.requires_grad
        ]

    def provenance(self):
        return {
            "backend": "native_tinyvla_b_gui",
            "backend_format": BACKEND_FORMAT,
            "recipe": copy.deepcopy(self.recipe),
            "dtype": self.dtype_name,
            "trainable_parameter_dtype": "float32",
            "actual_visual_tokens": self.image_tokens,
            "model_max_length": self.processor.max_length,
            "prompt_template": "pythia_gui_single_image_v1",
            "pooling": "mean over nonpadding multimodal tokens",
            "loss": "epsilon MSE, mean over valid actions per example, then examples",
            "prediction_horizon": 16,
            "temporal_aggregation": False,
            "inference_steps": self.inference_steps,
            "trainable_parameters": self.trainable_parameters(),
        }

    def _conditioning(self, batch):
        with self._autocast():
            return self._conditioning_impl(batch)

    def _conditioning_impl(self, batch):
        import torch

        native = self.native_model
        visual = native.encode_images(batch["image"].to(self.device, dtype=self.dtype))
        if visual.shape[1] != self.image_tokens:
            raise ValueError("Actual visual token count changed from the preflight image pipeline")
        embedded = []
        for index, ids in enumerate(batch["input_ids"]):
            ids = ids.to(self.device)
            positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(positions) != 1:
                raise ValueError("Expected exactly one image marker in each prompt")
            position = int(positions[0])
            pieces = (
                native.get_model().embed_in(ids[:position]),
                visual[index],
                native.get_model().embed_in(ids[position + 1:]),
            )
            value = torch.cat(pieces, dim=0)
            if value.shape[0] > self.processor.max_length:
                raise ValueError("Full multimodal input exceeds the model context window")
            embedded.append(value)
        lengths = torch.tensor([len(value) for value in embedded], device=self.device)
        inputs = torch.nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        valid = torch.arange(inputs.shape[1], device=self.device)[None] < lengths[:, None]
        hidden = native.get_model()(
            inputs_embeds=inputs, attention_mask=valid, use_cache=False, return_dict=True,
        ).last_hidden_state
        # The upstream U-Net mean-pools its sequence. Pool only genuine tokens,
        # then pass one token so padding cannot change a sample's conditioning.
        pooled = (hidden * valid[..., None]).sum(1) / lengths[:, None]
        return pooled[:, None].to(self.dtype)

    def loss(self, batch, *, generator=None):
        with self._autocast():
            return self._loss_impl(batch, generator=generator)

    def _loss_impl(self, batch, *, generator=None):
        import torch

        native = self.native_model
        actions = batch["action"].to(self.device, dtype=self.dtype)
        is_pad = batch["is_pad"].to(self.device)
        if actions.ndim != 3 or actions.shape[1:] != (16, 3):
            raise ValueError("TinyVLA loss requires action shape B x 16 x 3")
        if is_pad.shape != actions.shape[:2]:
            raise ValueError("TinyVLA loss requires padding shape B x 16")
        # Padded target values have no information and must not leak through the
        # temporal U-Net into valid targets. Gaussian noise is still sampled there.
        actions = actions.masked_fill(is_pad[..., None], 0)
        noise = torch.randn(
            actions.shape, device=self.device, dtype=self.dtype, generator=generator,
        )
        steps = torch.randint(
            native.noise_scheduler.config.num_train_timesteps, (len(actions),),
            device=self.device, generator=generator,
        )
        noisy = native.noise_scheduler.add_noise(actions, noise, steps)
        predicted = native.embed_out(
            noisy, steps, global_cond=self._conditioning(batch),
            states=batch["state"].to(self.device, dtype=self.dtype),
        )
        # Reduce in float32 even when the native model uses bf16/fp16.
        return masked_diffusion_loss(predicted.float(), noise.float(), is_pad)

    def sample(self, batch, *, generator=None):
        import torch

        native = self.native_model
        with torch.no_grad(), self._autocast():
            condition = self._conditioning(batch)
            actions = torch.randn(
                (len(condition), self.recipe["chunk_size"], self.recipe["action_dim"]),
                device=self.device, dtype=self.dtype, generator=generator,
            )
            scheduler = native.noise_scheduler
            scheduler.set_timesteps(self.inference_steps, device=self.device)
            state = batch["state"].to(self.device, dtype=self.dtype)
            for timestep in scheduler.timesteps:
                epsilon = native.embed_out(actions, timestep, global_cond=condition, states=state)
                actions = scheduler.step(epsilon, timestep, actions, eta=0.0).prev_sample
        return actions.float()

    def save(self, path):
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
        native = self.native_model
        auxiliary = {
            name: value.detach().cpu().contiguous()
            for name, value in native.state_dict().items()
            if name.startswith(("embed_out.", "gpt_neox.mm_projector."))
        }
        save_file(auxiliary, path / "non_lora_trainables.safetensors")
        native.config.save_pretrained(path)
        self.processor.tokenizer.save_pretrained(path)
        self.processor.image_processor.save_pretrained(path)
        (path / "backend.json").write_text(json.dumps({
            "format": BACKEND_FORMAT, "recipe": self.recipe,
            "image_tokens": self.image_tokens,
            "prompt_template": "pythia_gui_single_image_v1",
            "dtype": self.dtype_name,
        }, indent=2) + "\n")

    @classmethod
    def from_checkpoint(cls, path, *, device="cuda", dtype=None, training=False):
        from peft import PeftConfig, get_peft_model
        from safetensors.torch import load_file

        path = Path(path)
        metadata = json.loads((path / "backend.json").read_text())
        if metadata["format"] != BACKEND_FORMAT:
            raise ValueError("Unsupported TinyVLA checkpoint format")
        recipe = metadata["recipe"]
        dtype = dtype or metadata["dtype"]
        native, tokenizer, image_processor = cls._load_base(
            recipe, device, dtype, architecture_path=path,
        )
        adapter_config = PeftConfig.from_pretrained(path)
        adapter_config.inference_mode = not training
        model = get_peft_model(native, adapter_config)
        native.embed_out.requires_grad_(training)
        native.get_model().mm_projector.requires_grad_(training)
        # Cast before loading: loading FP32 master weights into bf16 parameters
        # and then casting back would lose the exact optimizer resume state.
        for name, parameter in model.named_parameters():
            if (parameter.requires_grad or "lora_" in name
                    or ".embed_out." in name or ".mm_projector." in name):
                parameter.data = parameter.data.float()
        loaded = model.load_adapter(path, "default", is_trainable=training)
        if any("lora_" in name for name in loaded.missing_keys) or loaded.unexpected_keys:
            raise ValueError("Checkpoint has missing or unexpected LoRA adapter weights")
        auxiliary = load_file(path / "non_lora_trainables.safetensors", device="cpu")
        expected = {
            name for name in native.state_dict()
            if name.startswith(("embed_out.", "gpt_neox.mm_projector."))
        }
        if set(auxiliary) != expected:
            raise ValueError("Checkpoint is missing or has unexpected diffusion/projector weights")
        native.load_state_dict(auxiliary, strict=False)
        native.embed_out.requires_grad_(training)
        native.get_model().mm_projector.requires_grad_(training)
        result = cls(model, tokenizer, image_processor, recipe, dtype=dtype)
        if result.image_tokens != metadata["image_tokens"]:
            raise ValueError("Checkpoint visual token count differs from the loaded image pipeline")
        model.train(training)
        if not training:
            model.requires_grad_(False)
        return result
