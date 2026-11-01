"""Check the optional TinyVLA runtime without fetching any model weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    import torch
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from llava_pythia.model.language_model.pythia.llava_pythia import (
        LlavaPythiaForCausalLM,
    )
    from policy_heads.models import ConditionalUnet1D
    from transformers import SiglipImageProcessor

    # Import the native classes, including their transitive dependencies, before
    # printing a successful result. No from_pretrained() or network calls here.
    assert LlavaPythiaForCausalLM and ConditionalUnet1D and SiglipImageProcessor
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(10)
    available = torch.cuda.is_available()
    if available:
        # Detect driver/architecture mismatches that device discovery alone can
        # miss, before a user starts a long backbone download.
        torch.ones(1, device="cuda").add_(1).item()
    result = {
        "imports": "ok",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch", "torchvision", "transformers", "diffusers", "peft",
                "accelerate", "huggingface-hub", "act", "llava-pythia",
            )
        },
        "cuda_available": available,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if available else "cpu",
    }
    print(json.dumps(result, indent=2))
    if args.require_cuda and not available:
        raise SystemExit(
            "CUDA is unavailable. Check the NVIDIA driver, or use the CPU environment "
            "for diagnostics. The CUDA environment uses PyTorch's CUDA 12.4 wheels."
        )


if __name__ == "__main__":
    main()
