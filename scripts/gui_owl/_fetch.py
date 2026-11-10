"""Download configured GUI-Owl assets using only huggingface-hub, without PyTorch."""

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import constants, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    model = json.loads(args.config.read_text())["model"]
    identifier, revision = model["model_id"], model["model_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("The model revision must be an immutable 40-character commit hash")
    # Match Transformers 4.57.6's cache resolution. Otherwise users with a
    # legacy cache variable would fetch here and download again during training.
    cache_dir = os.getenv(
        "TRANSFORMERS_CACHE",
        os.getenv(
            "PYTORCH_TRANSFORMERS_CACHE",
            os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE),
        ),
    )
    print(f"Fetching {identifier} at {revision}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    path = snapshot_download(
        repo_id=identifier,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model", "*.tiktoken"],
    )
    print(f"Checkpoint ready: {path}", flush=True)


if __name__ == "__main__":
    main()
