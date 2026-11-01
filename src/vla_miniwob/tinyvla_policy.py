"""Autonomous screenshot-only benchmark adapter for a trained TinyVLA checkpoint."""

import json
from pathlib import Path

import numpy as np

from .contracts import PolicyObservation, format_context
from .dataset import Normalizer
from .policies import PolicyOutputError
from .training import file_digest, resolve_checkpoint, verify_checkpoint


class TinyVLAPolicy:
    def __init__(self, checkpoint: str | Path, *, device="cuda", dtype=None):
        import torch

        from .tinyvla_model import TinyVLABackend

        self.checkpoint = resolve_checkpoint(Path(checkpoint))
        self.metadata = verify_checkpoint(self.checkpoint)
        self.backend = TinyVLABackend.from_checkpoint(self.checkpoint, device=device, dtype=dtype)
        self.normalizer = Normalizer(
            json.loads((self.checkpoint / "normalization.json").read_text())
        )
        self.prediction_horizon = self.backend.recipe["chunk_size"]
        self.generator = torch.Generator(device=self.backend.device)
        self.checkpoint_sha256 = file_digest(self.checkpoint / "checkpoint.json")

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def predict(self, observation: PolicyObservation) -> np.ndarray:
        try:
            sample = self.backend.processor(
                {
                    "image": observation.image,
                    "raw_lang": format_context(observation.instruction, observation.history),
                    "state": self.normalizer.states(
                        np.asarray(observation.state, dtype=np.float32)
                    ),
                }
            )
        except ValueError as error:
            # Full-history overflow is a model failure, not browser infrastructure.
            raise PolicyOutputError(str(error)) from error
        chunk = self.backend.sample(self.backend.collate([sample]), generator=self.generator)
        actions = self.normalizer.undo_actions(chunk[0].float().cpu().numpy())
        if actions.shape != (self.prediction_horizon, 3) or not np.isfinite(actions).all():
            raise PolicyOutputError("TinyVLA returned an invalid numerical action chunk")
        return actions

    def provenance(self) -> dict:
        return {
            "kind": "tinyvla_b",
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_step": self.metadata["step"],
            "prediction_horizon": self.prediction_horizon,
            "recipe": self.backend.recipe,
            "dtype": self.backend.dtype_name,
            "device": str(self.backend.device),
            "temporal_aggregation": False,
            "chunk_samples_per_observation": 1,
            "training_fingerprint": self.metadata["run"]["training_fingerprint"],
        }
