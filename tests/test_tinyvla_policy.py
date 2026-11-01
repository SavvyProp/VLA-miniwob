"""Exercise the evaluation boundary without downloading a pretrained backbone."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vla_miniwob.collection import write_json
from vla_miniwob.contracts import HistoryEntry, MouseAction, PolicyObservation, format_context
from vla_miniwob.policies import PolicyOutputError
from vla_miniwob.training import file_digest

torch = pytest.importorskip("torch")

from vla_miniwob.tinyvla_policy import TinyVLAPolicy  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint-0000007"
    path.mkdir()
    normalization = {
        "version": "gui-train-minmax-v1", "fit_split": "train",
        "action_min": [0.2, 0.1, 0.0], "action_max": [0.8, 0.9, 1.0],
        "state_mean": [0.2, 0.4, 0.0], "state_std": [0.1, 0.2, 1.0],
    }
    write_json(path / "normalization.json", normalization)
    # These payloads are intentionally synthetic. The backend is replaced below,
    # while the policy uses real file integrity checks and normalization.
    for name in (
        "backend.json", "adapter_config.json", "adapter_model.safetensors", "config.json",
        "non_lora_trainables.safetensors", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "training_state.pt",
    ):
        (path / name).write_bytes(b"test-only-backend-payload")
    write_json(path / "checkpoint.json", {
        "format": 1, "step": 7, "run": {"training_fingerprint": "frozen-train-hash"},
        "files": {file.name: file_digest(file) for file in path.iterdir()},
    })
    write_json(tmp_path / "latest.json", {"checkpoint": path.name, "step": 7})
    write_json(tmp_path / "training_run.json", {"format": 1})
    return path


@pytest.fixture
def backend(monkeypatch):
    from vla_miniwob.tinyvla_model import TinyVLABackend

    value = SimpleNamespace(
        recipe={"chunk_size": 16}, device=torch.device("cpu"), dtype_name="float32",
        received=[], calls=0, load_calls=0, fixed=None,
    )

    def processor(sample):
        value.received.append(sample)
        return sample

    def sample(batch, *, generator):
        assert len(batch) == 1
        value.calls += 1
        if value.fixed is not None:
            return value.fixed
        return torch.randn((1, 16, 3), generator=generator)

    def load(*args, **kwargs):
        value.load_calls += 1
        return value

    value.processor, value.collate, value.sample = processor, list, sample
    monkeypatch.setattr(TinyVLABackend, "from_checkpoint", load)
    return value


@pytest.fixture
def observation():
    return PolicyObservation(
        instruction="Click A, then B.", image=Image.new("RGB", (20, 10)),
        state=(0.3, 0.8, 0.0),
        history=(HistoryEntry(step=1, action=MouseAction(x=0.2, y=0.4, click=1),
                              feedback="executed"),),
    )


def test_policy_preserves_context_normalizes_state_and_samples_once(
    checkpoint, backend, observation,
):
    policy = TinyVLAPolicy(checkpoint.parent, device="cpu", dtype="float32")
    backend.fixed = torch.tensor([3.0, -3.0, 0.2]).repeat(1, 16, 1)
    policy.reset(19)
    actions = policy.predict(observation)
    assert backend.calls == 1
    received = backend.received[0]
    assert set(received) == {"image", "raw_lang", "state"}
    assert received["image"] is observation.image
    assert received["raw_lang"] == format_context(observation.instruction, observation.history)
    np.testing.assert_allclose(received["state"], [1, 2, 0], atol=1e-6)
    # Undo min/max normalization, preserving out-of-bounds coordinates and soft
    # click values for the harness to clip/threshold exactly once.
    np.testing.assert_allclose(actions[0], [1.4, -0.7, 0.6], atol=1e-6)
    assert actions.shape == (16, 3) and actions.dtype == np.float32
    provenance = policy.provenance()
    assert provenance["training_step"] == 7
    assert provenance["checkpoint_sha256"] == file_digest(checkpoint / "checkpoint.json")
    assert provenance["training_fingerprint"] == "frozen-train-hash"
    assert provenance["chunk_samples_per_observation"] == 1
    assert provenance["temporal_aggregation"] is False


def test_policy_seed_reset_replays_the_same_diffusion_noise(checkpoint, backend, observation):
    policy = TinyVLAPolicy(checkpoint, device="cpu")
    policy.reset(23)
    first = policy.predict(observation)
    next_action = policy.predict(observation)
    assert not np.array_equal(first, next_action)
    policy.reset(23)
    np.testing.assert_array_equal(policy.predict(observation), first)


@pytest.mark.parametrize("chunk", [
    torch.zeros((1, 15, 3)), torch.full((1, 16, 3), float("nan")),
    torch.full((1, 16, 3), float("inf")),
])
def test_policy_rejects_invalid_native_chunks(checkpoint, backend, observation, chunk):
    policy = TinyVLAPolicy(checkpoint, device="cpu")
    backend.fixed = chunk
    with pytest.raises(PolicyOutputError, match="invalid numerical action chunk"):
        policy.predict(observation)


def test_policy_rejects_checkpoint_corruption_before_loading_model(checkpoint, backend):
    stats = json.loads((checkpoint / "normalization.json").read_text())
    stats["action_max"] = [1, 1, 1]
    write_json(checkpoint / "normalization.json", stats)
    with pytest.raises(ValueError, match="checksum differs"):
        TinyVLAPolicy(checkpoint, device="cpu")
    assert backend.load_calls == 0


def test_context_overflow_counts_as_model_failure(checkpoint, backend, tmp_path):
    from test_collection import FixtureEnvironment

    from vla_miniwob.collection import run
    from vla_miniwob.tasks import make_manifest

    def overflow(_):
        raise ValueError("Full GUI procedure/history needs 9999 tokens; context was not truncated")

    backend.processor = overflow
    policy = TinyVLAPolicy(checkpoint, device="cpu")
    output = tmp_path / "evaluation"
    run(
        make_manifest(0, 0, 1, tasks=["click-button"]), output, policy,
        mode="evaluate", split="eval", attempts=1, max_episodes=1,
        env_factory=FixtureEnvironment,
    )
    result = json.loads(next((output / "episodes").glob("*/private.json")).read_text())
    assert result["verdict"]["reason"] == "invalid_policy_output"
    assert result["infrastructure_failure"] is False
    assert result["primitive_transitions"] == 0
    assert backend.calls == 0
