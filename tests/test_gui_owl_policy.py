"""Public screenshot memory, reset isolation, and train/eval prompt parity."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from scriptwob import gui_owl_prompt, gui_owl_runtime
from scriptwob.contracts import ClickAction, HistoryEntry, Observation
from scriptwob.gui_owl_policy import GUIOwlPolicy
from scriptwob.policies import PolicyOutputError


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def inference(monkeypatch):
    state = SimpleNamespace(
        metadata={}, adapter=None, requests=[], prompts=[], loads=[], decoded=[], answer=None,
    )
    state.answer = '{"action_type":4,"click_point":[0.251,0.751]}'

    def decode(tokens, **kwargs):
        state.decoded.append(tokens)
        return "".join(chr(token) for token in tokens)

    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=decode))

    def generate(**kwargs):
        state.requests.append(kwargs)
        return [kwargs["input_ids"].values[0] + [ord(char) for char in state.answer]]

    model = SimpleNamespace(
        eval=lambda: None, generate=generate,
        get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(device="cuda:0")),
    )

    def load_model(*args, **kwargs):
        state.loads.append(kwargs)
        return model

    def encode(processor, messages, max_length, reserve_tokens=64):
        state.prompts.append(messages)
        if max_length < 70:
            raise ValueError("context overflow")
        return {
            "input_ids": Tensor([[101, 102, 103]]),
            "attention_mask": Tensor([[1, 1, 1]]),
            "pixel_values": Tensor([[1.0]]),
            "image_grid_thw": Tensor([[1, 2, 2]]),
        }

    monkeypatch.setattr(gui_owl_runtime, "resolve_checkpoint", lambda *a, **k: (
        gui_owl_runtime.GUIOwlModelConfig(), state.adapter, state.metadata,
    ))
    monkeypatch.setattr(gui_owl_runtime, "load_model", load_model)
    monkeypatch.setattr(gui_owl_runtime, "load_processor", lambda config: processor)
    monkeypatch.setattr(gui_owl_prompt, "build_inference_inputs", encode)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    return state


def observation(step=0, *, instruction="IF checkbox Oak is checked, click Red. Then click Done."):
    # Coordinates are rounded as the browser rounds a prediction to physical pixels.
    previous = ClickAction(x=25 / 100, y=75 / 100)
    return Observation(
        instruction,
        Image.new("RGBA", (101, 101), (step * 10, 0, 0, 255)),
        tuple(HistoryEntry(step=i, action=previous) for i in range(1, step + 1)),
    )


def content_text(messages):
    return "".join(
        item["text"] for message in messages if isinstance(message["content"], list)
        for item in message["content"] if item["type"] == "text"
    )


def images(messages):
    return [
        item["image"] for message in messages if isinstance(message["content"], list)
        for item in message["content"] if item["type"] == "image"
    ]


def test_rolling_images_are_chronological_and_full_click_history_is_retained(inference):
    policy = GUIOwlPolicy(history_images=2)
    for step in range(5):
        assert policy.predict(observation(step)) == ClickAction(x=0.251, y=0.751)
    last = inference.prompts[-1]
    assert [image.getpixel((0, 0)) for image in images(last)] == [
        (20, 0, 0), (30, 0, 0), (40, 0, 0),
    ]
    text = content_text(last)
    for step in range(1, 5):
        assert f"{step}: (0.250000, 0.750000) clicked" in text
    assert "Screenshot after 2 clicks (previous)" in text
    assert "Screenshot after 4 clicks (current)" in text
    assert not any(private in text for private in ["oracle", "expected_action", "world"])
    assert policy.calls[-1]["image_steps"] == [2, 3, 4]
    assert policy.calls[-1]["history_image_count"] == 2
    assert policy.calls[-1]["image_count"] == 3
    assert len(set(policy.calls[-1]["images_sha256"])) == 3
    assert len(inference.loads) == 1
    request = inference.requests[-1]
    assert request["do_sample"] is False and request["num_beams"] == 1
    assert request["use_cache"] is True
    assert request["max_new_tokens"] == 64
    assert request["pixel_values"].device == "cuda:0"
    assert inference.decoded[-1] == [ord(char) for char in inference.answer]
    provenance = policy.provenance()
    assert provenance["history_images"] == 2 and provenance["action_history"] == "complete"
    provenance["history_images"] = 30
    assert policy.provenance()["history_images"] == 2


def test_current_image_only_ablation_and_reset_cannot_leak_prior_frames(inference):
    policy = GUIOwlPolicy(history_images=0)
    policy.predict(observation())
    policy.predict(observation(1))
    assert len(images(inference.prompts[-1])) == 1
    assert policy.calls[-1]["image_steps"] == [1]
    policy.reset(42)
    policy.predict(observation(instruction="Click Done."))
    assert len(policy.calls) == 1 and policy.calls[0]["image_steps"] == [0]
    assert "(none)" in content_text(inference.prompts[-1])
    assert "checkbox Oak" not in content_text(inference.prompts[-1])


def test_cached_images_are_copied_not_mutable_environment_references(inference):
    policy = GUIOwlPolicy()
    initial = observation()
    policy.predict(initial)
    initial.image.paste("white", (0, 0, 101, 101))
    policy.predict(observation(1))
    assert images(inference.prompts[-1])[0].getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("failure", ["skip", "repeat", "changed_prefix", "wrong_click", "task"])
def test_history_must_match_observed_sequential_execution(inference, failure):
    policy = GUIOwlPolicy()
    policy.predict(observation())
    policy.predict(observation(1))
    value = observation(2)
    if failure == "skip":
        value = observation(3)
    elif failure == "repeat":
        value = observation(1)
    elif failure == "changed_prefix":
        value = Observation(value.instruction, value.image, (
            HistoryEntry(step=1, action=ClickAction(x=0.5, y=0.5)), value.history[1],
        ))
    elif failure == "wrong_click":
        value = Observation(value.instruction, value.image, (
            value.history[0], HistoryEntry(step=2, action=ClickAction(x=0.5, y=0.5)),
        ))
    else:
        value = observation(2, instruction="Different task")
    with pytest.raises(PolicyOutputError):
        policy.predict(value)
    assert len(inference.requests) == 2
    assert policy.calls[-1]["error_type"] == "HistoryMismatch"


def test_first_observation_must_not_omit_previous_screenshots(inference):
    policy = GUIOwlPolicy()
    with pytest.raises(PolicyOutputError, match="initial observation"):
        policy.predict(observation(1))
    assert inference.requests == []


def test_code_checkpoint_inherits_prompt_and_history_settings(inference):
    instruction = observation().instruction
    code = "IF checkbox Oak is checked:\n    Click Red.\nEND IF\nClick Done."
    inference.metadata = {
        "history_images": 1, "max_length": 4096,
        "prompt_version": gui_owl_prompt.PROMPT_VERSION,
        "dataset": {"metadata": {
            "prompt_style": "code", "instruction_version": "scriptwob-code-v1",
        }},
    }
    mapping = {instruction: code}
    policy = GUIOwlPolicy(instruction_map=mapping)
    mapping[instruction] = "Tampered"
    policy.predict(observation())
    assert code in content_text(inference.prompts[-1])
    assert policy.calls[-1]["presented_instruction"] == code
    assert policy.history_images == 1 and policy.max_context_tokens == 4096
    assert policy.provenance()["instruction_version"] == "scriptwob-code-v1"
    assert policy.provenance()["prompt_style_overridden"] is False
    assert policy.provenance()["history_images_overridden"] is False
    assert not hasattr(policy, "world") and not hasattr(policy, "manifest")


def test_unknown_code_instruction_has_no_natural_fallback(inference):
    policy = GUIOwlPolicy(prompt_style="code", instruction_map={"known": "known code"})
    with pytest.raises(PolicyOutputError, match="static code prompt map"):
        policy.predict(observation())
    assert inference.requests == []


@pytest.mark.parametrize("kwargs", [
    {"history_images": -1}, {"history_images": 33}, {"history_images": True},
    {"max_context_tokens": 64}, {"max_new_tokens": 0}, {"max_context_tokens": 32769},
    {"prompt_style": "code"}, {"prompt_style": "code", "instruction_map": {"a": 42}},
])
def test_invalid_configuration_fails_before_model_load(inference, kwargs):
    with pytest.raises(ValueError):
        GUIOwlPolicy(**kwargs)
    assert inference.loads == []


def test_saved_prompt_version_mismatch_fails_before_model_load(inference):
    inference.metadata = {"prompt_version": "future-version"}
    with pytest.raises(ValueError, match="prompt version"):
        GUIOwlPolicy()
    assert inference.loads == []


def test_overflow_fails_without_truncating_or_generating(inference):
    policy = GUIOwlPolicy(max_context_tokens=65)
    with pytest.raises(PolicyOutputError, match="not truncated"):
        policy.predict(observation())
    assert inference.requests == []
    assert policy.calls[-1]["error_type"] == "ContextLimit"


def test_non_json_action_is_a_policy_failure(inference):
    inference.answer = "Thought: Click the red button. Action: click(25, 75)"
    policy = GUIOwlPolicy()
    with pytest.raises(PolicyOutputError):
        policy.predict(observation())
    assert policy.calls[-1]["error_type"] == "InvalidClickJSON"
    assert policy._images == []


def test_close_releases_model_and_visual_memory(inference):
    policy = GUIOwlPolicy()
    policy.predict(observation())
    policy.close()
    policy.close()
    assert policy.model is None and policy.processor is None and policy._images == []
    with pytest.raises(RuntimeError, match="closed"):
        policy.predict(observation())
