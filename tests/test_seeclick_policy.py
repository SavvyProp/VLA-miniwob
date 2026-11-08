"""Public-only inference, immutable checkpoint provenance, and strict actions."""

import hashlib
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import scriptwob
from scriptwob.contracts import ClickAction, HistoryEntry, Observation
from scriptwob.policies import PolicyOutputError
from scriptwob.seeclick_policy import SeeClickPolicy, checkpoint_hashes, parse_click


@pytest.mark.parametrize("text", [
    '{"action_type":4,"click_point":[0,1]}',
    '```json\n{"action_type":4,"click_point":[0,1]}\n```',
])
def test_normalized_click_accepts_json_and_single_fence(text):
    assert parse_click(text) == ClickAction(x=0, y=1)


@pytest.mark.parametrize("text", [
    "{'action_type':4,'click_point':[0.5,0.5]}",
    '{"action_type":4,"click_point":[NaN,0]}',
    '{"action_type":4,"click_point":[Infinity,0]}',
    '{"action_type":4,"click_point":[true,0]}',
    '{"action_type":4,"click_point":["0.2",0]}',
    '{"action_type":4,"click_point":[-0.1,0]}',
    '{"action_type":4,"click_point":[0,1.1]}',
    '{"action_type":4,"click_point":[0,0,0]}',
    '{"action_type":4.0,"click_point":[0,0]}',
    '{"action_type":3,"click_point":[0,0]}',
    '{"action_type":4,"click_point":[0,0],"reasoning":"extra"}',
    '{"action_type":3,"action_type":4,"click_point":[0,0]}',
    'Here is the click: {"action_type":4,"click_point":[0,0]}',
    '```json\n{"action_type":4,"click_point":[0,0]}\n```\nExtra',
    '[{"action_type":4,"click_point":[0,0]}]',
    '{"action_type":4,"click_point":[0,0]} {}',
    '__import__("os").system("echo unsafe")',
    "", None,
])
def test_invalid_output_never_becomes_a_click(text):
    with pytest.raises(PolicyOutputError):
        parse_click(text)


class Tokenizer:
    im_start_id = 1000
    im_end_id = 1001
    eod_id = 1002

    def __init__(self):
        self.prompts = []
        self.decoded = []

    def __call__(self, text, **kwargs):
        if "Picture 1:" in text:
            self.prompts.append(text)
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, tokens, *, skip_special_tokens):
        self.decoded.append(list(tokens))
        assert skip_special_tokens
        return "".join(chr(token) for token in tokens if token < 1000)


class Model:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = []
        self.eval_count = 0
        self.answer = '{"action_type":4,"click_point":[0.25,0.75]}'
        self.error = None

    def eval(self):
        self.eval_count += 1

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(device="cuda:7"))

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        # The image must still exist at the exact time Qwen-VL reads the prompt.
        image = Path(re.search(r"<img>(.*?)</img>", self.tokenizer.prompts[-1]).group(1))
        assert image.is_file()
        assert Image.open(image).mode == "RGB"
        if self.error:
            raise self.error
        return [kwargs["input_ids"][0] + [ord(char) for char in self.answer] + [1001]]


@pytest.fixture
def inference(monkeypatch):
    tokenizer = Tokenizer()
    model = Model(tokenizer)
    resolution = {"adapter": None, "metadata": {}}
    config = SimpleNamespace(model_dump=lambda **kwargs: {
        "model_id": "cckevinn/SeeClick", "model_revision": "pinned", "device": "cuda:7",
    })
    loads = []

    def load_model(*args, **kwargs):
        loads.append((args, kwargs))
        return model

    runtime = SimpleNamespace(
        resolve_checkpoint=lambda *args, **kwargs: (
            config, resolution["adapter"], resolution["metadata"],
        ),
        load_tokenizer=lambda *args, **kwargs: tokenizer,
        load_model=load_model,
    )
    monkeypatch.setitem(sys.modules, "scriptwob.seeclick_runtime", runtime)
    monkeypatch.setattr(scriptwob, "seeclick_runtime", runtime, raising=False)
    transfers = []

    def tensor(values, **kwargs):
        transfers.append(kwargs)
        return values

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        tensor=tensor, long="long", inference_mode=nullcontext,
    ))
    return SimpleNamespace(
        tokenizer=tokenizer, model=model, resolution=resolution, loads=loads, transfers=transfers,
    )


def observation():
    return Observation(
        "If the visible counter is 2, click Oak. Then click Done.",
        Image.new("RGBA", (30, 20), "white"),
        (HistoryEntry(step=1, action=ClickAction(x=0.1, y=0.2), feedback="clicked Oak"),),
    )


def test_policy_uses_shared_full_public_prompt_and_only_decodes_new_tokens(inference):
    policy = SeeClickPolicy()
    assert policy.predict(observation()) == ClickAction(x=0.25, y=0.75)
    assert policy.predict(observation()) == ClickAction(x=0.25, y=0.75)
    assert len(inference.loads) == 1
    assert inference.model.eval_count == 1
    prompt = inference.tokenizer.prompts[-1]
    assert observation().instruction in prompt
    assert "1: (0.100000, 0.200000) clicked Oak" in prompt
    assert all(private not in prompt for private in ["oracle", "expected_action", "program"])
    assert inference.tokenizer.decoded[-1] == [ord(c) for c in inference.model.answer] + [1001]
    request = inference.model.requests[-1]
    assert request["do_sample"] is False
    assert request["num_beams"] == 1
    assert request["use_cache"] is True
    assert request["max_new_tokens"] == 64
    assert request["eos_token_id"] == [1001, 1002]
    assert len(request["attention_mask"][0]) == len(request["input_ids"][0])
    assert all(transfer == {"dtype": "long", "device": "cuda:7"}
               for transfer in inference.transfers)
    assert len(policy.calls) == 2
    assert policy.calls[-1]["status"] == "completed"
    assert policy.calls[-1]["presented_instruction"] == observation().instruction
    assert policy.provenance()["prompt_style"] == "natural"
    assert policy.provenance()["instruction_version"] == "scriptwob-natural-v1"
    assert policy.provenance()["prompt_style_overridden"] is False
    assert policy.provenance()["instruction_map_sha256"] is None
    assert not Path(re.search(r"<img>(.*?)</img>", prompt).group(1)).exists()


def test_code_checkpoint_infers_style_and_uses_shared_prompt_without_hidden_state(inference):
    from scriptwob.generation import generate_manifest
    from scriptwob.prompt_styles import build_instruction_map, instruction_version
    from scriptwob.seeclick_prompt import PROMPT_VERSION, build_user_prompt

    manifest = generate_manifest(train=1, dev=0, test=0, levels=[2])
    episode = manifest.episodes[0]
    mapping = build_instruction_map(manifest, "code")
    presented = mapping[episode.instruction]
    inference.resolution["metadata"] = {"dataset": {"metadata": {
        "prompt_style": "code", "instruction_version": instruction_version("code"),
    }}}
    policy = SeeClickPolicy(instruction_map=mapping, max_context_tokens=8192)
    public = Observation(episode.instruction, observation().image, observation().history)
    policy.predict(public)
    prompt = inference.tokenizer.prompts[-1]
    image_path = re.search(r"<img>(.*?)</img>", prompt).group(1)
    # Export uses the same renderer and shared wrapper; no inference-only hints.
    assert prompt == build_user_prompt(presented, public.history, image_path)
    assert presented != episode.instruction
    assert policy.calls[-1]["presented_instruction"] == presented
    assert not hasattr(policy, "manifest")
    assert not hasattr(policy, "program")
    assert not hasattr(policy, "world")
    provenance = policy.provenance()
    assert provenance["prompt_version"] == PROMPT_VERSION
    assert provenance["prompt_style"] == "code"
    assert provenance["trained_prompt_style"] == "code"
    assert provenance["prompt_style_overridden"] is False
    assert provenance["instruction_version"] == instruction_version("code")


def test_explicit_code_ablation_defensively_copies_map_and_records_override(inference):
    original = observation().instruction
    code = 'IF the counter value equals 2:\n    Click "Oak".\nEND IF\nClick Done.'
    mapping = {original: code, "Click Done.": "Click Done."}
    policy = SeeClickPolicy(prompt_style="code", instruction_map=mapping)
    expected_digest = hashlib.sha256(json.dumps(
        mapping, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    mapping[original] = "Tampered instruction"
    policy.predict(observation())
    assert code in inference.tokenizer.prompts[-1]
    assert "Tampered" not in inference.tokenizer.prompts[-1]
    provenance = policy.provenance()
    assert provenance["prompt_style_overridden"] is True
    assert provenance["trained_prompt_style"] == "natural"
    assert provenance["instruction_map_sha256"] == expected_digest
    provenance["instruction_map_sha256"] = "tampered"
    assert policy.provenance()["instruction_map_sha256"] == expected_digest


def test_explicit_natural_override_ignores_code_map(inference):
    inference.resolution["metadata"] = {"dataset": {"metadata": {
        "prompt_style": "code", "instruction_version": "scriptwob-code-v1",
    }}}
    mapping = {observation().instruction: "Do something different"}
    policy = SeeClickPolicy(prompt_style="natural", instruction_map=mapping)
    policy.predict(observation())
    assert observation().instruction in inference.tokenizer.prompts[-1]
    assert "Do something different" not in inference.tokenizer.prompts[-1]
    assert mapping == {observation().instruction: "Do something different"}
    assert policy.provenance()["prompt_style_overridden"] is True
    assert policy.provenance()["instruction_map_sha256"] is None


@pytest.mark.parametrize("mapping", [None, {}, {"source": 42}, {42: "code"}, {"": "code"},
                                    {"source": " "}, {"source": {"program": []}}])
def test_invalid_code_map_fails_before_loading_model(inference, mapping):
    with pytest.raises(ValueError, match="instruction_map"):
        SeeClickPolicy(prompt_style="code", instruction_map=mapping)
    assert inference.loads == []


def test_code_checkpoint_requires_map_even_when_style_is_inferred(inference):
    inference.resolution["metadata"] = {"dataset": {"metadata": {
        "prompt_style": "code", "instruction_version": "scriptwob-code-v1",
    }}}
    with pytest.raises(ValueError, match="instruction_map"):
        SeeClickPolicy()
    assert inference.loads == []


def test_code_checkpoint_requires_instruction_version(inference):
    inference.resolution["metadata"] = {"dataset": {"metadata": {"prompt_style": "code"}}}
    with pytest.raises(ValueError, match="missing instruction_version"):
        SeeClickPolicy(instruction_map={"Known": "Known code"})
    assert inference.loads == []


def test_code_style_unknown_instruction_fails_without_original_prompt_fallback(inference):
    policy = SeeClickPolicy(prompt_style="code", instruction_map={"Known": "Known code"})
    with pytest.raises(PolicyOutputError, match="missing from the static code prompt map"):
        policy.predict(observation())
    assert inference.model.requests == []
    assert inference.tokenizer.prompts == []
    assert policy.calls == [{
        "call": 1, "status": "invalid_output", "error_type": "UnknownInstruction",
    }]


def test_instruction_version_mismatch_is_rejected_before_loading(inference):
    inference.resolution["metadata"] = {"dataset": {"metadata": {
        "prompt_style": "code", "instruction_version": "scriptwob-code-future",
    }}}
    with pytest.raises(ValueError, match="instruction version differs"):
        SeeClickPolicy(instruction_map={"Known": "Known code"})
    assert inference.loads == []


@pytest.mark.parametrize("explicit", [True, False])
def test_unknown_prompt_style_fails_before_loading(inference, explicit):
    kwargs = {"prompt_style": "unrecognized"} if explicit else {}
    if not explicit:
        inference.resolution["metadata"] = {
            "dataset": {"metadata": {"prompt_style": "unrecognized"}},
        }
    with pytest.raises(ValueError):
        SeeClickPolicy(**kwargs)
    assert inference.loads == []


def test_reset_retains_model_but_removes_episode_calls(inference):
    policy = SeeClickPolicy()
    policy.predict(observation())
    policy.reset(123)
    assert policy.calls == []
    policy.predict(Observation("Click Done.", observation().image))
    assert "(none)" in inference.tokenizer.prompts[-1]
    assert "clicked Oak" not in inference.tokenizer.prompts[-1]
    assert len(inference.loads) == 1


def test_context_overflow_is_policy_failure_without_generation_or_truncation(inference):
    policy = SeeClickPolicy(max_context_tokens=100, max_new_tokens=64)
    with pytest.raises(PolicyOutputError, match="not truncated"):
        policy.predict(observation())
    assert inference.model.requests == []
    assert policy.calls[-1]["error_type"] == "ContextLimit"
    prompt = inference.tokenizer.prompts[-1]
    assert observation().instruction in prompt
    assert not Path(re.search(r"<img>(.*?)</img>", prompt).group(1)).exists()


@pytest.mark.parametrize("failure", ["invalid_json", "generation_error"])
def test_screenshot_cleanup_even_when_policy_or_runtime_fails(inference, failure):
    policy = SeeClickPolicy()
    if failure == "invalid_json":
        inference.model.answer = "not an action"
        error = PolicyOutputError
    else:
        inference.model.error = RuntimeError("fake GPU failure")
        error = RuntimeError
    with pytest.raises(error):
        policy.predict(observation())
    path = Path(re.search(r"<img>(.*?)</img>", inference.tokenizer.prompts[-1]).group(1))
    assert not path.parent.exists()


def test_close_releases_model_and_is_idempotent(inference):
    policy = SeeClickPolicy()
    policy.close()
    policy.close()
    assert policy.model is None and policy.tokenizer is None
    with pytest.raises(RuntimeError, match="closed"):
        policy.predict(observation())


def test_checkpoint_hash_validation_and_frozen_provenance(inference, tmp_path):
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"small fake adapter")
    checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
    inference.resolution.update(adapter=tmp_path, metadata={
        "max_length": 1024, "training_fingerprint": "training-run",
        "adapter_hashes": {weights.name: checksum},
    })
    policy = SeeClickPolicy(tmp_path)
    assert policy.max_context_tokens == 1024
    provenance = policy.provenance()
    assert provenance["checkpoint_files_sha256"][weights.name] == checksum
    provenance["checkpoint_files_sha256"][weights.name] = "tampered copy"
    assert policy.provenance()["checkpoint_files_sha256"][weights.name] == checksum
    weights.write_bytes(b"changed fake adapter")
    assert policy.provenance()["checkpoint_files_sha256"][weights.name] == checksum
    with pytest.raises(ValueError, match="checksum mismatch"):
        SeeClickPolicy(tmp_path)
    assert len(inference.loads) == 1


def test_native_checkpoint_requires_weight_hashes(inference, tmp_path):
    inference.resolution.update(adapter=tmp_path)
    with pytest.raises(ValueError, match="missing adapter_hashes"):
        SeeClickPolicy(tmp_path)
    assert inference.loads == []


def test_artifact_fingerprint_covers_config_and_weights_not_training_state(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (tmp_path / "adapter_model.bin").write_bytes(b"weights")
    (tmp_path / "optimizer.pt").write_bytes(b"optimizer")
    (tmp_path / "README.md").write_text("documentation")
    assert set(checkpoint_hashes(tmp_path)) == {"adapter_config.json", "adapter_model.bin"}


@pytest.mark.parametrize("kwargs", [
    {"max_context_tokens": 0}, {"max_new_tokens": 0},
    {"max_context_tokens": 64, "max_new_tokens": 64},
])
def test_invalid_generation_limits_fail_before_loading(inference, kwargs):
    with pytest.raises(ValueError, match="Token limits"):
        SeeClickPolicy(**kwargs)
    assert inference.loads == []
