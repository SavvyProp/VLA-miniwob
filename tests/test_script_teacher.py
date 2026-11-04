"""Expert prompting must remain public-only, bounded, and auditable."""

from types import SimpleNamespace

import pytest
from openai import APIConnectionError
from PIL import Image

from scriptwob.contracts import ClickAction, HistoryEntry, Observation
from scriptwob.policies import OpenAITeacher, OraclePolicy, PolicyOutputError, TeacherConfig


class Responses:
    def __init__(self, *, status="completed", action=None, error=None):
        self.requests = []
        self.status = status
        self.action = action
        self.error = error

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            id="resp_mock", model="gpt-6-astra", status=self.status,
            usage=SimpleNamespace(model_dump=lambda **_: {
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            }), output_parsed=self.action,
        )


def observation(history=()):
    return Observation("Until the counter is 3, click Increase. Then Submit.",
                       Image.new("RGB", (20, 20), "white"), history)


def test_frontier_teacher_sends_complete_public_context_and_one_image():
    client = SimpleNamespace(responses=Responses(action=ClickAction(x=0.25, y=0.75)))
    teacher = OpenAITeacher(client=client)
    history = (HistoryEntry(step=1, action=ClickAction(x=0.123, y=0.456),
                            feedback="clicked Increase"),)
    assert teacher.predict(observation(history)) == ClickAction(x=0.25, y=0.75)
    request = client.responses.requests[0]
    assert request["model"] == "gpt-6-astra"
    assert request["text_format"] is ClickAction
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "medium"}
    assert "previous_response_id" not in request
    content = request["input"][0]["content"]
    assert observation().instruction in content[0]["text"]
    assert "1: (0.123000, 0.456000) clicked Increase" in content[0]["text"]
    assert "program" not in content[0]["text"]
    assert content[1]["detail"] == "high"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert teacher.calls[0]["usage"]["total_tokens"] == 120
    assert teacher.provenance()["supervision"] == "frontier"
    teacher.reset(30)
    assert teacher.calls == []


@pytest.mark.parametrize("status", ["incomplete", "completed"])
def test_refusal_or_incomplete_action_is_not_executed(status):
    client = SimpleNamespace(responses=Responses(status=status, action=None))
    teacher = OpenAITeacher(client=client)
    with pytest.raises(PolicyOutputError, match="refused or returned an incomplete"):
        teacher.predict(observation())
    assert len(teacher.calls) == 1
    assert teacher.calls[0]["status"] == status


def test_provider_error_metadata_is_recorded_without_credentials():
    error = APIConnectionError(request=SimpleNamespace(url="https://example.test/secret"))
    client = SimpleNamespace(responses=Responses(error=error))
    teacher = OpenAITeacher(client=client)
    with pytest.raises(APIConnectionError):
        teacher.predict(observation())
    assert teacher.calls[0]["error_type"] == "APIConnectionError"
    assert "secret" not in str(teacher.calls)


def test_context_limit_refuses_without_silent_truncation_or_provider_call():
    client = SimpleNamespace(responses=Responses())
    teacher = OpenAITeacher(TeacherConfig(max_context_chars=5), client=client)
    with pytest.raises(PolicyOutputError, match="not truncated"):
        teacher.predict(observation())
    assert client.responses.requests == []
    assert teacher.calls == []


def test_oracle_is_explicitly_privileged_and_requires_environment():
    oracle = OraclePolicy()
    assert oracle.provenance()["supervision"] == "privileged_oracle"
    with pytest.raises(RuntimeError, match="bound"):
        oracle.predict(observation())
    oracle.bind_environment(SimpleNamespace(oracle_action=lambda: ClickAction(x=0.2, y=0.8)))
    assert oracle.predict(observation()) == ClickAction(x=0.2, y=0.8)


@pytest.mark.parametrize("kwargs", [
    {"model": ""}, {"transport_retries": -1}, {"transport_retries": 4},
    {"timeout_seconds": 0}, {"max_output_tokens": 0}, {"reasoning_effort": "typo"},
])
def test_invalid_teacher_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TeacherConfig(**kwargs)
