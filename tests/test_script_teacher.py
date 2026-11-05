"""Expert prompting must remain public-only, bounded, and auditable."""

from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, AuthenticationError, OpenAI
from PIL import Image

from scriptwob.contracts import ClickAction, HistoryEntry, Observation
from scriptwob.policies import OpenAITeacher, OraclePolicy, PolicyOutputError, TeacherConfig


class Responses:
    def __init__(self, *, status="completed", action=None, error=None, retries_taken=0):
        self.requests = []
        self.status = status
        self.action = action
        self.error = error
        self.with_raw_response = self
        self.retries_taken = retries_taken

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        response = SimpleNamespace(
            id="resp_mock", model="gpt-6-astra", status=self.status,
            usage=SimpleNamespace(model_dump=lambda **_: {
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            }), output_parsed=self.action,
        )
        return SimpleNamespace(retries_taken=self.retries_taken, parse=lambda: response)


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
    assert teacher.calls[0]["usage_complete"] is True
    assert teacher.calls[0]["transport_attempts"] == 1
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
    assert teacher.calls[0]["cause_type"] is None
    assert teacher.calls[0]["transport_retries_taken"] is None
    assert teacher.calls[0]["transport_attempts"] is None
    assert teacher.calls[0]["usage_complete"] is False
    assert "secret" not in str(teacher.calls)


def successful_response():
    return httpx2.Response(200, json={
        "id": "resp_mock", "object": "response", "created_at": 0,
        "model": "gpt-6-astra", "status": "completed",
        "output": [{"type": "message", "id": "msg_mock", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                    "text": '{"x":0.25,"y":0.75}', "annotations": []}]}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    })


def sdk_teacher(handler, monkeypatch, *, retries=3):
    # Exercise the real SDK request/retry path without network traffic or delays.
    monkeypatch.setattr("openai._base_client.time.sleep", lambda _: None)
    client = OpenAI(
        api_key="secret-api-key", base_url="https://provider.test/secret-endpoint",
        max_retries=retries, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    return OpenAITeacher(TeacherConfig(transport_retries=retries), client=client)


@pytest.mark.parametrize("failure", ["connection", "timeout", 429, 500])
def test_sdk_retries_temporary_failure_without_another_action(monkeypatch, failure):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "connection":
                raise httpx2.ConnectError("connection to secret-endpoint failed", request=request)
            if failure == "timeout":
                raise httpx2.ReadTimeout("timeout at secret-endpoint", request=request)
            return httpx2.Response(failure, json={"error": {"message": "secret error"}})
        return successful_response()

    teacher = sdk_teacher(handler, monkeypatch)
    try:
        assert teacher.predict(observation()) == ClickAction(x=0.25, y=0.75)
        assert len(requests) == 2
        assert requests[0].content == requests[1].content
        assert [request.headers["x-stainless-retry-count"] for request in requests] == ["0", "1"]
        assert len(teacher.calls) == 1
        call = teacher.calls[0]
        assert call["transport_retries_limit"] == 3
        assert call["transport_retries_taken"] == 1
        assert call["transport_attempts"] == 2
        assert call["usage"]["total_tokens"] == 120
        assert call["usage_complete"] is False
        assert "secret" not in str(teacher.calls)
        teacher.reset(17)
        teacher.predict(observation())
        assert len(teacher.calls) == 1
        assert teacher.calls[0]["call"] == 1
        assert teacher.calls[0]["transport_retries_taken"] == 0
        assert teacher.calls[0]["transport_attempts"] == 1
        assert teacher.calls[0]["usage_complete"] is True
    finally:
        teacher.client.close()


@pytest.mark.parametrize("retries", [0, 3])
def test_sdk_connection_failure_retries_are_bounded_and_sanitized(monkeypatch, retries):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx2.ConnectError("credentials=secret-api-key", request=request)

    teacher = sdk_teacher(handler, monkeypatch, retries=retries)
    try:
        with pytest.raises(APIConnectionError):
            teacher.predict(observation())
        assert len(requests) == retries + 1
        assert len(teacher.calls) == 1
        call = teacher.calls[0]
        assert call["error_type"] == "APIConnectionError"
        assert call["cause_type"] == "ConnectError"
        assert call["transport_retries_taken"] == retries
        assert call["transport_attempts"] == retries + 1
        assert call["usage_complete"] is False
        assert "secret" not in str(teacher.calls)
    finally:
        teacher.client.close()


def test_sdk_authentication_error_is_not_retried(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(401, json={"error": {"message": "secret-api-key is invalid"}})

    teacher = sdk_teacher(handler, monkeypatch)
    try:
        with pytest.raises(AuthenticationError):
            teacher.predict(observation())
        assert len(requests) == 1
        assert teacher.calls[0]["transport_retries_taken"] == 0
        assert teacher.calls[0]["transport_attempts"] == 1
        assert teacher.calls[0]["usage_complete"] is False
        assert "secret" not in str(teacher.calls)
    finally:
        teacher.client.close()


@pytest.mark.parametrize("value", [None, "secret-headers", "-1", "1.5"])
def test_unknown_retry_metadata_is_not_stored_as_complete_usage(value):
    client = SimpleNamespace(responses=Responses(action=ClickAction(x=0.25, y=0.75),
                                                retries_taken=value))
    teacher = OpenAITeacher(client=client)
    teacher.predict(observation())
    assert teacher.calls[0]["transport_retries_taken"] is None
    assert teacher.calls[0]["transport_attempts"] is None
    assert teacher.calls[0]["usage_complete"] is False
    assert "secret" not in str(teacher.calls)


@pytest.mark.parametrize("headers", [None, {}, {"x-stainless-retry-count": "secret-header"}])
def test_failure_with_unknown_retry_count_preserves_original_exception(headers):
    error = APIConnectionError(request=SimpleNamespace(headers=headers))
    teacher = OpenAITeacher(client=SimpleNamespace(responses=Responses(error=error)))
    with pytest.raises(APIConnectionError) as caught:
        teacher.predict(observation())
    assert caught.value is error
    assert teacher.calls[0]["transport_retries_taken"] is None
    assert teacher.calls[0]["transport_attempts"] is None
    assert teacher.calls[0]["usage_complete"] is False
    assert "secret" not in str(teacher.calls)


def test_default_teacher_uses_three_sdk_retries(monkeypatch):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("scriptwob.policies.OpenAI", factory)
    teacher = OpenAITeacher()
    assert teacher.config.transport_retries == 3
    assert captured["max_retries"] == 3


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
