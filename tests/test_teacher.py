from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vla_miniwob.contracts import HistoryEntry, MouseAction, PolicyObservation
from vla_miniwob.policies import OpenAITeacher, PolicyOutputError, TeacherConfig


class FakeResponses:
    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="fixture",
            model="gpt-6-astra",
            status="completed",
            usage=None,
            output_parsed=MouseAction(x=0.2, y=0.3, click=1),
        )


def observation():
    return PolicyObservation(
        "A before B",
        Image.new("RGB", (72, 54)),
        (0.5, 0.5, 0),
        (HistoryEntry(step=1, action=MouseAction(x=0.5, y=0.5, click=1), feedback="executed"),),
    )


def test_teacher_request_uses_astra_structured_vision_and_literal_history():
    responses = FakeResponses()
    teacher = OpenAITeacher(client=SimpleNamespace(responses=responses))
    action = teacher.predict(observation())
    np.testing.assert_allclose(action, [[0.2, 0.3, 1]])
    request = responses.request
    assert request["model"] == "gpt-6-astra"
    assert request["text_format"] is MouseAction
    assert request["store"] is False
    assert "temperature" not in request
    content = request["input"][0]["content"]
    assert "PROCEDURE\nA before B" in content[0]["text"]
    assert "1: (0.500000, 0.500000, 1) executed" in content[0]["text"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert content[1]["detail"] == "high"
    assert teacher.calls[0]["response_id"] == "fixture"


def test_context_overflow_fails_without_silent_truncation():
    teacher = OpenAITeacher(TeacherConfig(max_context_chars=2), client=object())
    with pytest.raises(PolicyOutputError, match="not truncated"):
        teacher.predict(observation())
