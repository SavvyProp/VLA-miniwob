import numpy as np
import pytest
from pydantic import ValidationError

from vla_miniwob.contracts import MouseAction, decode
from vla_miniwob.tasks import Manifest, make_manifest
from vla_miniwob.validation import Event, OrderValidator


def test_click_event_threshold_and_clip():
    first = decode(MouseAction(x=-2, y=4, click=0.5), 720, 540)
    assert (first.pixel_x, first.pixel_y, first.kind) == (0, 539, "click")
    assert first.clipped
    assert decode(first.action, 720, 540).kind == "click"  # no toggle/held button semantics
    assert decode(MouseAction(x=0.5, y=0.5, click=0.499), 720, 540).kind == "move"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_actions_rejected(value):
    with pytest.raises(ValidationError):
        MouseAction(x=value, y=0, click=1)


def test_order_violation_survives_final_goal_completion():
    validator = OrderValidator([0, 1])
    validator.ingest(Event(component=1, kind="progress", step=1))
    validator.ingest(Event(component=0, kind="complete", step=2))
    validator.ingest(Event(component=1, kind="complete", step=3))
    result = validator.verdict(3, 3, True)
    assert result.goals_valid and result.budget_valid
    assert not result.success and not result.order_valid
    assert result.first_violation.step == 1


def test_budget_last_action_is_valid_but_over_budget_is_not():
    validator = OrderValidator([1, 0])
    validator.ingest(Event(component=1, kind="complete", step=1))
    validator.ingest(Event(component=0, kind="complete", step=2))
    assert validator.verdict(2, 2, True).success
    assert not validator.verdict(3, 2, True).success
    assert not validator.verdict(2, 2, False).success


def test_manifest_pairs_ui_and_separates_splits():
    manifest = make_manifest(2, 1, 1, tasks=["click-button_click-link"])
    assert manifest.digest == make_manifest(2, 1, 1, tasks=["click-button_click-link"]).digest
    first = manifest.instances[:4]
    assert len({(x.values_seed, x.layout_seed) for x in first}) == 1
    assert first[0].required_order == [0, 1]
    assert first[2].required_order == [1, 0]
    assert len({x.values_seed for x in manifest.instances}) == 4
    altered = manifest.model_dump()
    altered["instances"][1]["split"] = "eval"
    with pytest.raises(ValidationError, match="leakage"):
        Manifest.model_validate(altered)


def test_typing_tasks_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        make_manifest(1, 0, 0, tasks=["enter-text"])
