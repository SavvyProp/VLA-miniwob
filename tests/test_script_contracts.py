import pytest
from pydantic import ValidationError

from scriptwob.contracts import ClickAction


@pytest.mark.parametrize("x", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_coordinates_are_rejected_without_silent_clipping(x):
    with pytest.raises(ValidationError):
        ClickAction(x=x, y=0.5)


def test_coordinates_use_actual_screenshot_extent():
    assert ClickAction(x=1, y=1).pixels(1000, 760) == (999, 759)
    assert ClickAction(x=0, y=0).pixels(1000, 760) == (0, 0)
