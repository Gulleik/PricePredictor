"""Scenario tests for walk-forward optimization windowing."""

import pytest

from src.analysis.wfo import generate_wfo_windows

pytestmark = pytest.mark.scenario


def test_generate_wfo_windows_basic_layout() -> None:
    """WFO windows should be ordered, non-overlapping IS/OOS pairs."""
    windows = generate_wfo_windows(
        n_bars=30,
        is_window_bars=10,
        oos_window_bars=5,
        step_bars=5,
    )

    assert len(windows) == 4
    first = windows[0]
    assert first.is_start == 0
    assert first.is_end == 10
    assert first.oos_start == 10
    assert first.oos_end == 15

    for window in windows:
        assert window.is_end == window.oos_start
        assert window.is_start < window.is_end < window.oos_end


def test_generate_wfo_windows_respects_step() -> None:
    """Consecutive windows should roll by configured step size."""
    windows = generate_wfo_windows(
        n_bars=40,
        is_window_bars=12,
        oos_window_bars=6,
        step_bars=4,
    )

    assert len(windows) >= 2
    assert windows[1].is_start - windows[0].is_start == 4


def test_generate_wfo_windows_input_validation() -> None:
    """Invalid WFO inputs should fail fast with clear errors."""
    with pytest.raises(ValueError, match="n_bars"):
        generate_wfo_windows(
            n_bars=0,
            is_window_bars=10,
            oos_window_bars=5,
            step_bars=5,
        )

    with pytest.raises(ValueError, match="is_window_bars"):
        generate_wfo_windows(
            n_bars=20,
            is_window_bars=1,
            oos_window_bars=5,
            step_bars=5,
        )

    with pytest.raises(ValueError, match="oos_window_bars"):
        generate_wfo_windows(
            n_bars=20,
            is_window_bars=10,
            oos_window_bars=1,
            step_bars=5,
        )

    with pytest.raises(ValueError, match="step_bars"):
        generate_wfo_windows(
            n_bars=20,
            is_window_bars=10,
            oos_window_bars=5,
            step_bars=0,
        )
