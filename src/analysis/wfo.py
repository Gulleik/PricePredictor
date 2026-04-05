"""Walk-forward optimization window helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WFOWindow:
    """A single walk-forward split represented by iloc boundaries."""

    is_start: int
    is_end: int
    oos_start: int
    oos_end: int


def generate_wfo_windows(
    n_bars: int,
    *,
    is_window_bars: int,
    oos_window_bars: int,
    step_bars: int,
) -> list[WFOWindow]:
    """Build rolling in-sample/out-of-sample windows.

    Args:
        n_bars: Total number of bars in the time series.
        is_window_bars: Number of bars used for parameter fitting.
        oos_window_bars: Number of bars used for OOS evaluation.
        step_bars: Number of bars to roll windows each iteration.

    Returns:
        Ordered list of windows with half-open iloc boundaries.
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    if is_window_bars <= 1:
        raise ValueError("is_window_bars must be greater than 1")
    if oos_window_bars <= 1:
        raise ValueError("oos_window_bars must be greater than 1")
    if step_bars <= 0:
        raise ValueError("step_bars must be positive")

    windows: list[WFOWindow] = []
    is_start = 0

    while True:
        is_end = is_start + is_window_bars
        oos_start = is_end
        oos_end = oos_start + oos_window_bars

        if oos_end > n_bars:
            break

        windows.append(
            WFOWindow(
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
        is_start += step_bars

    return windows
