"""Market regime tagging helpers."""

import pandas as pd


def classify_regimes(
    price: pd.Series,
    *,
    fast_window: int,
    slow_window: int,
    sideways_band: float,
) -> pd.Series:
    """Label each bar as bull, bear, or sideways using moving averages."""
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("fast_window and slow_window must be positive")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")
    if sideways_band < 0:
        raise ValueError("sideways_band must be non-negative")

    fast_ma = price.rolling(window=fast_window, min_periods=fast_window).mean()
    slow_ma = price.rolling(window=slow_window, min_periods=slow_window).mean()

    relative_gap = (fast_ma - slow_ma) / slow_ma

    regime = pd.Series("sideways", index=price.index, dtype="object")
    regime[relative_gap > sideways_band] = "bull"
    regime[relative_gap < -sideways_band] = "bear"

    return regime
