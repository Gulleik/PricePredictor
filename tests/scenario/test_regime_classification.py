"""Scenario tests for market regime tagging."""

import pandas as pd
import pytest

from src.models.regime import classify_regimes

pytestmark = pytest.mark.scenario


def test_classify_regimes_bull_tail() -> None:
    """A steadily rising series should end in bull regime."""
    price = pd.Series([float(100 + i) for i in range(240)])

    regimes = classify_regimes(
        price,
        fast_window=20,
        slow_window=100,
        sideways_band=0.0,
    )

    assert regimes.iloc[-1] == "bull"


def test_classify_regimes_bear_tail() -> None:
    """A steadily falling series should end in bear regime."""
    price = pd.Series([float(400 - i) for i in range(240)])

    regimes = classify_regimes(
        price,
        fast_window=20,
        slow_window=100,
        sideways_band=0.0,
    )

    assert regimes.iloc[-1] == "bear"


def test_classify_regimes_sideways_band() -> None:
    """Small oscillations should remain sideways with a non-zero neutral band."""
    values = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(260)]
    price = pd.Series(values)

    regimes = classify_regimes(
        price,
        fast_window=20,
        slow_window=100,
        sideways_band=0.01,
    )

    assert regimes.iloc[-1] == "sideways"


def test_classify_regimes_validation() -> None:
    """Invalid regime settings should raise ValueError."""
    price = pd.Series([100.0, 101.0, 102.0, 103.0])

    with pytest.raises(ValueError, match="positive"):
        classify_regimes(price, fast_window=0, slow_window=3, sideways_band=0.01)

    with pytest.raises(ValueError, match="smaller"):
        classify_regimes(price, fast_window=3, slow_window=3, sideways_band=0.01)

    with pytest.raises(ValueError, match="non-negative"):
        classify_regimes(price, fast_window=1, slow_window=3, sideways_band=-0.01)
