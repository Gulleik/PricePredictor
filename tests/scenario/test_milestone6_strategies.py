"""Scenario smoke tests for Milestone 6 strategy modules."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion import run as mean_reversion_run
from src.strategies.orb import run as orb_run
from src.strategies.trend_following import run as trend_following_run
from src.strategies.volatility_breakout import run as breakout_run

pytestmark = pytest.mark.scenario


def _build_ohlc(n: int = 96) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    base = pd.Series(100 + np.sin(np.linspace(0, 12, n)) * 3, index=idx)
    high = base + 0.8
    low = base - 0.8
    return base, high, low


def test_mean_reversion_runs_with_next_bar_and_friction() -> None:
    """Mean reversion should execute with common portfolio kwargs enabled."""
    price, _, _ = _build_ohlc()
    pf, rsi, upper, lower = mean_reversion_run(
        price,
        next_bar_execution=True,
        fees=0.001,
        fixed_fees=1.0,
        slippage=0.001,
    )
    assert pf is not None
    assert len(rsi) == len(price)
    assert len(upper) == len(price)
    assert len(lower) == len(price)


def test_trend_following_runs_with_ohlc_inputs() -> None:
    """Trend-following should accept explicit high/low series and execute."""
    price, high, low = _build_ohlc()
    pf, fast_ema, slow_ema, atr, trail = trend_following_run(
        price,
        high=high,
        low=low,
        next_bar_execution=True,
    )
    assert pf is not None
    assert fast_ema.index.equals(price.index)
    assert slow_ema.index.equals(price.index)
    assert atr.index.equals(price.index)
    assert trail.index.equals(price.index)


def test_volatility_breakout_runs_with_donchian_filter() -> None:
    """Donchian breakout strategy should execute with ATR filter enabled."""
    price, high, low = _build_ohlc()
    pf, upper, lower, atr = breakout_run(
        price,
        high,
        low,
        use_atr_filter=True,
        next_bar_execution=True,
    )
    assert pf is not None
    assert upper.index.equals(price.index)
    assert lower.index.equals(price.index)
    assert atr.index.equals(price.index)


def test_orb_runs_with_utc_daily_sessions() -> None:
    """ORB should execute using UTC day boundaries for opening range."""
    price, high, low = _build_ohlc()
    pf, range_high, range_low = orb_run(
        price,
        high,
        low,
        range_bars=3,
        breakout_buffer=0.001,
        allow_short=True,
        next_bar_execution=True,
    )
    assert pf is not None
    assert range_high.index.equals(price.index)
    assert range_low.index.equals(price.index)
