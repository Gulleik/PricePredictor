"""Scenario tests for vectorized scalping strategies."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.vectorbt_scalping import (
    bb_rsi_mean_reversion,
    ema_ribbon_scalp,
)

pytestmark = pytest.mark.scenario


def _sample_ohlc(length: int = 120) -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.date_range("2025-01-01", periods=length, freq="h", tz="UTC")
    close = pd.Series(
        100 + np.sin(np.linspace(0, 10, length)) * 3 + np.linspace(0, 2, length),
        index=index,
    )
    high = close + 0.8
    low = close - 0.8
    return close, high, low


def test_ema_ribbon_scalp_returns_portfolio() -> None:
    """EMA ribbon scalp should return a portfolio object usable for stats."""
    close, high, low = _sample_ohlc()

    pf = ema_ribbon_scalp(
        close=close,
        high=high,
        low=low,
        next_bar_execution=True,
    )

    assert pf is not None
    assert hasattr(pf, "stats")


def test_bb_rsi_mean_reversion_returns_portfolio() -> None:
    """BB/RSI strategy should return a portfolio object usable for stats."""
    close, high, low = _sample_ohlc()

    pf = bb_rsi_mean_reversion(
        close=close,
        high=high,
        low=low,
        next_bar_execution=True,
    )

    assert pf is not None
    assert hasattr(pf, "stats")


def test_next_bar_shift_changes_results() -> None:
    """Next-bar execution should alter returns vs same-bar execution."""
    close, high, low = _sample_ohlc()

    pf_same_bar = ema_ribbon_scalp(
        close=close,
        high=high,
        low=low,
        next_bar_execution=False,
    )
    pf_next_bar = ema_ribbon_scalp(
        close=close,
        high=high,
        low=low,
        next_bar_execution=True,
    )

    assert float(pf_same_bar.total_return()) != float(pf_next_bar.total_return())


def test_ema_ribbon_scalp_rejects_negative_cooldown() -> None:
    """EMA ribbon should reject invalid negative cooldown values."""
    close, high, low = _sample_ohlc()

    with pytest.raises(ValueError, match="entry_cooldown_bars must be >= 0"):
        ema_ribbon_scalp(
            close=close,
            high=high,
            low=low,
            entry_cooldown_bars=-1,
        )
