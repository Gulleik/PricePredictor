"""Tests for look-ahead bias guards."""

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from src.strategies.sma_crossover import run as sma_run
from src.validators import validate_no_future_leakage

pytestmark = pytest.mark.scenario


def _entry_signal(price: pd.Series) -> pd.Series:
    fast_ma = vbt.MA.run(price, 5)
    slow_ma = vbt.MA.run(price, 15)
    return fast_ma.ma_crossed_above(slow_ma)


def _exit_signal(price: pd.Series) -> pd.Series:
    fast_ma = vbt.MA.run(price, 5)
    slow_ma = vbt.MA.run(price, 15)
    return fast_ma.ma_crossed_below(slow_ma)


def test_validate_no_future_leakage_passes_for_sma_signals() -> None:
    """SMA crossover entry/exit signals should be prefix-stable under future shocks."""
    index = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    base = 100 + np.cumsum(np.sin(np.linspace(0, 10, 120)))
    price = pd.Series(base, index=index)

    validate_no_future_leakage(price, _entry_signal, perturb_from=80)
    validate_no_future_leakage(price, _exit_signal, perturb_from=80)


def test_validate_no_future_leakage_raises_for_cheating_signal() -> None:
    """A deliberately leaky signal should be rejected."""
    index = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    price = pd.Series(np.arange(50, dtype=float), index=index)

    def leaky_signal(series: pd.Series) -> pd.Series:
        return series.shift(-1) > series

    with pytest.raises(ValueError, match="future data"):
        validate_no_future_leakage(price, leaky_signal, perturb_from=20)


def test_sma_run_ma_prefix_stable_under_future_shock() -> None:
    """Running SMA with altered future prices should not alter historical MA values."""
    index = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    price = pd.Series(np.linspace(100, 200, 100), index=index)

    _, fast_base, slow_base = sma_run(price, fast=5, slow=15)

    shocked = price.copy()
    shocked.iloc[70:] = shocked.iloc[70:] * 10
    _, fast_shocked, slow_shocked = sma_run(shocked, fast=5, slow=15)

    pd.testing.assert_series_equal(
        fast_base.ma.iloc[:70],
        fast_shocked.ma.iloc[:70],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        slow_base.ma.iloc[:70],
        slow_shocked.ma.iloc[:70],
        check_names=False,
    )
