"""Unit tests for advanced risk-adjusted metrics."""

import numpy as np
import pandas as pd
import pytest

from src.models.metrics import (
    compute_advanced_metrics,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)

pytestmark = pytest.mark.unit


def test_sharpe_ratio_returns_zero_for_flat_series() -> None:
    """Flat returns should produce zero Sharpe ratio."""
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert compute_sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_returns_zero_for_single_finite_observation() -> None:
    """One finite observation should not produce NaN Sharpe ratio."""
    returns = pd.Series([np.nan, np.inf, 0.01])
    assert compute_sharpe_ratio(returns) == 0.0


def test_sortino_ratio_handles_downside_only() -> None:
    """Sortino should be finite when downside returns are present."""
    returns = pd.Series([0.01, -0.02, 0.005, -0.01, 0.02])
    ratio = compute_sortino_ratio(returns)
    assert np.isfinite(ratio)


def test_calmar_ratio_uses_drawdown_denominator() -> None:
    """Calmar should be non-positive when CAGR is non-positive."""
    returns = pd.Series([0.01, -0.03, 0.01, -0.02])
    calmar = compute_calmar_ratio(returns)
    assert calmar <= 0.0


def test_max_drawdown_is_negative_or_zero() -> None:
    """Max drawdown is represented as a negative decimal."""
    returns = pd.Series([0.05, -0.10, 0.02, -0.03])
    max_dd = compute_max_drawdown(returns)
    assert max_dd <= 0.0


def test_max_drawdown_duration_counts_bars() -> None:
    """Drawdown duration should count consecutive bars below peak."""
    returns = pd.Series([0.10, -0.02, -0.01, -0.01, 0.05])
    assert compute_max_drawdown_duration(returns) == 3


def test_compute_advanced_metrics_has_expected_keys() -> None:
    """Advanced metrics helper should provide all milestone 5 keys."""
    returns = pd.Series([0.01, -0.005, 0.008, -0.002, 0.01])
    metrics = compute_advanced_metrics(returns)

    assert set(metrics.keys()) == {
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration",
    }


def test_metrics_raise_on_empty_returns() -> None:
    """Metric computation should reject empty series."""
    returns = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="returns cannot be empty"):
        compute_sharpe_ratio(returns)
