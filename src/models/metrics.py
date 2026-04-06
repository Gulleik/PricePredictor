"""Performance metrics utilities for risk-adjusted strategy evaluation."""

import math

import numpy as np
import pandas as pd


def _coerce_returns(returns: pd.Series) -> pd.Series:
    """Validate and clean a returns series for metric computations."""
    if returns.empty:
        raise ValueError("returns cannot be empty")

    cleaned = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna()
    if cleaned.empty:
        raise ValueError("returns contain no finite values")
    return cleaned


def compute_sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio from periodic returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    clean_returns = _coerce_returns(returns)
    excess = clean_returns - (risk_free_rate / periods_per_year)

    # Sample stdev with ddof=1 is undefined for fewer than 2 observations.
    if len(excess) < 2:
        return 0.0

    stdev = float(excess.std(ddof=1))
    if not np.isfinite(stdev) or stdev == 0.0:
        return 0.0

    mean_excess = float(excess.mean())
    return mean_excess / stdev * math.sqrt(periods_per_year)


def compute_sortino_ratio(
    returns: pd.Series,
    *,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio using downside deviation only."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    clean_returns = _coerce_returns(returns)
    target_per_period = target_return / periods_per_year
    downside = clean_returns[clean_returns < target_per_period] - target_per_period

    if downside.empty:
        return 0.0

    downside_dev = float(np.sqrt(np.mean(np.square(downside.to_numpy(dtype=float)))))
    if downside_dev == 0.0:
        return 0.0

    mean_excess = float((clean_returns - target_per_period).mean())
    return mean_excess / downside_dev * math.sqrt(periods_per_year)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute max drawdown from periodic returns as a negative decimal."""
    clean_returns = _coerce_returns(returns)
    equity_curve = (1.0 + clean_returns).cumprod()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0
    return float(drawdown.min())


def compute_calmar_ratio(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> float:
    """Compute Calmar ratio using CAGR divided by absolute max drawdown."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    clean_returns = _coerce_returns(returns)
    max_drawdown = abs(compute_max_drawdown(clean_returns))
    if max_drawdown == 0.0:
        return 0.0

    n_periods = len(clean_returns)
    total_return = float((1.0 + clean_returns).prod())
    years = n_periods / periods_per_year
    if years <= 0.0 or total_return <= 0.0:
        return 0.0

    cagr = total_return ** (1.0 / years) - 1.0
    return float(cagr / max_drawdown)


def compute_max_drawdown_duration(returns: pd.Series) -> int:
    """Compute longest drawdown stretch in bars."""
    clean_returns = _coerce_returns(returns)
    equity_curve = (1.0 + clean_returns).cumprod()
    rolling_peak = equity_curve.cummax()

    max_duration = 0
    current_duration = 0

    for is_drawdown in equity_curve < rolling_peak:
        if bool(is_drawdown):
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return int(max_duration)


def compute_advanced_metrics(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Compute milestone-5 advanced metrics for one return series."""
    sharpe = compute_sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino = compute_sortino_ratio(returns, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(returns, periods_per_year=periods_per_year)
    max_drawdown = compute_max_drawdown(returns)
    max_dd_duration = compute_max_drawdown_duration(returns)

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_duration": int(max_dd_duration),
    }
