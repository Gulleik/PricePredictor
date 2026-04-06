"""Mean reversion strategy using RSI/Bollinger filters."""

from collections.abc import Iterable
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.analysis.annualization import periods_per_year_from_freq
from src.config import DEFAULT_TIMEFRAME
from src.strategies.common import (
    apply_next_bar_execution,
    apply_valid_mask,
    sanitize_max_size,
)


def _build_signals(
    price: pd.Series,
    *,
    rsi_period: int,
    oversold: float,
    overbought: float,
    bb_window: int,
    bb_std: float,
    use_bollinger: bool,
    vol_lookback: int,
    vol_max_annualized: float,
    portfolio_freq: str | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if rsi_period <= 1:
        raise ValueError("rsi_period must be > 1")
    if oversold >= overbought:
        raise ValueError("oversold must be < overbought")
    if bb_window <= 1:
        raise ValueError("bb_window must be > 1")
    if bb_std <= 0:
        raise ValueError("bb_std must be > 0")

    rsi = vbt.RSI.run(price, window=rsi_period).rsi
    rolling_mean = price.rolling(bb_window).mean()
    rolling_std = price.rolling(bb_window).std()
    upper = rolling_mean + bb_std * rolling_std
    lower = rolling_mean - bb_std * rolling_std

    returns = price.pct_change().fillna(0.0)
    periods_per_year = periods_per_year_from_freq(portfolio_freq or DEFAULT_TIMEFRAME)
    annualized_vol = returns.rolling(vol_lookback).std() * sqrt(periods_per_year)
    vol_filter = annualized_vol <= vol_max_annualized

    entries = rsi < oversold
    exits = rsi > overbought

    if use_bollinger:
        entries = entries & (price < lower)
        exits = exits | (price > rolling_mean)

    entries = entries & vol_filter.fillna(False)

    # Avoid exits before indicators are fully initialized.
    warmup = max(rsi_period, bb_window, vol_lookback)
    entries.iloc[:warmup] = False
    exits.iloc[:warmup] = False

    return entries.astype(bool), exits.astype(bool), rsi, upper, lower


def run(
    price: pd.Series,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    bb_window: int = 20,
    bb_std: float = 2.0,
    use_bollinger: bool = True,
    vol_lookback: int = 24,
    vol_max_annualized: float = 1.5,
    init_cash: float = 10_000.0,
    *,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: Any | None = None,
    position_sizes: Any | None = None,
    portfolio_freq: str | None = None,
) -> tuple[Any, pd.Series, pd.Series, pd.Series]:
    """Run RSI/Bollinger mean-reversion backtest."""
    entries, exits, rsi, bb_upper, bb_lower = _build_signals(
        price,
        rsi_period=rsi_period,
        oversold=oversold,
        overbought=overbought,
        bb_window=bb_window,
        bb_std=bb_std,
        use_bollinger=use_bollinger,
        vol_lookback=vol_lookback,
        vol_max_annualized=vol_max_annualized,
        portfolio_freq=portfolio_freq,
    )

    if next_bar_execution:
        entries, exits = apply_next_bar_execution(entries, exits)

    safe_max_size, valid_mask = sanitize_max_size(max_size, price.index)
    if valid_mask is not None:
        entries = apply_valid_mask(entries, valid_mask)
        exits = apply_valid_mask(exits, valid_mask)

    portfolio_kwargs: dict[str, Any] = {
        "init_cash": init_cash,
        "fees": fees,
        "fixed_fees": fixed_fees,
        "slippage": slippage,
        "freq": portfolio_freq or DEFAULT_TIMEFRAME,
    }
    if safe_max_size is not None:
        portfolio_kwargs["max_size"] = safe_max_size
    if position_sizes is not None:
        portfolio_kwargs["size"] = position_sizes

    pf = vbt.Portfolio.from_signals(price, entries, exits, **portfolio_kwargs)
    return pf, rsi, bb_upper, bb_lower


def run_scan(
    price: pd.Series,
    rsi_periods: Iterable[int],
    bb_windows: Iterable[int],
    init_cash: float = 10_000.0,
    *,
    oversold: float = 30.0,
    overbought: float = 70.0,
    bb_std: float = 2.0,
    use_bollinger: bool = True,
    vol_lookback: int = 24,
    vol_max_annualized: float = 1.5,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
) -> Any:
    """Run mean-reversion scan across RSI and Bollinger windows."""
    entries_df: dict[str, pd.Series] = {}
    exits_df: dict[str, pd.Series] = {}

    for rsi_period in rsi_periods:
        for bb_window in bb_windows:
            label = f"rsi={int(rsi_period)}|bb={int(bb_window)}"
            entries, exits, _, _, _ = _build_signals(
                price,
                rsi_period=int(rsi_period),
                oversold=oversold,
                overbought=overbought,
                bb_window=int(bb_window),
                bb_std=bb_std,
                use_bollinger=use_bollinger,
                vol_lookback=vol_lookback,
                vol_max_annualized=vol_max_annualized,
                portfolio_freq=portfolio_freq,
            )
            entries_df[label] = entries
            exits_df[label] = exits

    entries_frame = pd.DataFrame(entries_df, index=price.index)
    exits_frame = pd.DataFrame(exits_df, index=price.index)

    if next_bar_execution:
        entries_frame, exits_frame = apply_next_bar_execution(entries_frame, exits_frame)

    safe_max_size, valid_mask = sanitize_max_size(max_size, price.index)
    if valid_mask is not None:
        entries_frame = apply_valid_mask(entries_frame, valid_mask)
        exits_frame = apply_valid_mask(exits_frame, valid_mask)

    portfolio_kwargs: dict[str, Any] = {
        "init_cash": init_cash,
        "fees": fees,
        "fixed_fees": fixed_fees,
        "slippage": slippage,
        "freq": portfolio_freq or DEFAULT_TIMEFRAME,
    }
    if safe_max_size is not None:
        max_size_arr = np.asarray(safe_max_size)
        if max_size_arr.ndim == 1:
            portfolio_kwargs["max_size"] = max_size_arr.reshape(-1, 1)
        else:
            portfolio_kwargs["max_size"] = max_size_arr

    return vbt.Portfolio.from_signals(price, entries_frame, exits_frame, **portfolio_kwargs)
