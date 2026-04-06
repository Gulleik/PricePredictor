"""Volatility breakout strategy using Donchian channels with ATR filter."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.config import DEFAULT_TIMEFRAME
from src.strategies.common import (
    apply_next_bar_execution,
    apply_valid_mask,
    sanitize_max_size,
)
from src.strategies.trend_following import _atr


def _signals(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    donchian_window: int,
    atr_window: int,
    use_atr_filter: bool,
    atr_min_fraction: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if donchian_window <= 1:
        raise ValueError("donchian_window must be > 1")
    if atr_window <= 1:
        raise ValueError("atr_window must be > 1")

    upper = high.rolling(donchian_window).max().shift(1)
    lower = low.rolling(donchian_window).min().shift(1)

    atr = _atr(high, low, price, atr_window)
    atr_ok = (atr / price) >= atr_min_fraction

    entries = price > upper
    exits = price < lower
    if use_atr_filter:
        entries = entries & atr_ok

    warmup = max(donchian_window, atr_window)
    entries.iloc[:warmup] = False
    exits.iloc[:warmup] = False

    return entries.fillna(False), exits.fillna(False), upper, lower, atr


def run(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    donchian_window: int = 20,
    atr_window: int = 14,
    use_atr_filter: bool = True,
    atr_min_fraction: float = 0.005,
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
    """Run Donchian breakout backtest."""
    entries, exits, upper, lower, atr = _signals(
        price,
        high,
        low,
        donchian_window=donchian_window,
        atr_window=atr_window,
        use_atr_filter=use_atr_filter,
        atr_min_fraction=atr_min_fraction,
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
    return pf, upper, lower, atr


def run_scan(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    donchian_windows: Iterable[int],
    init_cash: float = 10_000.0,
    *,
    atr_window: int = 14,
    use_atr_filter: bool = True,
    atr_min_fraction: float = 0.005,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
) -> Any:
    """Run Donchian scan across channel windows."""
    entries_df: dict[int, pd.Series] = {}
    exits_df: dict[int, pd.Series] = {}

    for window in donchian_windows:
        entries, exits, *_ = _signals(
            price,
            high,
            low,
            donchian_window=int(window),
            atr_window=atr_window,
            use_atr_filter=use_atr_filter,
            atr_min_fraction=atr_min_fraction,
        )
        entries_df[int(window)] = entries
        exits_df[int(window)] = exits

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
