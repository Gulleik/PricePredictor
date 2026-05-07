"""Trend-following strategy using EMA cross with ATR trailing stop."""

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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _signals_from_series(
    price: pd.Series,
    *,
    high: pd.Series | None,
    low: pd.Series | None,
    fast_window: int,
    slow_window: int,
    atr_window: int,
    atr_stop_multiple: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")
    if atr_window <= 1:
        raise ValueError("atr_window must be > 1")
    if atr_stop_multiple <= 0:
        raise ValueError("atr_stop_multiple must be > 0")

    high_series = high if high is not None else price
    low_series = low if low is not None else price

    fast_ema = vbt.MA.run(price, fast_window, ewm=True).ma
    slow_ema = vbt.MA.run(price, slow_window, ewm=True).ma
    entries = fast_ema.vbt.crossed_above(slow_ema)
    cross_exit = fast_ema.vbt.crossed_below(slow_ema)

    atr = _atr(high_series, low_series, price, atr_window)
    trail_level = price - atr_stop_multiple * atr
    rolling_trail = trail_level.ffill().cummax()
    atr_exit = price < rolling_trail
    exits = (cross_exit | atr_exit).fillna(False)

    return entries.fillna(False), exits, fast_ema, slow_ema, atr, rolling_trail


def run(
    price: pd.Series,
    fast_window: int = 12,
    slow_window: int = 26,
    atr_window: int = 14,
    atr_stop_multiple: float = 2.0,
    init_cash: float = 10_000.0,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: Any | None = None,
    position_sizes: Any | None = None,
    portfolio_freq: str | None = None,
    leverage: float = 1.0,
) -> tuple[Any, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Run EMA trend-following backtest."""
    entries, exits, fast_ema, slow_ema, atr, trailing_stop = _signals_from_series(
        price,
        high=high,
        low=low,
        fast_window=fast_window,
        slow_window=slow_window,
        atr_window=atr_window,
        atr_stop_multiple=atr_stop_multiple,
    )

    if next_bar_execution:
        entries, exits = apply_next_bar_execution(entries, exits)

    safe_max_size, valid_mask = sanitize_max_size(max_size, price.index)
    if valid_mask is not None:
        entries = apply_valid_mask(entries, valid_mask)
        exits = apply_valid_mask(exits, valid_mask)

    portfolio_kwargs: dict[str, Any] = {
        "init_cash": init_cash * leverage,
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
    return pf, fast_ema, slow_ema, atr, trailing_stop


def run_scan(
    price: pd.Series,
    fast_windows: Iterable[int],
    slow_windows: Iterable[int],
    init_cash: float = 10_000.0,
    *,
    atr_windows: Iterable[int] = (14,),
    atr_stop_multiples: Iterable[float] = (2.0,),
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
    leverage: float = 1.0,
) -> Any:
    """Run EMA scan across all 4 tunable parameters (cartesian product)."""
    entries_df: dict[tuple, pd.Series] = {}
    exits_df: dict[tuple, pd.Series] = {}

    for fast_window in fast_windows:
        for slow_window in slow_windows:
            if int(fast_window) >= int(slow_window):
                continue
            for atr_w in atr_windows:
                for atr_m in atr_stop_multiples:
                    key = (
                        int(fast_window),
                        int(slow_window),
                        int(atr_w),
                        float(atr_m),
                    )
                    entries, exits, *_ = _signals_from_series(
                        price,
                        high=high,
                        low=low,
                        fast_window=int(fast_window),
                        slow_window=int(slow_window),
                        atr_window=int(atr_w),
                        atr_stop_multiple=float(atr_m),
                    )
                    entries_df[key] = entries
                    exits_df[key] = exits

    if not entries_df:
        raise ValueError("No valid fast/slow combinations for trend_following scan.")

    entries_frame = pd.DataFrame(entries_df, index=price.index)
    exits_frame = pd.DataFrame(exits_df, index=price.index)

    if next_bar_execution:
        entries_frame, exits_frame = apply_next_bar_execution(
            entries_frame, exits_frame
        )

    safe_max_size, valid_mask = sanitize_max_size(max_size, price.index)
    if valid_mask is not None:
        entries_frame = apply_valid_mask(entries_frame, valid_mask)
        exits_frame = apply_valid_mask(exits_frame, valid_mask)

    portfolio_kwargs: dict[str, Any] = {
        "init_cash": init_cash * leverage,
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

    return vbt.Portfolio.from_signals(
        price, entries_frame, exits_frame, **portfolio_kwargs
    )
