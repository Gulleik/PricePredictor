"""Opening Range Breakout strategy using UTC daily sessions."""

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


def _daily_opening_range(
    high: pd.Series,
    low: pd.Series,
    range_bars: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if range_bars <= 0:
        raise ValueError("range_bars must be > 0")

    day_key = high.index.floor("D")

    range_high = pd.Series(np.nan, index=high.index, dtype=float)
    range_low = pd.Series(np.nan, index=low.index, dtype=float)
    ready = pd.Series(False, index=high.index)

    for _, idx in pd.Series(day_key, index=high.index).groupby(day_key).groups.items():
        day_high = high.loc[idx]
        day_low = low.loc[idx]

        opening_high = day_high.iloc[:range_bars].max()
        opening_low = day_low.iloc[:range_bars].min()

        range_high.loc[idx] = opening_high
        range_low.loc[idx] = opening_low
        if len(idx) > range_bars:
            ready.loc[idx[range_bars:]] = True

    return range_high, range_low, ready


def _signals(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    range_bars: int,
    breakout_buffer: float,
    allow_short: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if breakout_buffer < 0:
        raise ValueError("breakout_buffer must be >= 0")

    range_high, range_low, ready = _daily_opening_range(high, low, range_bars)

    upper_break = range_high * (1.0 + breakout_buffer)
    lower_break = range_low * (1.0 - breakout_buffer)

    long_entry = (price > upper_break) & ready
    long_exit = (price < range_low) & ready

    if allow_short:
        short_entry = (price < lower_break) & ready
        short_exit = (price > range_high) & ready
        entries = long_entry | short_entry
        exits = long_exit | short_exit
    else:
        entries = long_entry
        exits = long_exit

    return entries.fillna(False), exits.fillna(False), range_high, range_low


def run(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    range_bars: int = 3,
    breakout_buffer: float = 0.001,
    allow_short: bool = True,
    init_cash: float = 10_000.0,
    *,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: Any | None = None,
    position_sizes: Any | None = None,
    portfolio_freq: str | None = None,
) -> tuple[Any, pd.Series, pd.Series]:
    """Run ORB strategy backtest."""
    entries, exits, range_high, range_low = _signals(
        price,
        high,
        low,
        range_bars=range_bars,
        breakout_buffer=breakout_buffer,
        allow_short=allow_short,
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
    return pf, range_high, range_low


def run_scan(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    range_bars_values: Iterable[int],
    init_cash: float = 10_000.0,
    *,
    breakout_buffer: float = 0.001,
    allow_short: bool = True,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
) -> Any:
    """Run ORB scan across opening range lengths."""
    entries_df: dict[int, pd.Series] = {}
    exits_df: dict[int, pd.Series] = {}

    for bars in range_bars_values:
        entries, exits, *_ = _signals(
            price,
            high,
            low,
            range_bars=int(bars),
            breakout_buffer=breakout_buffer,
            allow_short=allow_short,
        )
        entries_df[int(bars)] = entries
        exits_df[int(bars)] = exits

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
