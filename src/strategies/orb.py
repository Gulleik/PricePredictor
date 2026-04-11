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


def _directional_signals(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    range_bars: int,
    breakout_buffer: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build separate long and short ORB signals from opening range levels."""
    if breakout_buffer < 0:
        raise ValueError("breakout_buffer must be >= 0")

    range_high, range_low, ready = _daily_opening_range(high, low, range_bars)

    upper_break = range_high * (1.0 + breakout_buffer)
    lower_break = range_low * (1.0 - breakout_buffer)

    long_entry = ((price > upper_break) & ready).fillna(False)
    long_exit = ((price < range_low) & ready).fillna(False)
    short_entry = ((price < lower_break) & ready).fillna(False)
    short_exit = ((price > range_high) & ready).fillna(False)

    return long_entry, long_exit, short_entry, short_exit, range_high, range_low


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
    long_entries, long_exits, short_entries, short_exits, range_high, range_low = (
        _directional_signals(
            price,
            high,
            low,
            range_bars=range_bars,
            breakout_buffer=breakout_buffer,
        )
    )

    if not allow_short:
        short_entries = None
        short_exits = None

    entries = long_entries
    exits = long_exits

    if next_bar_execution:
        entries, exits = apply_next_bar_execution(entries, exits)
        if short_entries is not None and short_exits is not None:
            short_entries, short_exits = apply_next_bar_execution(
                short_entries,
                short_exits,
            )

    safe_max_size, valid_mask = sanitize_max_size(max_size, price.index)
    if valid_mask is not None:
        entries = apply_valid_mask(entries, valid_mask)
        exits = apply_valid_mask(exits, valid_mask)
        if short_entries is not None and short_exits is not None:
            short_entries = apply_valid_mask(short_entries, valid_mask)
            short_exits = apply_valid_mask(short_exits, valid_mask)

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

    if short_entries is not None and short_exits is not None:
        portfolio_kwargs["short_entries"] = short_entries
        portfolio_kwargs["short_exits"] = short_exits

    pf = vbt.Portfolio.from_signals(price, entries, exits, **portfolio_kwargs)
    return pf, range_high, range_low


def run_scan(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    range_bars_values: Iterable[int],
    init_cash: float = 10_000.0,
    *,
    breakout_buffer_values: Iterable[float] = (0.001,),
    allow_short: bool = True,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
) -> Any:
    """Run ORB scan across range_bars and breakout_buffer (cartesian product)."""
    entries_df: dict[tuple, pd.Series] = {}
    exits_df: dict[tuple, pd.Series] = {}
    short_entries_df: dict[tuple, pd.Series] = {}
    short_exits_df: dict[tuple, pd.Series] = {}

    for bars in range_bars_values:
        for breakout_buffer in breakout_buffer_values:
            key = (int(bars), float(breakout_buffer))
            entries, exits, short_entries, short_exits, *_ = _directional_signals(
                price,
                high,
                low,
                range_bars=int(bars),
                breakout_buffer=float(breakout_buffer),
            )
            entries_df[key] = entries
            exits_df[key] = exits
            short_entries_df[key] = short_entries
            short_exits_df[key] = short_exits

    entries_frame = pd.DataFrame(entries_df, index=price.index)
    exits_frame = pd.DataFrame(exits_df, index=price.index)
    short_entries_frame = pd.DataFrame(short_entries_df, index=price.index)
    short_exits_frame = pd.DataFrame(short_exits_df, index=price.index)

    if next_bar_execution:
        entries_frame, exits_frame = apply_next_bar_execution(
            entries_frame, exits_frame
        )
        if allow_short:
            short_entries_frame, short_exits_frame = apply_next_bar_execution(
                short_entries_frame,
                short_exits_frame,
            )

    safe_max_size, valid_mask = sanitize_max_size(max_size, price.index)
    if valid_mask is not None:
        entries_frame = apply_valid_mask(entries_frame, valid_mask)
        exits_frame = apply_valid_mask(exits_frame, valid_mask)
        if allow_short:
            short_entries_frame = apply_valid_mask(short_entries_frame, valid_mask)
            short_exits_frame = apply_valid_mask(short_exits_frame, valid_mask)

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

    if allow_short:
        portfolio_kwargs["short_entries"] = short_entries_frame
        portfolio_kwargs["short_exits"] = short_exits_frame

    return vbt.Portfolio.from_signals(
        price, entries_frame, exits_frame, **portfolio_kwargs
    )
