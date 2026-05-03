"""Vectorized scalping strategies compatible with VectorBT from_signals."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.config import DEFAULT_TIMEFRAME
from src.strategies.common import apply_valid_mask, sanitize_max_size


def _coerce_size_series(
    size_like: pd.Series | np.ndarray | float,
    *,
    index: pd.Index,
) -> pd.Series:
    """Normalize externally-provided position sizes to a Series on price index."""
    if isinstance(size_like, pd.Series):
        return size_like.reindex(index).ffill().fillna(0.0).astype(float)

    arr = np.asarray(size_like, dtype=float)
    if arr.ndim == 0:
        return pd.Series(float(arr), index=index, dtype=float)
    if arr.shape[0] != len(index):
        raise ValueError("position_sizes length must match price length")
    return pd.Series(arr, index=index, dtype=float)


def _as_equity_series(
    close: pd.Series,
    *,
    init_cash: float,
    equity: pd.Series | float | None,
) -> pd.Series:
    if equity is None:
        return pd.Series(float(init_cash), index=close.index, dtype=float)
    if isinstance(equity, pd.Series):
        aligned = equity.reindex(close.index).astype(float)
        return aligned.ffill().fillna(float(init_cash))
    return pd.Series(float(equity), index=close.index, dtype=float)


def _risk_arrays(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    init_cash: float,
    equity: pd.Series | float | None,
    atr_window: int,
    atr_multiple: float,
    risk_fraction: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if atr_window <= 1:
        raise ValueError("atr_window must be > 1")
    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be > 0")
    if risk_fraction <= 0:
        raise ValueError("risk_fraction must be > 0")

    atr = vbt.ATR.run(high, low, close, window=atr_window).atr.astype(float)
    stop_distance = (atr * float(atr_multiple)).astype(float)

    equity_series = _as_equity_series(close, init_cash=init_cash, equity=equity)
    safe_denominator = stop_distance.replace(0.0, np.nan)
    size = (equity_series * float(risk_fraction)) / safe_denominator
    size = size.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    safe_stop = stop_distance.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return atr, safe_stop, size


def _shift_signal(signal: pd.Series) -> pd.Series:
    return signal.vbt.fshift(1).fillna(False).astype(bool)


def _apply_entry_cooldown(signal: pd.Series, cooldown_bars: int) -> pd.Series:
    """Suppress repeated entries for a fixed number of bars after a trigger."""
    if cooldown_bars <= 0:
        return signal.fillna(False).astype(bool)

    raw = signal.fillna(False).to_numpy(dtype=bool)
    filtered = np.zeros_like(raw, dtype=bool)
    cooldown_remaining = 0
    for idx, flag in enumerate(raw):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        if flag and cooldown_remaining == 0:
            filtered[idx] = True
            cooldown_remaining = cooldown_bars

    return pd.Series(filtered, index=signal.index, dtype=bool)


def ema_ribbon_scalp(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    init_cash: float = 10_000.0,
    ema_fast: int = 8,
    ema_medium: int = 13,
    ema_slow: int = 21,
    pullback_reclaim: bool = True,
    entry_cooldown_bars: int = 2,
    atr_window: int = 14,
    atr_multiple: float = 1.5,
    risk_fraction: float = 0.005,
    tp_r_multiple: float = 2.0,
    fees: float = 0.0005,
    slippage: float = 0.0001,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    equity: pd.Series | float | None = None,
    position_sizes: pd.Series | np.ndarray | float | None = None,
    max_size: pd.Series | np.ndarray | float | None = None,
    portfolio_freq: str | None = None,
    leverage: float = 1.0,
) -> Any:
    """Run EMA ribbon scalp strategy and return a VectorBT portfolio."""
    if not (ema_fast < ema_medium < ema_slow):
        raise ValueError("EMA windows must satisfy ema_fast < ema_medium < ema_slow")
    if entry_cooldown_bars < 0:
        raise ValueError("entry_cooldown_bars must be >= 0")
    if tp_r_multiple <= 0:
        raise ValueError("tp_r_multiple must be > 0")

    ma = vbt.MA.run(close, window=[ema_fast, ema_medium, ema_slow], ewm=True).ma
    fast = ma.iloc[:, 0]
    medium = ma.iloc[:, 1]
    slow = ma.iloc[:, 2]

    trend_filter = (fast > medium) & (medium > slow)
    pullback_touch = low <= medium
    reclaim_filter = (
        close >= medium if pullback_reclaim else pd.Series(True, index=close.index)
    )
    entries = (trend_filter & pullback_touch & reclaim_filter).fillna(False)
    entries = _apply_entry_cooldown(entries, int(entry_cooldown_bars))
    exits = close.vbt.crossed_below(medium).fillna(False)

    if next_bar_execution:
        entries = _shift_signal(entries)
        exits = _shift_signal(exits)

    safe_max_size, valid_mask = sanitize_max_size(max_size, close.index)
    if valid_mask is not None:
        entries = apply_valid_mask(entries, valid_mask)
        exits = apply_valid_mask(exits, valid_mask)

    _, stop_distance, risk_size = _risk_arrays(
        high,
        low,
        close,
        init_cash=init_cash,
        equity=equity,
        atr_window=atr_window,
        atr_multiple=atr_multiple,
        risk_fraction=risk_fraction,
    )
    size = (
        _coerce_size_series(position_sizes, index=close.index)
        if position_sizes is not None
        else risk_size * leverage
    )

    return vbt.Portfolio.from_signals(
        close,
        entries.astype(bool),
        exits.astype(bool),
        init_cash=init_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        size=size,
        max_size=safe_max_size,
        sl_stop=stop_distance,
        tp_stop=stop_distance * float(tp_r_multiple),
        freq=portfolio_freq or DEFAULT_TIMEFRAME,
    )


def bb_rsi_mean_reversion(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    init_cash: float = 10_000.0,
    bb_window: int = 20,
    bb_std: float = 2.0,
    rsi_window: int = 7,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    atr_window: int = 14,
    atr_multiple: float = 1.5,
    risk_fraction: float = 0.005,
    tp_r_multiple: float = 2.0,
    fees: float = 0.0005,
    slippage: float = 0.0001,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    equity: pd.Series | float | None = None,
    position_sizes: pd.Series | np.ndarray | float | None = None,
    max_size: pd.Series | np.ndarray | float | None = None,
    portfolio_freq: str | None = None,
    leverage: float = 1.0,
) -> Any:
    """Run BB/RSI mean-reversion strategy and return a VectorBT portfolio."""
    if bb_window <= 1:
        raise ValueError("bb_window must be > 1")
    if bb_std <= 0:
        raise ValueError("bb_std must be > 0")
    if rsi_window <= 1:
        raise ValueError("rsi_window must be > 1")
    if rsi_oversold >= rsi_overbought:
        raise ValueError("rsi_oversold must be < rsi_overbought")
    if tp_r_multiple <= 0:
        raise ValueError("tp_r_multiple must be > 0")

    bb = vbt.BBANDS.run(close, window=bb_window, alpha=bb_std)
    rsi = vbt.RSI.run(close, window=rsi_window).rsi

    raw_long_entries = close.vbt.crossed_above(bb.lower) & (rsi < rsi_oversold)
    raw_short_entries = close.vbt.crossed_below(bb.upper) & (rsi > rsi_overbought)

    long_entries = (raw_long_entries & ~raw_short_entries).fillna(False)
    short_entries = (raw_short_entries & ~raw_long_entries).fillna(False)

    long_exits = ((close >= bb.middle) | short_entries).fillna(False)
    short_exits = ((close <= bb.middle) | long_entries).fillna(False)

    if next_bar_execution:
        long_entries = _shift_signal(long_entries)
        short_entries = _shift_signal(short_entries)
        long_exits = _shift_signal(long_exits)
        short_exits = _shift_signal(short_exits)

    safe_max_size, valid_mask = sanitize_max_size(max_size, close.index)
    if valid_mask is not None:
        long_entries = apply_valid_mask(long_entries, valid_mask)
        short_entries = apply_valid_mask(short_entries, valid_mask)
        long_exits = apply_valid_mask(long_exits, valid_mask)
        short_exits = apply_valid_mask(short_exits, valid_mask)

    _, stop_distance, risk_size = _risk_arrays(
        high,
        low,
        close,
        init_cash=init_cash,
        equity=equity,
        atr_window=atr_window,
        atr_multiple=atr_multiple,
        risk_fraction=risk_fraction,
    )
    size = (
        _coerce_size_series(position_sizes, index=close.index)
        if position_sizes is not None
        else risk_size * leverage
    )

    return vbt.Portfolio.from_signals(
        close,
        entries=long_entries.astype(bool),
        exits=long_exits.astype(bool),
        short_entries=short_entries.astype(bool),
        short_exits=short_exits.astype(bool),
        init_cash=init_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        size=size,
        max_size=safe_max_size,
        sl_stop=stop_distance,
        tp_stop=stop_distance * float(tp_r_multiple),
        freq=portfolio_freq or DEFAULT_TIMEFRAME,
    )


def run(
    price: pd.Series,
    *,
    high: pd.Series,
    low: pd.Series,
    **kwargs: Any,
) -> Any:
    """Run one scalping strategy via a common strategy-module interface."""
    if "ema_fast" in kwargs or "ema_medium" in kwargs or "ema_slow" in kwargs:
        return ema_ribbon_scalp(price, high, low, **kwargs)
    return bb_rsi_mean_reversion(price, high, low, **kwargs)


def run_scan(
    price: pd.Series,
    *,
    high: pd.Series,
    low: pd.Series,
    strategy_variant: str = "ema_ribbon_scalp",
    ema_fast_windows: Iterable[int] = (8,),
    ema_medium_windows: Iterable[int] = (13,),
    ema_slow_windows: Iterable[int] = (21,),
    bb_windows: Iterable[int] = (20,),
    bb_std_values: Iterable[float] = (2.0,),
    rsi_windows: Iterable[int] = (7,),
    oversold_values: Iterable[float] = (30.0,),
    overbought_values: Iterable[float] = (70.0,),
    **kwargs: Any,
) -> Any:
    """Run scan for whichever scalping family is implied by provided grids."""
    if strategy_variant == "ema_ribbon_scalp":
        entries_df: dict[tuple[int, int, int], pd.Series] = {}
        exits_df: dict[tuple[int, int, int], pd.Series] = {}
        pullback_reclaim = bool(kwargs.get("pullback_reclaim", True))
        entry_cooldown_bars = int(kwargs.get("entry_cooldown_bars", 0))
        if entry_cooldown_bars < 0:
            raise ValueError("entry_cooldown_bars must be >= 0")
        safe_max_size, valid_mask = sanitize_max_size(
            kwargs.get("max_size"),
            price.index,
        )
        for ema_fast in ema_fast_windows:
            for ema_medium in ema_medium_windows:
                for ema_slow in ema_slow_windows:
                    if not (int(ema_fast) < int(ema_medium) < int(ema_slow)):
                        continue
                    ma = vbt.MA.run(
                        price,
                        window=[int(ema_fast), int(ema_medium), int(ema_slow)],
                        ewm=True,
                    ).ma
                    fast = ma.iloc[:, 0]
                    medium = ma.iloc[:, 1]
                    slow = ma.iloc[:, 2]
                    reclaim_filter = (
                        price >= medium
                        if pullback_reclaim
                        else pd.Series(True, index=price.index)
                    )
                    entries = (
                        (fast > medium)
                        & (medium > slow)
                        & (low <= medium)
                        & reclaim_filter
                    ).fillna(False)
                    entries = _apply_entry_cooldown(entries, entry_cooldown_bars)
                    exits = price.vbt.crossed_below(medium).fillna(False)
                    key = (int(ema_fast), int(ema_medium), int(ema_slow))
                    entries_df[key] = entries.astype(bool)
                    exits_df[key] = exits.astype(bool)

        if not entries_df:
            raise ValueError("No valid EMA ribbon parameter combinations for scan.")

        entries_frame = pd.DataFrame(entries_df, index=price.index)
        exits_frame = pd.DataFrame(exits_df, index=price.index)
        if valid_mask is not None:
            entries_frame = apply_valid_mask(entries_frame, valid_mask)
            exits_frame = apply_valid_mask(exits_frame, valid_mask)
        if kwargs.get("next_bar_execution", False):
            entries_frame = entries_frame.vbt.fshift(1).fillna(False).astype(bool)
            exits_frame = exits_frame.vbt.fshift(1).fillna(False).astype(bool)

        return vbt.Portfolio.from_signals(
            price,
            entries_frame,
            exits_frame,
            init_cash=kwargs.get("init_cash", 10_000.0),
            fees=kwargs.get("fees", 0.0),
            fixed_fees=kwargs.get("fixed_fees", 0.0),
            slippage=kwargs.get("slippage", 0.0),
            # Skip volume constraints during scan; apply them only to final runs.
            max_size=None,
            freq=kwargs.get("portfolio_freq") or DEFAULT_TIMEFRAME,
        )

    if strategy_variant != "bb_rsi_mean_reversion":
        raise ValueError(
            "Unknown strategy_variant for run_scan: "
            f"{strategy_variant}. Use 'ema_ribbon_scalp' or "
            "'bb_rsi_mean_reversion'."
        )

    entries_df2: dict[tuple[int, float, int, float, float], pd.Series] = {}
    exits_df2: dict[tuple[int, float, int, float, float], pd.Series] = {}
    short_entries_df2: dict[tuple[int, float, int, float, float], pd.Series] = {}
    short_exits_df2: dict[tuple[int, float, int, float, float], pd.Series] = {}
    safe_max_size, valid_mask = sanitize_max_size(
        kwargs.get("max_size"),
        price.index,
    )
    for bb_window in bb_windows:
        for bb_std in bb_std_values:
            for rsi_window in rsi_windows:
                for oversold in oversold_values:
                    for overbought in overbought_values:
                        if float(oversold) >= float(overbought):
                            continue
                        bb = vbt.BBANDS.run(
                            price,
                            window=int(bb_window),
                            alpha=float(bb_std),
                        )
                        rsi = vbt.RSI.run(price, window=int(rsi_window)).rsi
                        long_entries = (
                            price.vbt.crossed_above(bb.lower) & (rsi < float(oversold))
                        ).fillna(False)
                        short_entries = (
                            price.vbt.crossed_below(bb.upper)
                            & (rsi > float(overbought))
                        ).fillna(False)
                        long_exits = ((price >= bb.middle) | short_entries).fillna(
                            False
                        )
                        short_exits = ((price <= bb.middle) | long_entries).fillna(
                            False
                        )
                        key = (
                            int(bb_window),
                            float(bb_std),
                            int(rsi_window),
                            float(oversold),
                            float(overbought),
                        )
                        entries_df2[key] = long_entries.astype(bool)
                        exits_df2[key] = long_exits.astype(bool)
                        short_entries_df2[key] = short_entries.astype(bool)
                        short_exits_df2[key] = short_exits.astype(bool)

    if not entries_df2:
        raise ValueError("No valid BB/RSI parameter combinations for scan.")

    entries_frame2 = pd.DataFrame(entries_df2, index=price.index)
    exits_frame2 = pd.DataFrame(exits_df2, index=price.index)
    short_entries_frame2 = pd.DataFrame(short_entries_df2, index=price.index)
    short_exits_frame2 = pd.DataFrame(short_exits_df2, index=price.index)
    if valid_mask is not None:
        entries_frame2 = apply_valid_mask(entries_frame2, valid_mask)
        exits_frame2 = apply_valid_mask(exits_frame2, valid_mask)
        short_entries_frame2 = apply_valid_mask(short_entries_frame2, valid_mask)
        short_exits_frame2 = apply_valid_mask(short_exits_frame2, valid_mask)
    if kwargs.get("next_bar_execution", False):
        entries_frame2 = entries_frame2.vbt.fshift(1).fillna(False).astype(bool)
        exits_frame2 = exits_frame2.vbt.fshift(1).fillna(False).astype(bool)
        short_entries_frame2 = (
            short_entries_frame2.vbt.fshift(1).fillna(False).astype(bool)
        )
        short_exits_frame2 = short_exits_frame2.vbt.fshift(1).fillna(False).astype(bool)

    return vbt.Portfolio.from_signals(
        price,
        entries=entries_frame2,
        exits=exits_frame2,
        short_entries=short_entries_frame2,
        short_exits=short_exits_frame2,
        init_cash=kwargs.get("init_cash", 10_000.0),
        fees=kwargs.get("fees", 0.0),
        fixed_fees=kwargs.get("fixed_fees", 0.0),
        slippage=kwargs.get("slippage", 0.0),
        # Skip volume constraints during scan; apply them only to final runs.
        max_size=None,
        freq=kwargs.get("portfolio_freq") or DEFAULT_TIMEFRAME,
    )
