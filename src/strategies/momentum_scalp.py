"""Momentum scalping strategy with multi-TP levels via split sub-portfolios."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.config import DEFAULT_TIMEFRAME
from src.strategies.common import (
    apply_valid_mask,
    sanitize_max_size,
)


class CombinedPortfolio:
    """
    Aggregate wrapper for 3 sub-portfolios with different TP levels.

    Provides interface methods (.returns(), .total_return(), .value(), .trades)
    compatible with Optuna integration and Kelly sizing estimation.
    """

    def __init__(
        self,
        tp1_pf: Any,
        tp2_pf: Any,
        tp3_pf: Any,
        tp1_alloc: float = 0.40,
        tp2_alloc: float = 0.30,
        tp3_alloc: float = 0.30,
    ):
        """
        Initialize combined portfolio from 3 sub-portfolios.

        Args:
            tp1_pf: VectorBT Portfolio for TP1 (fixed SL/TP).
            tp2_pf: VectorBT Portfolio for TP2 (trailing SL).
            tp3_pf: VectorBT Portfolio for TP3 (trailing SL only).
            tp1_alloc: Fraction of position allocated to TP1 (default 0.40).
            tp2_alloc: Fraction of position allocated to TP2 (default 0.30).
            tp3_alloc: Fraction of position allocated to TP3 (default 0.30).
        """
        if not np.isclose(tp1_alloc + tp2_alloc + tp3_alloc, 1.0):
            raise ValueError("Allocations must sum to 1.0")

        self.tp1_pf = tp1_pf
        self.tp2_pf = tp2_pf
        self.tp3_pf = tp3_pf
        self.tp1_alloc = float(tp1_alloc)
        self.tp2_alloc = float(tp2_alloc)
        self.tp3_alloc = float(tp3_alloc)

    def returns(self) -> pd.Series:
        """Compute blended returns from all 3 sub-portfolios."""
        ret1 = self.tp1_pf.returns()
        ret2 = self.tp2_pf.returns()
        ret3 = self.tp3_pf.returns()

        if isinstance(ret1, pd.DataFrame):
            ret1 = ret1.iloc[:, 0]
        if isinstance(ret2, pd.DataFrame):
            ret2 = ret2.iloc[:, 0]
        if isinstance(ret3, pd.DataFrame):
            ret3 = ret3.iloc[:, 0]

        # Align and blend
        all_idx = ret1.index.union(ret2.index).union(ret3.index)
        ret1 = ret1.reindex(all_idx, fill_value=0.0)
        ret2 = ret2.reindex(all_idx, fill_value=0.0)
        ret3 = ret3.reindex(all_idx, fill_value=0.0)

        blended = self.tp1_alloc * ret1 + self.tp2_alloc * ret2 + self.tp3_alloc * ret3
        return blended

    def total_return(self) -> float:
        """Compute blended total return."""
        tr1 = float(self.tp1_pf.total_return())
        tr2 = float(self.tp2_pf.total_return())
        tr3 = float(self.tp3_pf.total_return())
        return self.tp1_alloc * tr1 + self.tp2_alloc * tr2 + self.tp3_alloc * tr3

    def value(self) -> pd.Series:
        """Compute blended portfolio value."""
        v1 = self.tp1_pf.value()
        v2 = self.tp2_pf.value()
        v3 = self.tp3_pf.value()

        if isinstance(v1, pd.DataFrame):
            v1 = v1.iloc[:, 0]
        if isinstance(v2, pd.DataFrame):
            v2 = v2.iloc[:, 0]
        if isinstance(v3, pd.DataFrame):
            v3 = v3.iloc[:, 0]

        all_idx = v1.index.union(v2.index).union(v3.index)
        v1 = v1.reindex(all_idx, fill_value=0.0)
        v2 = v2.reindex(all_idx, fill_value=0.0)
        v3 = v3.reindex(all_idx, fill_value=0.0)

        blended = self.tp1_alloc * v1 + self.tp2_alloc * v2 + self.tp3_alloc * v3
        return blended

    @property
    def trades(self):
        """Aggregate trade records from all 3 sub-portfolios, scaled by allocation."""

        class AggregatedTrades:
            def __init__(self, tp1_trades, tp2_trades, tp3_trades, tp1_a, tp2_a, tp3_a):
                has_tp1 = hasattr(tp1_trades, "records")
                has_tp2 = hasattr(tp2_trades, "records")
                has_tp3 = hasattr(tp3_trades, "records")
                self.tp1_records = tp1_trades.records if has_tp1 else None
                self.tp2_records = tp2_trades.records if has_tp2 else None
                self.tp3_records = tp3_trades.records if has_tp3 else None
                self.tp1_a = float(tp1_a)
                self.tp2_a = float(tp2_a)
                self.tp3_a = float(tp3_a)

            @property
            def records(self) -> pd.DataFrame:
                """Concatenate and scale trade records by allocation."""
                dfs = []

                if self.tp1_records is not None and len(self.tp1_records) > 0:
                    df1 = self.tp1_records.copy()
                    df1["pnl"] = df1["pnl"] * self.tp1_a
                    df1["pnl_pct"] = df1["pnl_pct"] * self.tp1_a
                    dfs.append(df1)

                if self.tp2_records is not None and len(self.tp2_records) > 0:
                    df2 = self.tp2_records.copy()
                    df2["pnl"] = df2["pnl"] * self.tp2_a
                    df2["pnl_pct"] = df2["pnl_pct"] * self.tp2_a
                    dfs.append(df2)

                if self.tp3_records is not None and len(self.tp3_records) > 0:
                    df3 = self.tp3_records.copy()
                    df3["pnl"] = df3["pnl"] * self.tp3_a
                    df3["pnl_pct"] = df3["pnl_pct"] * self.tp3_a
                    dfs.append(df3)

                if not dfs:
                    return pd.DataFrame()

                return pd.concat(dfs, axis=0, ignore_index=True)

        return AggregatedTrades(
            self.tp1_pf.trades,
            self.tp2_pf.trades,
            self.tp3_pf.trades,
            self.tp1_alloc,
            self.tp2_alloc,
            self.tp3_alloc,
        )


def _shift_signal(signal: pd.Series) -> pd.Series:
    """Shift signal forward by 1 bar to avoid same-bar fill."""
    return signal.vbt.fshift(1).fillna(False).astype(bool)


def _apply_entry_cooldown(signal: pd.Series, cooldown_bars: int) -> pd.Series:
    """Suppress repeated entries for a fixed number of bars after signal trigger."""
    if cooldown_bars <= 0:
        return signal.fillna(False).astype(bool)

    # Ensure signal is a Series and extract as numpy bool array
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0]

    raw = signal.fillna(False).astype(bool).to_numpy(dtype=np.bool_)
    filtered = np.zeros(len(raw), dtype=np.bool_)
    cooldown_remaining = 0

    for idx in range(len(raw)):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        if bool(raw[idx]) and cooldown_remaining == 0:
            filtered[idx] = True
            cooldown_remaining = cooldown_bars

    return pd.Series(filtered, index=signal.index, dtype=bool)


def _build_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    ema_fast: int,
    ema_medium: int,
    ema_slow: int,
    rsi_period: int,
    rsi_overbought: float,
    rsi_oversold: float,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    vol_window: int,
    vol_threshold: float,
    entry_cooldown_bars: int,
    portfolio_freq: str | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate long and short entry/exit signals based on momentum criteria.

    Returns: (long_entries, short_entries, long_exits, short_exits)
    """
    if not (ema_fast < ema_medium < ema_slow):
        raise ValueError("EMA windows must satisfy: fast < medium < slow")
    if rsi_overbought <= rsi_oversold:
        raise ValueError("rsi_overbought must be > rsi_oversold")
    if vol_threshold <= 0:
        raise ValueError("vol_threshold must be > 0")

    # EMA ribbon - FORCE extraction as numpy arrays
    emas = vbt.MA.run(close, window=[ema_fast, ema_medium, ema_slow], ewm=True).ma
    if not isinstance(emas, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from EMA run, got {type(emas)}")
    ema_f = emas.iloc[:, 0].values
    ema_m = emas.iloc[:, 1].values
    ema_s = emas.iloc[:, 2].values

    # RSI momentum - extract as numpy array
    rsi_result = vbt.RSI.run(close, window=rsi_period).rsi
    if isinstance(rsi_result, pd.DataFrame):
        rsi_values = rsi_result.iloc[:, 0].values
    else:
        rsi_values = rsi_result.values

    # MACD acceleration - FORCE extraction as numpy arrays
    macd_obj = vbt.MACD.run(
        close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal
    )
    macd_raw = macd_obj.macd
    signal_raw = macd_obj.signal
    macd_line = (
        macd_raw.iloc[:, 0].values
        if isinstance(macd_raw, pd.DataFrame)
        else macd_raw.values
    )
    signal_line = (
        signal_raw.iloc[:, 0].values
        if isinstance(signal_raw, pd.DataFrame)
        else signal_raw.values
    )
    macd_hist = np.nan_to_num(macd_line - signal_line, nan=0.0)

    close_values = close.values

    # Long signals - pure numpy operations, NO Series
    ema_aligned_long_arr = (ema_f > ema_m) & (ema_m > ema_s)
    rsi_aligned_long_arr = (rsi_values > 50) & (rsi_values < rsi_overbought)
    macd_aligned_long_arr = macd_hist > 0
    price_above_fast_arr = close_values > ema_f

    long_entries = pd.Series(
        ema_aligned_long_arr
        & rsi_aligned_long_arr
        & macd_aligned_long_arr
        & price_above_fast_arr,
        index=close.index,
        dtype=bool,
    )
    long_entries = _apply_entry_cooldown(long_entries, int(entry_cooldown_bars))

    # Short signals - pure numpy operations, NO Series
    ema_aligned_short_arr = (ema_f < ema_m) & (ema_m < ema_s)
    rsi_aligned_short_arr = (rsi_values < 50) & (rsi_values > rsi_oversold)
    macd_aligned_short_arr = macd_hist < 0
    price_below_fast_arr = close_values < ema_f

    short_entries = pd.Series(
        ema_aligned_short_arr
        & rsi_aligned_short_arr
        & macd_aligned_short_arr
        & price_below_fast_arr,
        index=close.index,
        dtype=bool,
    )
    short_entries = _apply_entry_cooldown(short_entries, int(entry_cooldown_bars))

    # Exit signals - pure numpy operations, NO Series
    long_exits = pd.Series(
        ((ema_f < ema_m) | (close_values < ema_f)).astype(bool),
        index=close.index,
        dtype=bool,
    )
    short_exits = pd.Series(
        ((ema_f > ema_m) | (close_values > ema_f)).astype(bool),
        index=close.index,
        dtype=bool,
    )

    # Warmup period - use explicit integer comparison, NOT Series
    warmup = max(int(ema_slow), int(rsi_period), int(macd_slow + macd_signal))
    warmup_end = min(warmup, len(long_entries)) if warmup > 0 else 0
    if warmup_end > 0:
        long_entries.iloc[:warmup_end] = False
        short_entries.iloc[:warmup_end] = False
        long_exits.iloc[:warmup_end] = False
        short_exits.iloc[:warmup_end] = False

    return (
        long_entries.astype(bool),
        short_entries.astype(bool),
        long_exits.astype(bool),
        short_exits.astype(bool),
    )


def _risk_arrays(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    init_cash: float,
    atr_window: int,
    sl_atr_multiple: float,
) -> tuple[pd.Series, pd.Series]:
    """Compute ATR-based stop distance and position sizing."""
    if atr_window <= 1:
        raise ValueError("atr_window must be > 1")
    if sl_atr_multiple <= 0:
        raise ValueError("sl_atr_multiple must be > 0")

    atr = vbt.ATR.run(high, low, close, window=atr_window).atr.astype(float)
    stop_distance = (atr * float(sl_atr_multiple)).astype(float)
    safe_stop = stop_distance.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return atr, safe_stop


def run(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    init_cash: float = 10_000.0,
    ema_fast: int = 8,
    ema_medium: int = 13,
    ema_slow: int = 21,
    rsi_period: int = 7,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    vol_window: int = 20,
    vol_threshold: float = 1.5,
    atr_window: int = 14,
    sl_atr_multiple: float = 1.0,
    tp1_multiple: float = 1.5,
    tp2_multiple: float = 3.0,
    tp3_trail_multiple: float = 1.5,
    tp1_allocation: float = 0.40,
    tp2_allocation: float = 0.30,
    tp3_allocation: float = 0.30,
    entry_cooldown_bars: int = 2,
    fees: float = 0.0005,
    slippage: float = 0.0001,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    max_size: Any | None = None,
    position_sizes: Any | None = None,
    portfolio_freq: str | None = None,
) -> CombinedPortfolio:
    """
    Run momentum scalping strategy with 3 TP levels via split sub-portfolios.

    Each sub-portfolio receives a fraction of the position with different TP/SL configs:
    - TP1 (40%): Fixed SL and moderate TP for quick scalps.
    - TP2 (30%): Trailing SL with higher TP for runners.
    - TP3 (30%): Trailing SL only (no TP cap) for home runs.

    Args:
        close: Close price series.
        high: High price series.
        low: Low price series.
        init_cash: Initial portfolio cash.
        ema_fast: Fast EMA window.
        ema_medium: Medium EMA window.
        ema_slow: Slow EMA window.
        rsi_period: RSI period.
        rsi_overbought: RSI overbought threshold.
        rsi_oversold: RSI oversold threshold.
        macd_fast: MACD fast period.
        macd_slow: MACD slow period.
        macd_signal: MACD signal period.
        vol_window: Volume MA window.
        vol_threshold: Volume surge threshold.
        atr_window: ATR window for stops.
        sl_atr_multiple: SL distance as ATR multiple.
        tp1_multiple: TP1 distance as SL multiple.
        tp2_multiple: TP2 distance as SL multiple.
        tp3_trail_multiple: TP3 trailing stop as ATR multiple.
        tp1_allocation: Fraction allocated to TP1.
        tp2_allocation: Fraction allocated to TP2.
        tp3_allocation: Fraction allocated to TP3.
        entry_cooldown_bars: Bars to wait between entries.
        fees: Commission as fraction.
        slippage: Slippage as fraction.
        fixed_fees: Fixed fee per order.
        next_bar_execution: If True, shift signals forward 1 bar.
        max_size: Per-bar max position size (volume constraint).
        position_sizes: Custom position sizes (for Kelly).
        portfolio_freq: Portfolio frequency (e.g., "15m").

    Returns:
        CombinedPortfolio aggregating 3 sub-portfolios.
    """
    # Generate signals
    long_entries, short_entries, long_exits, short_exits = _build_signals(
        close,
        high,
        low,
        ema_fast=int(ema_fast),
        ema_medium=int(ema_medium),
        ema_slow=int(ema_slow),
        rsi_period=int(rsi_period),
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        vol_window=int(vol_window),
        vol_threshold=float(vol_threshold),
        entry_cooldown_bars=int(entry_cooldown_bars),
        portfolio_freq=portfolio_freq,
    )

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

    # Compute ATR-based stops
    atr, stop_distance = _risk_arrays(
        high,
        low,
        close,
        init_cash=init_cash,
        atr_window=int(atr_window),
        sl_atr_multiple=float(sl_atr_multiple),
    )

    # Position sizes: allocate to each TP level
    if position_sizes is not None:
        tp1_size = position_sizes * tp1_allocation
        tp2_size = position_sizes * tp2_allocation
        tp3_size = position_sizes * tp3_allocation
    else:
        base_size = (
            (init_cash * 0.005 / stop_distance.replace(0, np.nan))
            .fillna(0.0)
            .clip(lower=0.0)
        )
        tp1_size = base_size * tp1_allocation
        tp2_size = base_size * tp2_allocation
        tp3_size = base_size * tp3_allocation

    portfolio_kwargs = {
        "init_cash": (init_cash * tp1_allocation),  # Each sub-portfolio allocated cash
        "fees": fees,
        "fixed_fees": fixed_fees,
        "slippage": slippage,
        "freq": portfolio_freq or DEFAULT_TIMEFRAME,
    }
    if safe_max_size is not None:
        portfolio_kwargs["max_size"] = safe_max_size

    # TP1: Fixed SL/TP for quick scalps (40%)
    tp1_pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries.astype(bool),
        exits=long_exits.astype(bool),
        short_entries=short_entries.astype(bool),
        short_exits=short_exits.astype(bool),
        size=tp1_size,
        sl_stop=stop_distance,
        tp_stop=atr * float(tp1_multiple),
        **portfolio_kwargs,
    )

    # TP2: Trailing SL with moderate TP (30%)
    portfolio_kwargs["init_cash"] = init_cash * tp2_allocation
    tp2_pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries.astype(bool),
        exits=long_exits.astype(bool),
        short_entries=short_entries.astype(bool),
        short_exits=short_exits.astype(bool),
        size=tp2_size,
        sl_stop=stop_distance,
        tp_stop=atr * float(tp2_multiple),
        sl_trail=True,  # Enable trailing stop
        **portfolio_kwargs,
    )

    # TP3: Trailing SL only, no TP cap for home runs (30%)
    portfolio_kwargs["init_cash"] = init_cash * tp3_allocation
    tp3_pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries.astype(bool),
        exits=long_exits.astype(bool),
        short_entries=short_entries.astype(bool),
        short_exits=short_exits.astype(bool),
        size=tp3_size,
        sl_stop=atr * float(tp3_trail_multiple),
        sl_trail=True,  # Trailing SL only, no TP
        **portfolio_kwargs,
    )

    return CombinedPortfolio(
        tp1_pf,
        tp2_pf,
        tp3_pf,
        tp1_alloc=tp1_allocation,
        tp2_alloc=tp2_allocation,
        tp3_alloc=tp3_allocation,
    )


def run_scan(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    ema_fast_windows: Iterable[int] = (8,),
    ema_medium_windows: Iterable[int] = (13,),
    ema_slow_windows: Iterable[int] = (21,),
    rsi_period_values: Iterable[int] = (7,),
    vol_threshold_values: Iterable[float] = (1.5,),
    atr_window_values: Iterable[int] = (14,),
    sl_atr_multiple_values: Iterable[float] = (1.0,),
    tp1_multiple_values: Iterable[float] = (1.5,),
    tp2_multiple_values: Iterable[float] = (3.0,),
    tp3_trail_multiple_values: Iterable[float] = (1.5,),
    entry_cooldown_bars_values: Iterable[int] = (2,),
    init_cash: float = 10_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0001,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run momentum scalp strategy scan across parameter grids.

    Primarily for pytest or integration testing. Optuna integration handles
    parameter search for hyperparameter optimization.
    """
    # Placeholder: for now, just run with first parameter set from each grid
    # In practice, Optuna will handle the search via run_hyperparameter_search.py
    ema_f = next(iter(ema_fast_windows))
    ema_m = next(iter(ema_medium_windows))
    ema_s = next(iter(ema_slow_windows))
    rsi_p = next(iter(rsi_period_values))
    vol_t = next(iter(vol_threshold_values))
    atr_w = next(iter(atr_window_values))
    sl_mult = next(iter(sl_atr_multiple_values))
    tp1_m = next(iter(tp1_multiple_values))
    tp2_m = next(iter(tp2_multiple_values))
    tp3_m = next(iter(tp3_trail_multiple_values))
    cooldown = next(iter(entry_cooldown_bars_values))

    return run(
        close,
        high,
        low,
        init_cash=init_cash,
        ema_fast=int(ema_f),
        ema_medium=int(ema_m),
        ema_slow=int(ema_s),
        rsi_period=int(rsi_p),
        vol_threshold=float(vol_t),
        atr_window=int(atr_w),
        sl_atr_multiple=float(sl_mult),
        tp1_multiple=float(tp1_m),
        tp2_multiple=float(tp2_m),
        tp3_trail_multiple=float(tp3_m),
        entry_cooldown_bars=int(cooldown),
        fees=fees,
        slippage=slippage,
        fixed_fees=fixed_fees,
        next_bar_execution=next_bar_execution,
        max_size=max_size,
        portfolio_freq=portfolio_freq,
    )
