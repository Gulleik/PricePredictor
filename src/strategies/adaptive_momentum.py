"""Adaptive momentum strategy with scored signals and 2-tier TP architecture."""

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


class DualPortfolio:
    """
    Aggregate wrapper for 2 sub-portfolios with different TP levels.

    Provides interface methods (.returns(), .total_return(), .value(), .trades)
    compatible with Optuna integration and Kelly sizing estimation.
    """

    def __init__(
        self,
        tp1_pf: Any,
        tp2_pf: Any,
        tp1_alloc: float = 0.60,
        tp2_alloc: float = 0.40,
    ):
        if not np.isclose(tp1_alloc + tp2_alloc, 1.0):
            raise ValueError("Allocations must sum to 1.0")

        self.tp1_pf = tp1_pf
        self.tp2_pf = tp2_pf
        self.tp1_alloc = float(tp1_alloc)
        self.tp2_alloc = float(tp2_alloc)

    def returns(self) -> pd.Series:
        """Compute blended returns from both sub-portfolios."""
        ret1 = self.tp1_pf.returns()
        ret2 = self.tp2_pf.returns()

        if isinstance(ret1, pd.DataFrame):
            ret1 = ret1.iloc[:, 0]
        if isinstance(ret2, pd.DataFrame):
            ret2 = ret2.iloc[:, 0]

        all_idx = ret1.index.union(ret2.index)
        ret1 = ret1.reindex(all_idx, fill_value=0.0)
        ret2 = ret2.reindex(all_idx, fill_value=0.0)

        return self.tp1_alloc * ret1 + self.tp2_alloc * ret2

    def total_return(self) -> float:
        """Compute blended total return."""
        tr1 = float(self.tp1_pf.total_return())
        tr2 = float(self.tp2_pf.total_return())
        return self.tp1_alloc * tr1 + self.tp2_alloc * tr2

    def value(self) -> pd.Series:
        """Compute blended portfolio value."""
        v1 = self.tp1_pf.value()
        v2 = self.tp2_pf.value()

        if isinstance(v1, pd.DataFrame):
            v1 = v1.iloc[:, 0]
        if isinstance(v2, pd.DataFrame):
            v2 = v2.iloc[:, 0]

        all_idx = v1.index.union(v2.index)
        v1 = v1.reindex(all_idx, fill_value=0.0)
        v2 = v2.reindex(all_idx, fill_value=0.0)

        return self.tp1_alloc * v1 + self.tp2_alloc * v2

    @property
    def trades(self):
        """Aggregate trade records from both sub-portfolios."""

        class AggregatedTrades:
            def __init__(self, tp1_trades, tp2_trades, tp1_a, tp2_a):
                has_tp1 = hasattr(tp1_trades, "records")
                has_tp2 = hasattr(tp2_trades, "records")
                self.tp1_records = tp1_trades.records if has_tp1 else None
                self.tp2_records = tp2_trades.records if has_tp2 else None
                self.tp1_a = float(tp1_a)
                self.tp2_a = float(tp2_a)

            @property
            def records(self) -> pd.DataFrame:
                """Concatenate and scale trade records by allocation."""
                dfs = []

                if self.tp1_records is not None and len(self.tp1_records) > 0:
                    df1 = self.tp1_records.copy()
                    df1["pnl"] = df1["pnl"] * self.tp1_a
                    if "pnl_pct" in df1.columns:
                        df1["pnl_pct"] = df1["pnl_pct"] * self.tp1_a
                    dfs.append(df1)

                if self.tp2_records is not None and len(self.tp2_records) > 0:
                    df2 = self.tp2_records.copy()
                    df2["pnl"] = df2["pnl"] * self.tp2_a
                    if "pnl_pct" in df2.columns:
                        df2["pnl_pct"] = df2["pnl_pct"] * self.tp2_a
                    dfs.append(df2)

                if not dfs:
                    return pd.DataFrame()

                return pd.concat(dfs, axis=0, ignore_index=True)

        return AggregatedTrades(
            self.tp1_pf.trades,
            self.tp2_pf.trades,
            self.tp1_alloc,
            self.tp2_alloc,
        )


def _shift_signal(signal: pd.Series) -> pd.Series:
    """Shift signal forward by 1 bar to avoid same-bar fill."""
    return signal.vbt.fshift(1).fillna(False).astype(bool)


def _apply_entry_cooldown(signal: pd.Series, cooldown_bars: int) -> pd.Series:
    """Suppress repeated entries for a fixed number of bars after signal trigger."""
    if cooldown_bars <= 0:
        return signal.fillna(False).astype(bool)

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


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> np.ndarray:
    """Compute Average Directional Index using Wilder's smoothing."""
    h = high.values.astype(float)
    l_ = low.values.astype(float)
    c = close.values.astype(float)
    n = len(h)

    # True Range
    tr = np.zeros(n)
    tr[0] = h[0] - l_[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l_[i], abs(h[i] - c[i - 1]), abs(l_[i] - c[i - 1]))

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = h[i] - h[i - 1]
        down_move = l_[i - 1] - l_[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder's smoothing
    smooth_tr = np.zeros(n)
    smooth_plus_dm = np.zeros(n)
    smooth_minus_dm = np.zeros(n)

    # Initialize with simple sum of first `period` values
    if n >= period:
        smooth_tr[period - 1] = np.sum(tr[:period])
        smooth_plus_dm[period - 1] = np.sum(plus_dm[:period])
        smooth_minus_dm[period - 1] = np.sum(minus_dm[:period])

        for i in range(period, n):
            smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]
            smooth_plus_dm[i] = (
                smooth_plus_dm[i - 1] - smooth_plus_dm[i - 1] / period + plus_dm[i]
            )
            smooth_minus_dm[i] = (
                smooth_minus_dm[i - 1] - smooth_minus_dm[i - 1] / period + minus_dm[i]
            )

    # DI+, DI-, DX
    safe_tr = np.where(smooth_tr > 0, smooth_tr, 1.0)
    plus_di = 100.0 * smooth_plus_dm / safe_tr
    minus_di = 100.0 * smooth_minus_dm / safe_tr

    di_sum = plus_di + minus_di
    safe_di_sum = np.where(di_sum > 0, di_sum, 1.0)
    dx = 100.0 * np.abs(plus_di - minus_di) / safe_di_sum

    # ADX = smoothed DX
    adx = np.zeros(n)
    if n >= 2 * period - 1:
        adx[2 * period - 2] = np.mean(dx[period - 1 : 2 * period - 1])
        for i in range(2 * period - 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


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
    signal_threshold: float,
    adx_period: int,
    adx_threshold: float,
    atr_percentile_min: float,
    atr_percentile_window: int,
    entry_cooldown_bars: int,
    volume: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate scored long/short entry signals and exits.

    Each indicator contributes a 0.0-1.0 score. The composite score must
    exceed signal_threshold for entry. This replaces the rigid 4-way AND.

    Returns: (long_entries, short_entries, long_exits, short_exits)
    """
    if not (ema_fast < ema_medium < ema_slow):
        raise ValueError("EMA windows must satisfy: fast < medium < slow")
    if rsi_overbought <= rsi_oversold:
        raise ValueError("rsi_overbought must be > rsi_oversold")
    if macd_fast >= macd_slow:
        raise ValueError("macd_fast must be < macd_slow")

    n = len(close)

    # --- Indicators ---
    # EMA ribbon
    emas = vbt.MA.run(close, window=[ema_fast, ema_medium, ema_slow], ewm=True).ma
    if not isinstance(emas, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from EMA run, got {type(emas)}")
    ema_f = emas.iloc[:, 0].values
    ema_m = emas.iloc[:, 1].values
    ema_s = emas.iloc[:, 2].values

    # RSI
    rsi_result = vbt.RSI.run(close, window=rsi_period).rsi
    rsi_values = (
        rsi_result.iloc[:, 0].values
        if isinstance(rsi_result, pd.DataFrame)
        else rsi_result.values
    )

    # MACD
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
    # MACD histogram change (rising/falling)
    macd_hist_prev = np.roll(macd_hist, 1)
    macd_hist_prev[0] = 0.0
    macd_hist_rising = macd_hist > macd_hist_prev

    # ADX
    adx_values = _compute_adx(high, low, close, adx_period)

    # ATR for regime filter
    atr_result = vbt.ATR.run(high, low, close, window=14).atr
    atr_values = (
        atr_result.iloc[:, 0].values
        if isinstance(atr_result, pd.DataFrame)
        else atr_result.values
    )
    # Rolling ATR percentile
    atr_series = pd.Series(atr_values, index=close.index)
    atr_pctile = atr_series.rolling(atr_percentile_window, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    atr_pctile_values = atr_pctile.values
    regime_ok = atr_pctile_values >= (atr_percentile_min / 100.0)

    # Volume surge
    if volume is not None:
        vol_values = volume.values.astype(float)
        vol_ma = pd.Series(vol_values, index=close.index).rolling(vol_window).mean()
        vol_ma_values = vol_ma.values
        safe_vol_ma = np.where(vol_ma_values > 0, vol_ma_values, 1.0)
        vol_ratio = vol_values / safe_vol_ma
    else:
        vol_ratio = np.ones(n)

    close_values = close.values

    # --- LONG SIGNAL SCORING ---
    # EMA score: 1.0 if fully aligned, 0.5 if fast > medium only
    long_ema_full = (ema_f > ema_m) & (ema_m > ema_s)
    long_ema_partial = (ema_f > ema_m) & ~(ema_m > ema_s)
    long_ema_score = np.where(long_ema_full, 1.0, np.where(long_ema_partial, 0.5, 0.0))

    # RSI score: 1.0 in ideal zone (50-70), 0.5 near edges (45-50 or 70-75)
    long_rsi_ideal = (rsi_values > 50) & (rsi_values < rsi_overbought)
    long_rsi_edge = ((rsi_values >= 45) & (rsi_values <= 50)) | (
        (rsi_values >= rsi_overbought) & (rsi_values <= rsi_overbought + 5)
    )
    long_rsi_score = np.where(long_rsi_ideal, 1.0, np.where(long_rsi_edge, 0.5, 0.0))

    # MACD score: 1.0 if hist > 0 and rising, 0.5 if just > 0
    long_macd_strong = (macd_hist > 0) & macd_hist_rising
    long_macd_ok = (macd_hist > 0) & ~macd_hist_rising
    long_macd_score = np.where(long_macd_strong, 1.0, np.where(long_macd_ok, 0.5, 0.0))

    # Volume score: 1.0 if > 2x avg, 0.5 if > 1.5x avg
    long_vol_strong = vol_ratio >= 2.0
    long_vol_moderate = (vol_ratio >= 1.5) & (vol_ratio < 2.0)
    long_vol_score = np.where(
        long_vol_strong, 1.0, np.where(long_vol_moderate, 0.5, 0.0)
    )

    # Composite long score (average of 4 components)
    long_score = (
        long_ema_score + long_rsi_score + long_macd_score + long_vol_score
    ) / 4.0

    # Long entry: score >= threshold, price > fast EMA, ADX ok, regime ok
    long_raw = (
        (long_score >= signal_threshold)
        & (close_values > ema_f)
        & (adx_values >= adx_threshold)
        & regime_ok
    )

    long_entries = pd.Series(long_raw, index=close.index, dtype=bool)
    long_entries = _apply_entry_cooldown(long_entries, int(entry_cooldown_bars))

    # --- SHORT SIGNAL SCORING ---
    # EMA score (inverted)
    short_ema_full = (ema_f < ema_m) & (ema_m < ema_s)
    short_ema_partial = (ema_f < ema_m) & ~(ema_m < ema_s)
    short_ema_score = np.where(
        short_ema_full, 1.0, np.where(short_ema_partial, 0.5, 0.0)
    )

    # RSI score (inverted for shorts)
    short_rsi_ideal = (rsi_values < 50) & (rsi_values > rsi_oversold)
    short_rsi_edge = ((rsi_values <= 55) & (rsi_values >= 50)) | (
        (rsi_values <= rsi_oversold) & (rsi_values >= rsi_oversold - 5)
    )
    short_rsi_score = np.where(short_rsi_ideal, 1.0, np.where(short_rsi_edge, 0.5, 0.0))

    # MACD score (inverted)
    short_macd_strong = (macd_hist < 0) & ~macd_hist_rising
    short_macd_ok = (macd_hist < 0) & macd_hist_rising
    short_macd_score = np.where(
        short_macd_strong, 1.0, np.where(short_macd_ok, 0.5, 0.0)
    )

    # Volume score (same for shorts)
    short_vol_score = long_vol_score

    short_score = (
        short_ema_score + short_rsi_score + short_macd_score + short_vol_score
    ) / 4.0

    short_raw = (
        (short_score >= signal_threshold)
        & (close_values < ema_f)
        & (adx_values >= adx_threshold)
        & regime_ok
    )

    short_entries = pd.Series(short_raw, index=close.index, dtype=bool)
    short_entries = _apply_entry_cooldown(short_entries, int(entry_cooldown_bars))

    # --- EXIT SIGNALS ---
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

    # Warmup period
    warmup = max(
        int(ema_slow),
        int(rsi_period),
        int(macd_slow + macd_signal),
        2 * int(adx_period),
        int(atr_percentile_window),
    )
    warmup_end = min(warmup, n) if warmup > 0 else 0
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
    """Compute ATR-based stop distance."""
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
    signal_threshold: float = 0.5,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    atr_percentile_min: float = 30.0,
    atr_percentile_window: int = 100,
    atr_window: int = 14,
    sl_atr_multiple: float = 1.5,
    tp1_multiple: float = 2.0,
    tp2_trail_multiple: float = 1.5,
    tp1_allocation: float = 0.60,
    tp2_allocation: float = 0.40,
    entry_cooldown_bars: int = 2,
    fees: float = 0.0005,
    slippage: float = 0.0001,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    max_size: Any | None = None,
    position_sizes: Any | None = None,
    portfolio_freq: str | None = None,
    volume: pd.Series | None = None,
    leverage: float = 1.0,
) -> DualPortfolio:
    """
    Run adaptive momentum strategy with scored signals and 2-tier TP.

    TP1 (60%): Fixed SL/TP for quick scalps.
    TP2 (40%): Trailing SL, no TP cap for runners.

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
        signal_threshold: Composite score threshold for entry (0.0-1.0).
        adx_period: ADX calculation period.
        adx_threshold: Minimum ADX value for trend strength.
        atr_percentile_min: Minimum ATR percentile for regime filter.
        atr_percentile_window: Rolling window for ATR percentile.
        atr_window: ATR window for stops.
        sl_atr_multiple: SL distance as ATR multiple.
        tp1_multiple: TP1 distance as SL multiple.
        tp2_trail_multiple: TP2 trailing stop as ATR multiple.
        tp1_allocation: Fraction allocated to TP1 (default 0.60).
        tp2_allocation: Fraction allocated to TP2 (default 0.40).
        entry_cooldown_bars: Bars to wait between entries.
        fees: Commission as fraction.
        slippage: Slippage as fraction.
        fixed_fees: Fixed fee per order.
        next_bar_execution: If True, shift signals forward 1 bar.
        max_size: Per-bar max position size (volume constraint).
        position_sizes: Custom position sizes (for Kelly).
        portfolio_freq: Portfolio frequency (e.g., "15m").
        volume: Volume series for volume surge filter.

    Returns:
        DualPortfolio aggregating 2 sub-portfolios.
    """
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
        signal_threshold=float(signal_threshold),
        adx_period=int(adx_period),
        adx_threshold=float(adx_threshold),
        atr_percentile_min=float(atr_percentile_min),
        atr_percentile_window=int(atr_percentile_window),
        entry_cooldown_bars=int(entry_cooldown_bars),
        volume=volume,
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

    # Position sizes
    if position_sizes is not None:
        tp1_size = position_sizes * tp1_allocation
        tp2_size = position_sizes * tp2_allocation
    else:
        base_size = (
            (init_cash * 0.005 / stop_distance.replace(0, np.nan))
            .fillna(0.0)
            .clip(lower=0.0)
        )
        # Apply leverage only to internally-computed sizes
        tp1_size = base_size * tp1_allocation * leverage
        tp2_size = base_size * tp2_allocation * leverage

    portfolio_kwargs = {
        "init_cash": (init_cash * tp1_allocation),
        "fees": fees,
        "fixed_fees": fixed_fees,
        "slippage": slippage,
        "freq": portfolio_freq or DEFAULT_TIMEFRAME,
    }
    if safe_max_size is not None:
        portfolio_kwargs["max_size"] = safe_max_size

    # TP1: Fixed SL/TP for quick wins (60%)
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

    # TP2: Trailing SL only, no TP cap for runners (40%)
    portfolio_kwargs["init_cash"] = init_cash * tp2_allocation
    tp2_pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries.astype(bool),
        exits=long_exits.astype(bool),
        short_entries=short_entries.astype(bool),
        short_exits=short_exits.astype(bool),
        size=tp2_size,
        sl_stop=atr * float(tp2_trail_multiple),
        sl_trail=True,
        **portfolio_kwargs,
    )

    return DualPortfolio(
        tp1_pf,
        tp2_pf,
        tp1_alloc=tp1_allocation,
        tp2_alloc=tp2_allocation,
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
    signal_threshold_values: Iterable[float] = (0.5,),
    adx_period_values: Iterable[int] = (14,),
    adx_threshold_values: Iterable[float] = (20.0,),
    atr_percentile_min_values: Iterable[float] = (30.0,),
    atr_window_values: Iterable[int] = (14,),
    sl_atr_multiple_values: Iterable[float] = (1.5,),
    tp1_multiple_values: Iterable[float] = (2.0,),
    tp2_trail_multiple_values: Iterable[float] = (1.5,),
    entry_cooldown_bars_values: Iterable[int] = (2,),
    init_cash: float = 10_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0001,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    max_size: np.ndarray | None = None,
    portfolio_freq: str | None = None,
    volume: pd.Series | None = None,
    **kwargs: Any,
) -> Any:
    """Run adaptive momentum strategy scan with first parameter from each grid."""
    return run(
        close,
        high,
        low,
        init_cash=init_cash,
        ema_fast=int(next(iter(ema_fast_windows))),
        ema_medium=int(next(iter(ema_medium_windows))),
        ema_slow=int(next(iter(ema_slow_windows))),
        rsi_period=int(next(iter(rsi_period_values))),
        vol_threshold=float(next(iter(vol_threshold_values))),
        signal_threshold=float(next(iter(signal_threshold_values))),
        adx_period=int(next(iter(adx_period_values))),
        adx_threshold=float(next(iter(adx_threshold_values))),
        atr_percentile_min=float(next(iter(atr_percentile_min_values))),
        atr_window=int(next(iter(atr_window_values))),
        sl_atr_multiple=float(next(iter(sl_atr_multiple_values))),
        tp1_multiple=float(next(iter(tp1_multiple_values))),
        tp2_trail_multiple=float(next(iter(tp2_trail_multiple_values))),
        entry_cooldown_bars=int(next(iter(entry_cooldown_bars_values))),
        fees=fees,
        slippage=slippage,
        fixed_fees=fixed_fees,
        next_bar_execution=next_bar_execution,
        max_size=max_size,
        portfolio_freq=portfolio_freq,
        volume=volume,
    )
