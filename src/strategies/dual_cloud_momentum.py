"""Dual-Cloud Momentum: HTF trend filter + LTF pullback trigger with 4-tier Fib TP."""

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.config import DEFAULT_TIMEFRAME
from src.strategies.common import (
    apply_valid_mask,
    sanitize_max_size,
)

# ---------------------------------------------------------------------------
# QuadPortfolio: 4-tier TP aggregation wrapper
# ---------------------------------------------------------------------------


class QuadPortfolio:
    """Aggregate wrapper for 4 sub-portfolios with Fibonacci TP levels.

    Provides interface methods (.returns(), .total_return(), .value(), .trades)
    compatible with Optuna integration and Kelly sizing estimation.
    """

    def __init__(
        self,
        tp1_pf: Any,
        tp2_pf: Any,
        tp3_pf: Any,
        tp4_pf: Any,
        tp1_alloc: float = 0.25,
        tp2_alloc: float = 0.25,
        tp3_alloc: float = 0.25,
        tp4_alloc: float = 0.25,
    ):
        total = tp1_alloc + tp2_alloc + tp3_alloc + tp4_alloc
        if not np.isclose(total, 1.0):
            raise ValueError("Allocations must sum to 1.0")

        self._pfs = [tp1_pf, tp2_pf, tp3_pf, tp4_pf]
        self._allocs = [
            float(tp1_alloc),
            float(tp2_alloc),
            float(tp3_alloc),
            float(tp4_alloc),
        ]

    # -- helpers ----------------------------------------------------------

    def _blend_series(self, getter: str) -> pd.Series:
        """Blend a per-portfolio Series by allocation weight."""
        series_list: list[pd.Series] = []
        for pf in self._pfs:
            s = getattr(pf, getter)()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            series_list.append(s)

        all_idx = series_list[0].index
        for s in series_list[1:]:
            all_idx = all_idx.union(s.index)

        aligned = [s.reindex(all_idx, fill_value=0.0) for s in series_list]
        blended = sum(a * s for a, s in zip(self._allocs, aligned))
        return blended

    # -- public interface -------------------------------------------------

    def returns(self) -> pd.Series:
        """Compute blended returns from all 4 sub-portfolios."""
        return self._blend_series("returns")

    def total_return(self) -> float:
        """Compute blended total return."""
        return sum(
            a * float(pf.total_return()) for a, pf in zip(self._allocs, self._pfs)
        )

    def value(self) -> pd.Series:
        """Compute blended portfolio value."""
        return self._blend_series("value")

    @property
    def trades(self):
        """Aggregate trade records from all 4 sub-portfolios, scaled by allocation."""
        allocs = self._allocs
        pfs = self._pfs

        class _AggregatedTrades:
            def __init__(self) -> None:
                self._items: list[tuple[Any, float]] = []
                for pf, a in zip(pfs, allocs):
                    if hasattr(pf.trades, "records"):
                        self._items.append((pf.trades, a))

            @property
            def records(self) -> pd.DataFrame:
                dfs: list[pd.DataFrame] = []
                for trades_obj, alloc in self._items:
                    recs = trades_obj.records
                    if recs is None or len(recs) == 0:
                        continue
                    df = recs.copy()
                    df["pnl"] = df["pnl"] * alloc
                    if "pnl_pct" in df.columns:
                        df["pnl_pct"] = df["pnl_pct"] * alloc
                    dfs.append(df)
                if not dfs:
                    return pd.DataFrame()
                return pd.concat(dfs, axis=0, ignore_index=True)

        return _AggregatedTrades()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shift_signal(sig: pd.Series) -> pd.Series:
    """Shift boolean signal forward one bar (next-bar execution)."""
    return sig.shift(1, fill_value=False).astype(bool)


def _resample_to_htf(
    market_data: pd.DataFrame,
    htf_rule: str = "1h",
) -> pd.DataFrame:
    """Resample OHLCV data to higher timeframe using standard OHLC aggregation."""
    return (
        market_data.resample(htf_rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def _compute_htf_bias(
    htf_close: pd.Series,
    trend_ema_fast: int,
    trend_ema_slow: int,
) -> pd.Series:
    """Classify HTF trend bias as 'bull', 'bear', or 'neutral'.

    Bullish: close > EMA_fast AND EMA_fast > EMA_slow
    Bearish: close < EMA_fast AND EMA_fast < EMA_slow
    """
    ema_fast = htf_close.ewm(span=trend_ema_fast, adjust=False).mean()
    ema_slow = htf_close.ewm(span=trend_ema_slow, adjust=False).mean()

    bullish = (htf_close > ema_fast) & (ema_fast > ema_slow)
    bearish = (htf_close < ema_fast) & (ema_fast < ema_slow)

    bias = pd.Series("neutral", index=htf_close.index, dtype="object")
    bias[bullish] = "bull"
    bias[bearish] = "bear"
    return bias


def _build_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    htf_bias_ltf: pd.Series,
    *,
    ltf_ema_window: int,
    rsi_length: int,
    rsi_level_long: float,
    rsi_level_short: float,
    enable_weekend_trading: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate long/short entry and exit signals on the LTF (15m) data.

    Returns:
        (long_entries, short_entries, long_exits, short_exits)
    """
    # LTF EMA (same period as HTF fast EMA, applied to LTF data)
    ema_ltf = close.ewm(span=ltf_ema_window, adjust=False).mean()

    # RSI on LTF
    rsi = vbt.RSI.run(close, window=rsi_length).rsi

    # RSI trending direction
    rsi_trending_up = rsi > rsi.shift(1)
    rsi_trending_down = rsi < rsi.shift(1)

    # EMA touch: candle reaches the EMA level
    ema_touch_long = low <= ema_ltf  # price pulls back to/below EMA
    ema_touch_short = high >= ema_ltf  # price rallies to/above EMA

    # Combine entry conditions
    long_entries = (
        (htf_bias_ltf == "bull")
        & ema_touch_long
        & rsi_trending_up
        & (rsi > rsi_level_long)
    )
    short_entries = (
        (htf_bias_ltf == "bear")
        & ema_touch_short
        & rsi_trending_down
        & (rsi < rsi_level_short)
    )

    # Weekend filter
    if not enable_weekend_trading:
        weekday = close.index.dayofweek
        is_weekday = weekday < 5
        long_entries = long_entries & is_weekday
        short_entries = short_entries & is_weekday

    # Exits: bias reversal (HTF flips against position)
    long_exits = htf_bias_ltf == "bear"
    short_exits = htf_bias_ltf == "bull"

    return (
        long_entries.astype(bool).fillna(False),
        short_entries.astype(bool).fillna(False),
        long_exits.astype(bool).fillna(False),
        short_exits.astype(bool).fillna(False),
    )


def _compute_stop_distances(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    atr_length: int,
    sl_atr_multiplier: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ATR and directional stop-loss distances.

    Returns:
        (atr, long_sl_distance, short_sl_distance)
        Distances are absolute price distances (positive).
    """
    atr = vbt.ATR.run(high, low, close, window=atr_length).atr.astype(float)

    # Long SL: lowest low of last 3 candles - ATR * multiplier
    lowest_low_3 = low.rolling(3, min_periods=1).min()
    long_sl_level = lowest_low_3 - atr * sl_atr_multiplier
    long_sl_dist = (close - long_sl_level).clip(lower=0.0)

    # Short SL: highest high of last 3 candles + ATR * multiplier
    highest_high_3 = high.rolling(3, min_periods=1).max()
    short_sl_level = highest_high_3 + atr * sl_atr_multiplier
    short_sl_dist = (short_sl_level - close).clip(lower=0.0)

    # Use symmetric stop: average of long and short distances as a
    # unified fractional stop for VectorBT (which applies the same stop
    # to both directions in a single from_signals call).
    # Convert to fractional (relative to close) for sl_stop parameter.
    avg_sl_dist = (long_sl_dist + short_sl_dist) / 2.0
    avg_sl_dist = avg_sl_dist.replace(0, np.nan).fillna(atr * sl_atr_multiplier)

    return atr, avg_sl_dist, avg_sl_dist  # symmetric for simplicity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None = None,
    *,
    init_cash: float = 10_000.0,
    trend_ema_fast: int = 50,
    trend_ema_slow: int = 200,
    rsi_length: int = 14,
    rsi_level_long: float = 45.0,
    rsi_level_short: float = 55.0,
    atr_length: int = 14,
    sl_atr_multiplier: float = 1.5,
    tp_fib_1: float = 0.618,
    tp_fib_2: float = 1.0,
    tp_fib_3: float = 1.272,
    tp_fib_4: float = 1.618,
    enable_weekend_trading: bool = True,
    tp1_allocation: float = 0.25,
    tp2_allocation: float = 0.25,
    tp3_allocation: float = 0.25,
    tp4_allocation: float = 0.25,
    fees: float = 0.0,
    slippage: float = 0.0,
    fixed_fees: float = 0.0,
    next_bar_execution: bool = True,
    max_size: Any = None,
    position_sizes: Any = None,
    portfolio_freq: str | None = None,
    leverage: float = 1.0,
) -> QuadPortfolio:
    """Run Dual-Cloud Momentum strategy with HTF filter and 4-tier Fib TP.

    The strategy receives 15m OHLCV data and internally resamples to 1h
    for the Higher Timeframe trend filter. Entries are triggered on the
    15m chart when price pulls back to the LTF EMA with confirming RSI.

    TP1 (25%): Fib 0.618 of risk distance — close 25%, SL to break-even.
    TP2 (25%): Fib 1.0 of risk distance.
    TP3 (25%): Fib 1.272 of risk distance.
    TP4 (25%): Fib 1.618 of risk distance.

    Args:
        close: Close price series (15m).
        high: High price series (15m).
        low: Low price series (15m).
        volume: Volume series (unused, kept for API consistency).
        init_cash: Initial portfolio cash.
        trend_ema_fast: Fast EMA period for HTF trend filter.
        trend_ema_slow: Slow EMA period for HTF trend filter.
        rsi_length: RSI period on LTF.
        rsi_level_long: Minimum RSI to confirm bullish momentum.
        rsi_level_short: Maximum RSI to confirm bearish momentum.
        atr_length: ATR period for stop calculation.
        sl_atr_multiplier: ATR multiplier for stop-loss distance.
        tp_fib_1: Fibonacci level for TP1 (fraction of risk distance).
        tp_fib_2: Fibonacci level for TP2.
        tp_fib_3: Fibonacci level for TP3.
        tp_fib_4: Fibonacci level for TP4.
        enable_weekend_trading: If False, suppress entries on weekends.
        tp1_allocation: Fraction of position allocated to TP1.
        tp2_allocation: Fraction of position allocated to TP2.
        tp3_allocation: Fraction of position allocated to TP3.
        tp4_allocation: Fraction of position allocated to TP4.
        fees: Commission as fraction.
        slippage: Slippage as fraction.
        fixed_fees: Fixed fee per order.
        next_bar_execution: If True, shift signals forward 1 bar.
        max_size: Per-bar max position size (volume constraint).
        position_sizes: Custom position sizes (for Kelly).
        portfolio_freq: Portfolio frequency (e.g., "15m").
        leverage: Leverage multiplier.

    Returns:
        QuadPortfolio aggregating 4 sub-portfolios with Fibonacci TP levels.
    """
    # --- HTF Bias: resample 15m → 1h and classify trend ----------------
    market_data = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume}
        if volume is not None
        else {"open": close, "high": high, "low": low, "close": close}
    )
    # Add volume column with zeros if not provided (needed for resample agg)
    if "volume" not in market_data.columns:
        market_data["volume"] = 0.0

    htf_data = _resample_to_htf(market_data, htf_rule="1h")
    htf_bias = _compute_htf_bias(
        htf_data["close"],
        trend_ema_fast=int(trend_ema_fast),
        trend_ema_slow=int(trend_ema_slow),
    )
    # Forward-fill HTF bias to LTF index
    htf_bias_ltf = htf_bias.reindex(close.index, method="ffill").fillna("neutral")

    # --- LTF Signals ---------------------------------------------------
    long_entries, short_entries, long_exits, short_exits = _build_signals(
        close,
        high,
        low,
        htf_bias_ltf,
        ltf_ema_window=int(trend_ema_fast),
        rsi_length=int(rsi_length),
        rsi_level_long=float(rsi_level_long),
        rsi_level_short=float(rsi_level_short),
        enable_weekend_trading=bool(enable_weekend_trading),
    )

    # --- Next-bar execution --------------------------------------------
    if next_bar_execution:
        long_entries = _shift_signal(long_entries)
        short_entries = _shift_signal(short_entries)
        long_exits = _shift_signal(long_exits)
        short_exits = _shift_signal(short_exits)

    # --- Volume mask ---------------------------------------------------
    safe_max_size, valid_mask = sanitize_max_size(max_size, close.index)
    if valid_mask is not None:
        long_entries = apply_valid_mask(long_entries, valid_mask)
        short_entries = apply_valid_mask(short_entries, valid_mask)
        long_exits = apply_valid_mask(long_exits, valid_mask)
        short_exits = apply_valid_mask(short_exits, valid_mask)

    # --- Risk arrays ---------------------------------------------------
    atr, sl_dist_long, sl_dist_short = _compute_stop_distances(
        high,
        low,
        close,
        atr_length=int(atr_length),
        sl_atr_multiplier=float(sl_atr_multiplier),
    )

    # Use unified stop distance (symmetric average) as fractional stop
    sl_stop_frac = sl_dist_long / close
    sl_stop_frac = sl_stop_frac.replace(0, np.nan).fillna(
        atr * float(sl_atr_multiplier) / close
    )

    # Fibonacci TP distances as fractions of close
    fib_levels = [
        float(tp_fib_1),
        float(tp_fib_2),
        float(tp_fib_3),
        float(tp_fib_4),
    ]
    tp_stop_fracs = [sl_stop_frac * fib for fib in fib_levels]

    # --- Position sizing -----------------------------------------------
    allocs = [
        float(tp1_allocation),
        float(tp2_allocation),
        float(tp3_allocation),
        float(tp4_allocation),
    ]

    if position_sizes is not None:
        tier_sizes = [position_sizes * a for a in allocs]
    else:
        base_size = (
            (init_cash * 0.005 / sl_dist_long.replace(0, np.nan))
            .fillna(0.0)
            .clip(lower=0.0)
        )
        tier_sizes = [base_size * a * leverage for a in allocs]

    # --- Build 4 sub-portfolios ----------------------------------------
    freq = portfolio_freq or DEFAULT_TIMEFRAME
    sub_portfolios: list[Any] = []

    for i, (tp_frac, size, alloc) in enumerate(zip(tp_stop_fracs, tier_sizes, allocs)):
        pf_kwargs: dict[str, Any] = {
            "init_cash": init_cash * alloc,
            "fees": fees,
            "fixed_fees": fixed_fees,
            "slippage": slippage,
            "freq": freq,
        }
        if safe_max_size is not None:
            pf_kwargs["max_size"] = safe_max_size

        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entries.astype(bool),
            exits=long_exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=short_exits.astype(bool),
            size=size,
            sl_stop=sl_stop_frac,
            tp_stop=tp_frac,
            **pf_kwargs,
        )
        sub_portfolios.append(pf)

    return QuadPortfolio(
        sub_portfolios[0],
        sub_portfolios[1],
        sub_portfolios[2],
        sub_portfolios[3],
        tp1_alloc=allocs[0],
        tp2_alloc=allocs[1],
        tp3_alloc=allocs[2],
        tp4_alloc=allocs[3],
    )
