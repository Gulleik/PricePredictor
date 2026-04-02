"""SMA crossover strategy: buy when fast MA crosses above slow MA."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt


def run(
    price: pd.Series,
    fast: int = 10,
    slow: int = 30,
    init_cash: float = 10_000.0,
    *,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: Any | None = None,
    position_sizes: Any | None = None,
) -> tuple[Any, Any, Any]:
    """
    Run SMA crossover backtest with optional execution fidelity modeling.

    Args:
        price: Close price series.
        fast: Fast MA window in bars.
        slow: Slow MA window in bars.
        init_cash: Initial cash for portfolio.
        next_bar_execution: If True, shift signals forward 1 bar (fill at T+1).
        fees: Percentage fee per order (e.g., 0.001 for 0.1%).
        fixed_fees: Fixed fee in currency per order.
        slippage: Percentage slippage (e.g., 0.002 for 0.2%).
        max_size: Array of max position sizes per bar (for volume constraints).
        position_sizes: Array of position sizes per signal (for Kelly sizing).

    Returns:
        Tuple of (portfolio, fast_ma, slow_ma) for plotting.
    """
    fast_ma = vbt.MA.run(price, fast)
    slow_ma = vbt.MA.run(price, slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    # Shift execution to next bar to avoid same-bar signal fills when requested.
    if next_bar_execution:
        entries = entries.vbt.fshift(1).fillna(False).astype(bool)
        exits = exits.vbt.fshift(1).fillna(False).astype(bool)

    portfolio_kwargs: dict[str, Any] = {
        "init_cash": init_cash,
        "fees": fees,
        "fixed_fees": fixed_fees,
        "slippage": slippage,
    }
    if max_size is not None:
        portfolio_kwargs["max_size"] = max_size
    if position_sizes is not None:
        portfolio_kwargs["size"] = position_sizes

    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        **portfolio_kwargs,
    )
    return pf, fast_ma, slow_ma


def run_scan(
    price: pd.Series,
    fast_windows: Iterable[int],
    slow_windows: Iterable[int],
    init_cash: float = 10_000.0,
    next_bar_execution: bool = False,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    max_size: np.ndarray | None = None,
) -> Any:
    """
    Run SMA crossover backtest across all valid (fast, slow) combinations
    where fast < slow. Uses vectorized MA.run_combs for efficiency.

    Args:
        price: Close price series.
        fast_windows: Iterable of fast MA window sizes.
        slow_windows: Iterable of slow MA window sizes.
        init_cash: Initial cash for each portfolio.
        next_bar_execution: If True, shift signals forward 1 bar.
        fees: Percentage fee per order.
        fixed_fees: Fixed fee per order.
        slippage: Percentage slippage.
        max_size: Array of max position sizes per bar.

    Returns:
        Portfolio with one column per (fast, slow) parameter combination.
    """
    windows = np.unique(
        np.concatenate([np.asarray(fast_windows), np.asarray(slow_windows)])
    )
    fast_ma, slow_ma = vbt.MA.run_combs(
        price, window=windows, r=2, short_names=["fast", "slow"]
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # Apply next-bar execution if requested
    if next_bar_execution:
        entries = entries.vbt.fshift(1)
        exits = exits.vbt.fshift(1)

    broadcast_max_size = max_size
    if max_size is not None:
        max_size_arr = np.asarray(max_size)
        # For parameter grids, VectorBT expects a shape that can broadcast
        # against (n_rows, n_cols). Convert (n_rows,) -> (n_rows, 1).
        if max_size_arr.ndim == 1:
            broadcast_max_size = max_size_arr.reshape(-1, 1)
    
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        max_size=broadcast_max_size,
    )
    return pf
