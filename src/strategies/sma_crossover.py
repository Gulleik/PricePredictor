"""SMA crossover strategy: buy when fast MA crosses above slow MA."""

import numpy as np
import vectorbt as vbt


def run(price, fast: int = 10, slow: int = 30, init_cash: float = 10_000.0):
    """
    Run SMA crossover backtest.

    Returns:
        Tuple of (portfolio, fast_ma, slow_ma) for plotting.
    """
    fast_ma = vbt.MA.run(price, fast)
    slow_ma = vbt.MA.run(price, slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=init_cash,
    )
    return pf, fast_ma, slow_ma


def run_scan(price, fast_windows, slow_windows, init_cash: float = 10_000.0):
    """
    Run SMA crossover backtest across all valid (fast, slow) combinations
    where fast < slow. Uses vectorized MA.run_combs for efficiency.

    Args:
        price: Price series (e.g. Close) for backtesting.
        fast_windows: Iterable of fast MA window sizes.
        slow_windows: Iterable of slow MA window sizes.
        init_cash: Initial cash for each portfolio.

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
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=init_cash,
    )
    return pf
