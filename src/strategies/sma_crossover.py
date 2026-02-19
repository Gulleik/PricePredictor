"""SMA crossover strategy: buy when fast MA crosses above slow MA."""

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
