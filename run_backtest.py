"""Run a backtest and display stats + chart."""

from src.config import (
    DEFAULT_END,
    DEFAULT_INIT_CASH,
    DEFAULT_START,
    DEFAULT_TIMEFRAME,
)
from src.data import load_crypto_bars
from src.strategies.sma_crossover import run as sma_run


def main() -> None:
    symbol = "BTC/USD"
    price = load_crypto_bars(
        symbol,
        start=DEFAULT_START,
        end=DEFAULT_END,
        timeframe=DEFAULT_TIMEFRAME,
    )

    pf, fast_ma, slow_ma = sma_run(
        price,
        fast=10,
        slow=30,
        init_cash=DEFAULT_INIT_CASH,
    )

    print(pf.stats())
    print()

    # Plot: price, MAs, position markers
    fig = price.vbt.plot(trace_kwargs=dict(name="Close"))
    fast_ma.ma.vbt.plot(trace_kwargs=dict(name="Fast MA"), fig=fig)
    slow_ma.ma.vbt.plot(trace_kwargs=dict(name="Slow MA"), fig=fig)
    pf.plot_positions(close_trace_kwargs=dict(visible=False), fig=fig)
    fig.show()


if __name__ == "__main__":
    main()
