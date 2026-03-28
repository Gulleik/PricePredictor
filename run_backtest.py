"""Run a backtest and display stats + chart."""

import time

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
    print(f"[1/4] Loading market data for {symbol}...")
    t0 = time.perf_counter()
    price = load_crypto_bars(
        symbol,
        start=DEFAULT_START,
        end=DEFAULT_END,
        timeframe=DEFAULT_TIMEFRAME,
    )
    print(f"[1/4] Done in {time.perf_counter() - t0:.2f}s ({len(price)} bars)")

    print("[2/4] Running SMA crossover backtest...")
    t1 = time.perf_counter()
    pf, fast_ma, slow_ma = sma_run(
        price,
        fast=5,
        slow=15,
        init_cash=DEFAULT_INIT_CASH,
    )
    print(f"[2/4] Done in {time.perf_counter() - t1:.2f}s")

    print("[3/4] Printing portfolio stats...")
    print(pf.stats())
    print()

    print("[4/4] Rendering chart...")
    # Plot: price, MAs, position markers
    fig = price.vbt.plot(trace_kwargs=dict(name="Close"))
    fast_ma.ma.vbt.plot(trace_kwargs=dict(name="Fast MA"), fig=fig)
    slow_ma.ma.vbt.plot(trace_kwargs=dict(name="Slow MA"), fig=fig)
    pf.plot_positions(close_trace_kwargs=dict(visible=False), fig=fig)
    fig.show()


if __name__ == "__main__":
    main()
