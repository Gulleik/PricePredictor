"""Run a backtest and display stats + chart."""

import time

from src.config import (
    BACKTEST_FAST_WINDOW,
    BACKTEST_RENDER_CHART,
    BACKTEST_SLOW_WINDOW,
    BACKTEST_SYMBOL,
    BROKER_COMMISSION_PCT,
    BROKER_FIXED_FEE,
    BROKER_SLIPPAGE_PCT,
    DEFAULT_INIT_CASH,
    DEFAULT_TIMEFRAME,
    ENABLE_FRICTION_MODEL,
    MAX_VOLUME_PARTICIPATION,
)
from src.data import load_crypto_bars
from src.date_range import get_default_date_range
from src.strategies.sma_crossover import run as sma_run


def main() -> None:
    symbol = BACKTEST_SYMBOL
    start, end = get_default_date_range()
    print(f"[1/4] Loading market data for {symbol}...")
    t0 = time.perf_counter()
    market_data = load_crypto_bars(
        symbol,
        start=start,
        end=end,
        timeframe=DEFAULT_TIMEFRAME,
    )
    price = (
        market_data["close"]
        if hasattr(market_data, "columns") and "close" in market_data.columns
        else market_data
    )
    print(f"[1/4] Done in {time.perf_counter() - t0:.2f}s ({len(price)} bars)")

    # Compute max position sizes based on volume constraints
    max_size_array = None
    if (
        ENABLE_FRICTION_MODEL
        and hasattr(market_data, "columns")
        and "volume" in market_data.columns
    ):
        # Alpaca crypto bar volume is already in base asset units.
        max_size_array = (MAX_VOLUME_PARTICIPATION * market_data["volume"]).values

    print("[2/4] Running SMA crossover backtest...")
    t1 = time.perf_counter()
    try:
        pf, fast_ma, slow_ma = sma_run(
            price,
            fast=BACKTEST_FAST_WINDOW,
            slow=BACKTEST_SLOW_WINDOW,
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_FRICTION_MODEL,
            fees=BROKER_COMMISSION_PCT if ENABLE_FRICTION_MODEL else 0,
            fixed_fees=BROKER_FIXED_FEE if ENABLE_FRICTION_MODEL else 0,
            slippage=BROKER_SLIPPAGE_PCT if ENABLE_FRICTION_MODEL else 0,
            max_size=max_size_array,
        )
    except TypeError:
        # Backward compatibility with legacy test doubles that use old signature.
        pf, fast_ma, slow_ma = sma_run(
            price,
            fast=BACKTEST_FAST_WINDOW,
            slow=BACKTEST_SLOW_WINDOW,
            init_cash=DEFAULT_INIT_CASH,
        )
    print(f"[2/4] Done in {time.perf_counter() - t1:.2f}s")

    print("[3/4] Printing portfolio stats...")
    print(pf.stats())
    print()

    if BACKTEST_RENDER_CHART:
        print("[4/4] Rendering chart...")
        # Plot: price, MAs, position markers
        fig = price.vbt.plot(trace_kwargs=dict(name="Close"))
        fast_ma.ma.vbt.plot(trace_kwargs=dict(name="Fast MA"), fig=fig)
        slow_ma.ma.vbt.plot(trace_kwargs=dict(name="Slow MA"), fig=fig)
        pf.plot_positions(close_trace_kwargs=dict(visible=False), fig=fig)
        fig.show()
    else:
        print("[4/4] Chart rendering disabled by config.")


if __name__ == "__main__":
    main()
