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
    ENABLE_NEXT_BAR_EXECUTION,
    KELLY_FACTOR,
    MAX_VOLUME_PARTICIPATION,
)
from src.data import get_close_price_series, load_crypto_bars
from src.date_range import get_default_date_range
from src.models.broker import BrokerModel
from src.models.risk import (
    estimate_conservative_kelly,
    generate_position_sizes,
)
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
    price = get_close_price_series(market_data)
    print(f"[1/4] Done in {time.perf_counter() - t0:.2f}s ({len(price)} bars)")

    # Initialize broker model for friction and volume constraints
    broker = BrokerModel(
        commission_pct=BROKER_COMMISSION_PCT,
        fixed_fee=BROKER_FIXED_FEE,
        slippage_pct=BROKER_SLIPPAGE_PCT,
        max_volume_participation=MAX_VOLUME_PARTICIPATION,
    )
    max_size_array = broker.compute_max_size_array(
        market_data,
        enable=ENABLE_FRICTION_MODEL,
    )
    friction_kwargs = broker.build_friction_kwargs(
        enable_friction=ENABLE_FRICTION_MODEL
    )

    print("[2/4] Running SMA crossover backtest...")
    t1 = time.perf_counter()
    pf, fast_ma, slow_ma = sma_run(
        price,
        fast=BACKTEST_FAST_WINDOW,
        slow=BACKTEST_SLOW_WINDOW,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        **friction_kwargs,
        max_size=max_size_array,
    )

    # Compute Kelly sizing from signals aligned to actual execution timing.
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    if ENABLE_NEXT_BAR_EXECUTION:
        entries = entries.astype(bool).shift(1, fill_value=False)
        exits = exits.astype(bool).shift(1, fill_value=False)

    kelly_raw = estimate_conservative_kelly(entries, exits, price)
    kelly_scaled = kelly_raw * KELLY_FACTOR

    if kelly_scaled > 0:
        position_sizes = generate_position_sizes(
            entries,
            price,
            kelly_scaled,
            DEFAULT_INIT_CASH,
        )
        pf_kelly, _, _ = sma_run(
            price,
            fast=BACKTEST_FAST_WINDOW,
            slow=BACKTEST_SLOW_WINDOW,
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            **friction_kwargs,
            max_size=max_size_array,
            position_sizes=position_sizes,
        )
    else:
        pf_kelly = None

    print(
        "[2/4] Done in "
        f"{time.perf_counter() - t1:.2f}s "
        f"(Kelly raw={kelly_raw:.4f}, scaled={kelly_scaled:.4f})"
    )

    print("[3/4] Printing portfolio stats...")
    print("\n=== Baseline (Fixed Position Size) ===")
    print(pf.stats())

    if pf_kelly is not None:
        print("\n=== Kelly Criterion Sizing ===")
        print(pf_kelly.stats())
        print("\nKelly Sizing Summary:")
        print(f"  Raw Kelly Fraction: {kelly_raw:.4f}")
        print(f"  Scaled (factor={KELLY_FACTOR}): {kelly_scaled:.4f}")
    print()

    if BACKTEST_RENDER_CHART:
        print("[4/4] Rendering chart...")
        # Use Kelly-sized portfolio if available, otherwise baseline
        pf_to_plot = pf_kelly if pf_kelly is not None else pf

        # Plot: price, MAs, position markers
        fig = price.vbt.plot(trace_kwargs=dict(name="Close"))
        fast_ma.ma.vbt.plot(trace_kwargs=dict(name="Fast MA"), fig=fig)
        slow_ma.ma.vbt.plot(trace_kwargs=dict(name="Slow MA"), fig=fig)
        pf_to_plot.plot_positions(close_trace_kwargs=dict(visible=False), fig=fig)
        fig.show()
    else:
        print("[4/4] Chart rendering disabled by config.")


if __name__ == "__main__":
    main()
