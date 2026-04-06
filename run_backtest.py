"""Run a backtest and display stats + chart."""

import time

from src.config import (
    ACTIVE_STRATEGY,
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
    MEAN_REVERSION_BB_STD,
    MEAN_REVERSION_BB_WINDOW,
    MEAN_REVERSION_OVERBOUGHT,
    MEAN_REVERSION_OVERSOLD,
    MEAN_REVERSION_RSI_PERIOD,
    MEAN_REVERSION_USE_BOLLINGER,
    MEAN_REVERSION_VOL_LOOKBACK,
    MEAN_REVERSION_VOL_MAX_ANNUALIZED,
    ORB_ALLOW_SHORT,
    ORB_BREAKOUT_BUFFER,
    ORB_RANGE_BARS,
    TREND_ATR_STOP_MULTIPLE,
    TREND_ATR_WINDOW,
    TREND_EMA_FAST_WINDOW,
    TREND_EMA_SLOW_WINDOW,
    VOL_BREAKOUT_ATR_MIN_FRACTION,
    VOL_BREAKOUT_ATR_WINDOW,
    VOL_BREAKOUT_DONCHIAN_WINDOW,
    VOL_BREAKOUT_USE_ATR_FILTER,
)
from src.data import get_close_price_series, load_crypto_bars
from src.date_range import get_default_date_range
from src.models.broker import BrokerModel
from src.models.risk import (
    estimate_conservative_kelly,
    generate_position_sizes,
)
from src.strategies import get_strategy_module
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

    strategy_module = get_strategy_module(ACTIVE_STRATEGY)
    if ACTIVE_STRATEGY == "sma_crossover":
        print("[2/4] Running SMA crossover backtest...")
    else:
        print(f"[2/4] Running {ACTIVE_STRATEGY} backtest...")
    t1 = time.perf_counter()
    strategy_kwargs = {
        "init_cash": DEFAULT_INIT_CASH,
        "next_bar_execution": ENABLE_NEXT_BAR_EXECUTION,
        "max_size": max_size_array,
        **friction_kwargs,
    }

    if ACTIVE_STRATEGY == "sma_crossover":
        strategy_kwargs.update(
            {
                "fast": BACKTEST_FAST_WINDOW,
                "slow": BACKTEST_SLOW_WINDOW,
            }
        )
        pf, fast_ma, slow_ma = sma_run(price, **strategy_kwargs)
        indicators = {
            "Fast MA": fast_ma.ma,
            "Slow MA": slow_ma.ma,
        }
    elif ACTIVE_STRATEGY == "mean_reversion":
        strategy_kwargs.update(
            {
                "portfolio_freq": DEFAULT_TIMEFRAME,
                "rsi_period": MEAN_REVERSION_RSI_PERIOD,
                "oversold": MEAN_REVERSION_OVERSOLD,
                "overbought": MEAN_REVERSION_OVERBOUGHT,
                "bb_window": MEAN_REVERSION_BB_WINDOW,
                "bb_std": MEAN_REVERSION_BB_STD,
                "use_bollinger": MEAN_REVERSION_USE_BOLLINGER,
                "vol_lookback": MEAN_REVERSION_VOL_LOOKBACK,
                "vol_max_annualized": MEAN_REVERSION_VOL_MAX_ANNUALIZED,
            }
        )
        pf, rsi, bb_upper, bb_lower = strategy_module.run(price, **strategy_kwargs)
        indicators = {
            "RSI": rsi,
            "BB Upper": bb_upper,
            "BB Lower": bb_lower,
        }
    elif ACTIVE_STRATEGY == "trend_following":
        strategy_kwargs.update(
            {
                "portfolio_freq": DEFAULT_TIMEFRAME,
                "fast_window": TREND_EMA_FAST_WINDOW,
                "slow_window": TREND_EMA_SLOW_WINDOW,
                "atr_window": TREND_ATR_WINDOW,
                "atr_stop_multiple": TREND_ATR_STOP_MULTIPLE,
                "high": market_data["high"],
                "low": market_data["low"],
            }
        )
        pf, fast_ema, slow_ema, atr, trail = strategy_module.run(
            price, **strategy_kwargs
        )
        indicators = {
            "Fast EMA": fast_ema,
            "Slow EMA": slow_ema,
            "ATR": atr,
            "ATR Trail": trail,
        }
    elif ACTIVE_STRATEGY == "volatility_breakout":
        strategy_kwargs.update(
            {
                "portfolio_freq": DEFAULT_TIMEFRAME,
                "high": market_data["high"],
                "low": market_data["low"],
                "donchian_window": VOL_BREAKOUT_DONCHIAN_WINDOW,
                "atr_window": VOL_BREAKOUT_ATR_WINDOW,
                "use_atr_filter": VOL_BREAKOUT_USE_ATR_FILTER,
                "atr_min_fraction": VOL_BREAKOUT_ATR_MIN_FRACTION,
            }
        )
        pf, donchian_high, donchian_low, atr = strategy_module.run(
            price, **strategy_kwargs
        )
        indicators = {
            "Donchian High": donchian_high,
            "Donchian Low": donchian_low,
            "ATR": atr,
        }
    elif ACTIVE_STRATEGY == "orb":
        strategy_kwargs.update(
            {
                "portfolio_freq": DEFAULT_TIMEFRAME,
                "high": market_data["high"],
                "low": market_data["low"],
                "range_bars": ORB_RANGE_BARS,
                "breakout_buffer": ORB_BREAKOUT_BUFFER,
                "allow_short": ORB_ALLOW_SHORT,
            }
        )
        pf, range_high, range_low = strategy_module.run(price, **strategy_kwargs)
        indicators = {
            "ORB High": range_high,
            "ORB Low": range_low,
        }
    else:
        raise ValueError(f"Unsupported ACTIVE_STRATEGY: {ACTIVE_STRATEGY}")

    # Compute Kelly sizing from signals aligned to actual execution timing.
    pf_kelly = None
    kelly_raw = 0.0
    kelly_scaled = 0.0
    if ACTIVE_STRATEGY == "sma_crossover":
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

        # Plot: price, indicators, position markers
        fig = price.vbt.plot(trace_kwargs=dict(name="Close"))
        for label, indicator in indicators.items():
            indicator.vbt.plot(trace_kwargs=dict(name=label), fig=fig)
        pf_to_plot.plot_positions(close_trace_kwargs=dict(visible=False), fig=fig)
        fig.show()
    else:
        print("[4/4] Chart rendering disabled by config.")


if __name__ == "__main__":
    main()
