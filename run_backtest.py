"""Run a backtest and display stats + chart."""

import time

from src.config import (
    ACTIVE_STRATEGY,
    ADAPTIVE_MOMENTUM_ADX_PERIOD,
    ADAPTIVE_MOMENTUM_ADX_THRESHOLD,
    ADAPTIVE_MOMENTUM_ATR_PERCENTILE_MIN,
    ADAPTIVE_MOMENTUM_ATR_PERCENTILE_WINDOW,
    ADAPTIVE_MOMENTUM_ATR_WINDOW,
    ADAPTIVE_MOMENTUM_EMA_FAST,
    ADAPTIVE_MOMENTUM_EMA_MEDIUM,
    ADAPTIVE_MOMENTUM_EMA_SLOW,
    ADAPTIVE_MOMENTUM_ENTRY_COOLDOWN_BARS,
    ADAPTIVE_MOMENTUM_MACD_FAST,
    ADAPTIVE_MOMENTUM_MACD_SIGNAL,
    ADAPTIVE_MOMENTUM_MACD_SLOW,
    ADAPTIVE_MOMENTUM_RSI_OVERBOUGHT,
    ADAPTIVE_MOMENTUM_RSI_OVERSOLD,
    ADAPTIVE_MOMENTUM_RSI_PERIOD,
    ADAPTIVE_MOMENTUM_SIGNAL_THRESHOLD,
    ADAPTIVE_MOMENTUM_SL_ATR_MULTIPLE,
    ADAPTIVE_MOMENTUM_TP1_MULTIPLE,
    ADAPTIVE_MOMENTUM_TP2_TRAIL_MULTIPLE,
    ADAPTIVE_MOMENTUM_VOL_THRESHOLD,
    ADAPTIVE_MOMENTUM_VOLUME_WINDOW,
    BACKTEST_FAST_WINDOW,
    BACKTEST_RENDER_CHART,
    BACKTEST_SLOW_WINDOW,
    BACKTEST_SYMBOL,
    BB_RSI_OVERBOUGHT,
    BB_RSI_OVERSOLD,
    BB_RSI_RSI_WINDOW,
    BB_RSI_STD,
    BB_RSI_WINDOW,
    BROKER_COMMISSION_PCT,
    BROKER_FIXED_FEE,
    BROKER_SLIPPAGE_PCT,
    DCM_ATR_LENGTH,
    DCM_ENABLE_WEEKEND_TRADING,
    DCM_RSI_LENGTH,
    DCM_RSI_LEVEL_LONG,
    DCM_RSI_LEVEL_SHORT,
    DCM_SL_ATR_MULTIPLIER,
    DCM_TIMEFRAME,
    DCM_TP_FIB_1,
    DCM_TP_FIB_2,
    DCM_TP_FIB_3,
    DCM_TP_FIB_4,
    DCM_TREND_EMA_FAST,
    DCM_TREND_EMA_SLOW,
    DEFAULT_INIT_CASH,
    DEFAULT_TIMEFRAME,
    EMA_RIBBON_ENTRY_COOLDOWN_BARS,
    EMA_RIBBON_FAST_WINDOW,
    EMA_RIBBON_MEDIUM_WINDOW,
    EMA_RIBBON_PULLBACK_RECLAIM,
    EMA_RIBBON_SLOW_WINDOW,
    ENABLE_FRICTION_MODEL,
    ENABLE_NEXT_BAR_EXECUTION,
    KELLY_FACTOR,
    LEVERAGE,
    MAX_VOLUME_PARTICIPATION,
    MEAN_REVERSION_BB_STD,
    MEAN_REVERSION_BB_WINDOW,
    MEAN_REVERSION_OVERBOUGHT,
    MEAN_REVERSION_OVERSOLD,
    MEAN_REVERSION_RSI_PERIOD,
    MEAN_REVERSION_USE_BOLLINGER,
    MEAN_REVERSION_VOL_LOOKBACK,
    MEAN_REVERSION_VOL_MAX_ANNUALIZED,
    MOMENTUM_SCALP_ATR_WINDOW,
    MOMENTUM_SCALP_EMA_FAST,
    MOMENTUM_SCALP_EMA_MEDIUM,
    MOMENTUM_SCALP_EMA_SLOW,
    MOMENTUM_SCALP_ENTRY_COOLDOWN_BARS,
    MOMENTUM_SCALP_MACD_FAST,
    MOMENTUM_SCALP_MACD_SIGNAL,
    MOMENTUM_SCALP_MACD_SLOW,
    MOMENTUM_SCALP_RSI_OVERBOUGHT,
    MOMENTUM_SCALP_RSI_OVERSOLD,
    MOMENTUM_SCALP_RSI_PERIOD,
    MOMENTUM_SCALP_SL_ATR_MULTIPLE,
    MOMENTUM_SCALP_TP1_MULTIPLE,
    MOMENTUM_SCALP_TP2_MULTIPLE,
    MOMENTUM_SCALP_TP3_TRAIL_MULTIPLE,
    MOMENTUM_SCALP_VOL_THRESHOLD,
    MOMENTUM_SCALP_VOLUME_WINDOW,
    ORB_ALLOW_SHORT,
    ORB_BREAKOUT_BUFFER,
    ORB_RANGE_BARS,
    SCALP_ATR_MULTIPLE,
    SCALP_ATR_WINDOW,
    SCALP_FEES,
    SCALP_RISK_FRACTION,
    SCALP_SLIPPAGE,
    SCALP_TP_R_MULTIPLE,
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
    estimate_kelly_from_portfolio,
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
        leverage=LEVERAGE,
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
        "leverage": LEVERAGE,
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
    elif ACTIVE_STRATEGY == "ema_ribbon_scalp":
        pf = strategy_module.ema_ribbon_scalp(
            close=price,
            high=market_data["high"],
            low=market_data["low"],
            init_cash=DEFAULT_INIT_CASH,
            ema_fast=EMA_RIBBON_FAST_WINDOW,
            ema_medium=EMA_RIBBON_MEDIUM_WINDOW,
            ema_slow=EMA_RIBBON_SLOW_WINDOW,
            pullback_reclaim=EMA_RIBBON_PULLBACK_RECLAIM,
            entry_cooldown_bars=EMA_RIBBON_ENTRY_COOLDOWN_BARS,
            atr_window=SCALP_ATR_WINDOW,
            atr_multiple=SCALP_ATR_MULTIPLE,
            risk_fraction=SCALP_RISK_FRACTION,
            tp_r_multiple=SCALP_TP_R_MULTIPLE,
            fees=SCALP_FEES if ENABLE_FRICTION_MODEL else 0.0,
            slippage=SCALP_SLIPPAGE if ENABLE_FRICTION_MODEL else 0.0,
            fixed_fees=BROKER_FIXED_FEE if ENABLE_FRICTION_MODEL else 0.0,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            portfolio_freq=DEFAULT_TIMEFRAME,
            leverage=LEVERAGE,
        )
        indicators = {}
    elif ACTIVE_STRATEGY == "momentum_scalp":
        pf = strategy_module.run(
            close=price,
            high=market_data["high"],
            low=market_data["low"],
            volume=market_data["volume"],
            init_cash=DEFAULT_INIT_CASH,
            ema_fast=MOMENTUM_SCALP_EMA_FAST,
            ema_medium=MOMENTUM_SCALP_EMA_MEDIUM,
            ema_slow=MOMENTUM_SCALP_EMA_SLOW,
            rsi_period=MOMENTUM_SCALP_RSI_PERIOD,
            rsi_overbought=MOMENTUM_SCALP_RSI_OVERBOUGHT,
            rsi_oversold=MOMENTUM_SCALP_RSI_OVERSOLD,
            macd_fast=MOMENTUM_SCALP_MACD_FAST,
            macd_slow=MOMENTUM_SCALP_MACD_SLOW,
            macd_signal=MOMENTUM_SCALP_MACD_SIGNAL,
            vol_window=MOMENTUM_SCALP_VOLUME_WINDOW,
            vol_threshold=MOMENTUM_SCALP_VOL_THRESHOLD,
            atr_window=MOMENTUM_SCALP_ATR_WINDOW,
            sl_atr_multiple=MOMENTUM_SCALP_SL_ATR_MULTIPLE,
            tp1_multiple=MOMENTUM_SCALP_TP1_MULTIPLE,
            tp2_multiple=MOMENTUM_SCALP_TP2_MULTIPLE,
            tp3_trail_multiple=MOMENTUM_SCALP_TP3_TRAIL_MULTIPLE,
            entry_cooldown_bars=MOMENTUM_SCALP_ENTRY_COOLDOWN_BARS,
            fees=SCALP_FEES if ENABLE_FRICTION_MODEL else 0.0,
            slippage=SCALP_SLIPPAGE if ENABLE_FRICTION_MODEL else 0.0,
            fixed_fees=BROKER_FIXED_FEE if ENABLE_FRICTION_MODEL else 0.0,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            max_size=max_size_array,
            portfolio_freq=DEFAULT_TIMEFRAME,
            leverage=LEVERAGE,
        )
        indicators = {}
    elif ACTIVE_STRATEGY == "adaptive_momentum":
        pf = strategy_module.run(
            close=price,
            high=market_data["high"],
            low=market_data["low"],
            volume=market_data["volume"],
            init_cash=DEFAULT_INIT_CASH,
            ema_fast=ADAPTIVE_MOMENTUM_EMA_FAST,
            ema_medium=ADAPTIVE_MOMENTUM_EMA_MEDIUM,
            ema_slow=ADAPTIVE_MOMENTUM_EMA_SLOW,
            rsi_period=ADAPTIVE_MOMENTUM_RSI_PERIOD,
            rsi_overbought=ADAPTIVE_MOMENTUM_RSI_OVERBOUGHT,
            rsi_oversold=ADAPTIVE_MOMENTUM_RSI_OVERSOLD,
            macd_fast=ADAPTIVE_MOMENTUM_MACD_FAST,
            macd_slow=ADAPTIVE_MOMENTUM_MACD_SLOW,
            macd_signal=ADAPTIVE_MOMENTUM_MACD_SIGNAL,
            vol_window=ADAPTIVE_MOMENTUM_VOLUME_WINDOW,
            vol_threshold=ADAPTIVE_MOMENTUM_VOL_THRESHOLD,
            signal_threshold=ADAPTIVE_MOMENTUM_SIGNAL_THRESHOLD,
            adx_period=ADAPTIVE_MOMENTUM_ADX_PERIOD,
            adx_threshold=ADAPTIVE_MOMENTUM_ADX_THRESHOLD,
            atr_percentile_min=ADAPTIVE_MOMENTUM_ATR_PERCENTILE_MIN,
            atr_percentile_window=ADAPTIVE_MOMENTUM_ATR_PERCENTILE_WINDOW,
            atr_window=ADAPTIVE_MOMENTUM_ATR_WINDOW,
            sl_atr_multiple=ADAPTIVE_MOMENTUM_SL_ATR_MULTIPLE,
            tp1_multiple=ADAPTIVE_MOMENTUM_TP1_MULTIPLE,
            tp2_trail_multiple=ADAPTIVE_MOMENTUM_TP2_TRAIL_MULTIPLE,
            entry_cooldown_bars=ADAPTIVE_MOMENTUM_ENTRY_COOLDOWN_BARS,
            fees=SCALP_FEES if ENABLE_FRICTION_MODEL else 0.0,
            slippage=SCALP_SLIPPAGE if ENABLE_FRICTION_MODEL else 0.0,
            fixed_fees=BROKER_FIXED_FEE if ENABLE_FRICTION_MODEL else 0.0,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            max_size=max_size_array,
            portfolio_freq=DEFAULT_TIMEFRAME,
            leverage=LEVERAGE,
        )
        indicators = {}
    elif ACTIVE_STRATEGY == "bb_rsi_mean_reversion":
        pf = strategy_module.bb_rsi_mean_reversion(
            close=price,
            high=market_data["high"],
            low=market_data["low"],
            init_cash=DEFAULT_INIT_CASH,
            bb_window=BB_RSI_WINDOW,
            bb_std=BB_RSI_STD,
            rsi_window=BB_RSI_RSI_WINDOW,
            rsi_oversold=BB_RSI_OVERSOLD,
            rsi_overbought=BB_RSI_OVERBOUGHT,
            atr_window=SCALP_ATR_WINDOW,
            atr_multiple=SCALP_ATR_MULTIPLE,
            risk_fraction=SCALP_RISK_FRACTION,
            tp_r_multiple=SCALP_TP_R_MULTIPLE,
            fees=SCALP_FEES if ENABLE_FRICTION_MODEL else 0.0,
            slippage=SCALP_SLIPPAGE if ENABLE_FRICTION_MODEL else 0.0,
            fixed_fees=BROKER_FIXED_FEE if ENABLE_FRICTION_MODEL else 0.0,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            portfolio_freq=DEFAULT_TIMEFRAME,
            leverage=LEVERAGE,
        )
        indicators = {}
    elif ACTIVE_STRATEGY == "dual_cloud_momentum":
        # DCM uses 15m data; reload at strategy-specific timeframe
        dcm_market_data = load_crypto_bars(
            symbol,
            start=start,
            end=end,
            timeframe=DCM_TIMEFRAME,
        )
        dcm_price = get_close_price_series(dcm_market_data)
        # Recompute broker constraints for 15m data
        dcm_max_size = broker.compute_max_size_array(
            dcm_market_data,
            enable=ENABLE_FRICTION_MODEL,
        )
        pf = strategy_module.run(
            close=dcm_price,
            high=dcm_market_data["high"],
            low=dcm_market_data["low"],
            volume=dcm_market_data["volume"],
            init_cash=DEFAULT_INIT_CASH,
            trend_ema_fast=DCM_TREND_EMA_FAST,
            trend_ema_slow=DCM_TREND_EMA_SLOW,
            rsi_length=DCM_RSI_LENGTH,
            rsi_level_long=DCM_RSI_LEVEL_LONG,
            rsi_level_short=DCM_RSI_LEVEL_SHORT,
            atr_length=DCM_ATR_LENGTH,
            sl_atr_multiplier=DCM_SL_ATR_MULTIPLIER,
            tp_fib_1=DCM_TP_FIB_1,
            tp_fib_2=DCM_TP_FIB_2,
            tp_fib_3=DCM_TP_FIB_3,
            tp_fib_4=DCM_TP_FIB_4,
            enable_weekend_trading=DCM_ENABLE_WEEKEND_TRADING,
            fees=SCALP_FEES if ENABLE_FRICTION_MODEL else 0.0,
            slippage=SCALP_SLIPPAGE if ENABLE_FRICTION_MODEL else 0.0,
            fixed_fees=BROKER_FIXED_FEE if ENABLE_FRICTION_MODEL else 0.0,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            max_size=dcm_max_size,
            portfolio_freq=DCM_TIMEFRAME,
            leverage=LEVERAGE,
        )
        # Override price for plotting/stats to use 15m data
        price = dcm_price
        indicators = {}
    else:
        raise ValueError(f"Unsupported ACTIVE_STRATEGY: {ACTIVE_STRATEGY}")

    # Compute Kelly sizing from actual portfolio trades.
    # This is more accurate than signal-pair estimation.
    # The baseline run shows real trade dynamics including friction and execution.
    pf_kelly = None
    kelly_raw = 0.0
    kelly_scaled = 0.0
    if ACTIVE_STRATEGY == "sma_crossover":
        # Estimate Kelly from actual portfolio trades (not signal pairs)
        kelly_raw = estimate_kelly_from_portfolio(pf)
        kelly_scaled = kelly_raw * KELLY_FACTOR

        if kelly_scaled > 0:
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            if ENABLE_NEXT_BAR_EXECUTION:
                entries = entries.astype(bool).shift(1, fill_value=False)
                exits = exits.astype(bool).shift(1, fill_value=False)

            position_sizes = generate_position_sizes(
                entries,
                price,
                kelly_scaled,
                DEFAULT_INIT_CASH,
                leverage=LEVERAGE,
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

    # Handle CombinedPortfolio / DualPortfolio / QuadPortfolio (no .stats() method)
    if ACTIVE_STRATEGY in {
        "momentum_scalp",
        "adaptive_momentum",
        "dual_cloud_momentum",
    }:
        combined_pf = pf
        print(f"\nCombined Portfolio Return: {combined_pf.total_return():.4f}")
        print(f"Combined Portfolio Value (final): {combined_pf.value().iloc[-1]:.2f}")
        print("Note: Detailed stats aggregation skipped for CombinedPortfolio.")
    else:
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
