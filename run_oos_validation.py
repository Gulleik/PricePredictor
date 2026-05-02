"""Out-of-sample and walk-forward validation for fixed parameter sets.

Validates one or more strategy configs by:
1. Running on the full dataset
2. Splitting into 75% in-sample / 25% out-of-sample
3. Running walk-forward windows on OOS slices
4. Reporting per-segment metrics to detect overfitting
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.analysis.annualization import periods_per_year_from_freq
from src.analysis.wfo import generate_wfo_windows
from src.config import (
    BROKER_COMMISSION_PCT,
    BROKER_FIXED_FEE,
    BROKER_SLIPPAGE_PCT,
    DEFAULT_INIT_CASH,
    ENABLE_FRICTION_MODEL,
    ENABLE_NEXT_BAR_EXECUTION,
    LEVERAGE,
    MAX_VOLUME_PARTICIPATION,
)
from src.data import get_close_price_series, load_crypto_bars
from src.date_range import get_default_date_range
from src.models.broker import BrokerModel
from src.models.metrics import compute_advanced_metrics
from src.strategies import get_strategy_module

VALIDATION_CONFIGS: list[dict[str, Any]] = [
    {
        "strategy_name": "sma_crossover",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "best_params": {"fast": 15, "slow": 50},
    },
    {
        "strategy_name": "trend_following",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "best_params": {
            "fast_window": 8,
            "slow_window": 35,
            "atr_window": 10,
            "atr_stop_multiple": 1.5,
        },
    },
    {
        "strategy_name": "mean_reversion",
        "symbol": "DOGE/USD",
        "timeframe": "1h",
        "best_params": {
            "bb_std": 1.5,
            "bb_window": 26,
            "overbought": 65.0,
            "oversold": 20.0,
            "rsi_period": 18,
            "vol_max_annualized": 1.5,
        },
    },
    {
        "strategy_name": "mean_reversion",
        "symbol": "SOL/USD",
        "timeframe": "1h",
        "best_params": {
            "bb_std": 1.5,
            "bb_window": 20,
            "overbought": 65.0,
            "oversold": 20.0,
            "rsi_period": 8,
            "vol_max_annualized": 1.0,
        },
    },
    {
        "strategy_name": "volatility_breakout",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "best_params": {
            "atr_min_fraction": 0.008,
            "atr_window": 10,
            "donchian_window": 30,
        },
    },
]


def _resolve_wfo_windows(timeframe: str) -> tuple[int, int, int]:
    """Set WFO windows using ~6 month IS and ~2 month OOS equivalents."""
    periods = periods_per_year_from_freq(timeframe)
    is_bars = max(240, int(round(periods * 0.5)))
    oos_bars = max(80, int(round(periods / 6)))
    step_bars = oos_bars
    return is_bars, oos_bars, step_bars


def _needs_hl(strategy_name: str) -> bool:
    return strategy_name in {
        "trend_following",
        "volatility_breakout",
        "orb",
        "ema_ribbon_scalp",
        "bb_rsi_mean_reversion",
        "momentum_scalp",
        "adaptive_momentum",
    }


def _needs_volume(strategy_name: str) -> bool:
    return strategy_name in {"momentum_scalp", "adaptive_momentum"}


def _normalize_portfolio(run_result: Any) -> Any:
    """Get the portfolio object from strategy run() return shapes."""
    if isinstance(run_result, tuple):
        return run_result[0]
    return run_result


def _run_on_slice(
    *,
    strategy_name: str,
    timeframe: str,
    best_params: dict[str, Any],
    market_data: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    broker: BrokerModel,
) -> dict[str, Any]:
    """Run a strategy on one data slice and return metrics."""
    sliced = market_data.iloc[start_idx:end_idx].copy()
    close = get_close_price_series(sliced)

    if len(close) < 60:
        return {
            "n_bars": len(close),
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "total_return": 0.0,
            "n_trades": 0,
        }

    max_size_array = broker.compute_max_size_array(
        sliced,
        enable=ENABLE_FRICTION_MODEL,
    )
    friction_kwargs = broker.build_friction_kwargs(
        enable_friction=ENABLE_FRICTION_MODEL,
    )

    strategy_module = get_strategy_module(strategy_name)
    run_kwargs: dict[str, Any] = {
        "init_cash": DEFAULT_INIT_CASH,
        "next_bar_execution": ENABLE_NEXT_BAR_EXECUTION,
        "max_size": max_size_array,
        "position_sizes": None,
        "portfolio_freq": timeframe,
        "leverage": LEVERAGE,
        **friction_kwargs,
        **best_params,
    }

    if _needs_hl(strategy_name):
        run_kwargs["high"] = sliced["high"]
        run_kwargs["low"] = sliced["low"]
    if _needs_volume(strategy_name) and "volume" in sliced.columns:
        run_kwargs["volume"] = sliced["volume"]

    run_result = strategy_module.run(close, **run_kwargs)
    pf = _normalize_portfolio(run_result)

    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    periods_per_year = periods_per_year_from_freq(timeframe)
    metrics = compute_advanced_metrics(returns, periods_per_year=periods_per_year)
    total_return = float((1.0 + returns).prod() - 1.0)

    n_trades = 0
    if hasattr(pf, "trades") and hasattr(pf.trades, "records"):
        rec = pf.trades.records
        if isinstance(rec, pd.DataFrame):
            n_trades = len(rec)

    return {
        "n_bars": len(close),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "sortino_ratio": float(metrics["sortino_ratio"]),
        "calmar_ratio": float(metrics["calmar_ratio"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "max_drawdown_duration": int(metrics["max_drawdown_duration"]),
        "total_return": float(total_return),
        "n_trades": int(n_trades),
    }


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """Print one formatted metric block."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Bars:         {metrics['n_bars']}")
    print(f"  Trades:       {metrics['n_trades']}")
    ret_pct = metrics["total_return"] * 100
    print(f"  Total Return: {metrics['total_return']:+.4f}  ({ret_pct:+.2f}%)")
    print(f"  Sharpe:       {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino:      {metrics['sortino_ratio']:.4f}")
    print(f"  Calmar:       {metrics['calmar_ratio']:.4f}")
    dd_pct = metrics["max_drawdown"] * 100
    print(f"  Max DD:       {metrics['max_drawdown']:.4f}  ({dd_pct:.2f}%)")
    print(f"  Max DD Dur:   {metrics['max_drawdown_duration']} bars")


def _print_params(best_params: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print("  PARAMETER SET UNDER TEST")
    print(f"{'=' * 60}")
    for key, value in sorted(best_params.items()):
        print(f"  {key}: {value}")


def _verdict_from_wfo(oos_sharpes: list[float]) -> str:
    if not oos_sharpes:
        return "INSUFFICIENT DATA"

    pct_positive = sum(1 for value in oos_sharpes if value > 0) / len(oos_sharpes)
    avg_sharpe = sum(oos_sharpes) / len(oos_sharpes)

    if avg_sharpe > 0.5 and pct_positive >= 0.6:
        return "PASS"
    if avg_sharpe > 0 and pct_positive >= 0.5:
        return "MARGINAL"
    return "FAIL"


def _run_validation(config: dict[str, Any], broker: BrokerModel) -> dict[str, Any]:
    strategy_name = str(config["strategy_name"])
    symbol = str(config["symbol"])
    timeframe = str(config["timeframe"])
    best_params = dict(config["best_params"])

    print(f"\n{'#' * 72}")
    print(f"### Validation: {strategy_name} | {symbol} | {timeframe}")
    print(f"{'#' * 72}")

    start, end = get_default_date_range()
    print(f"[1/4] Loading {symbol} {timeframe} data...")
    t0 = time.perf_counter()
    market_data = load_crypto_bars(symbol, start=start, end=end, timeframe=timeframe)
    n_bars = len(market_data)
    print(f"[1/4] Done in {time.perf_counter() - t0:.2f}s ({n_bars} bars)")

    print("\n[2/4] Running full-period baseline...")
    t1 = time.perf_counter()
    full_metrics = _run_on_slice(
        strategy_name=strategy_name,
        timeframe=timeframe,
        best_params=best_params,
        market_data=market_data,
        start_idx=0,
        end_idx=n_bars,
        broker=broker,
    )
    print(f"[2/4] Done in {time.perf_counter() - t1:.2f}s")
    _print_metrics(f"FULL PERIOD  ({symbol} {timeframe}, {n_bars} bars)", full_metrics)

    split_idx = int(n_bars * 0.75)
    print(
        f"\n[3/4] Running 75/25 split (IS: 0-{split_idx}, OOS: {split_idx}-{n_bars})..."
    )
    t2 = time.perf_counter()

    is_metrics = _run_on_slice(
        strategy_name=strategy_name,
        timeframe=timeframe,
        best_params=best_params,
        market_data=market_data,
        start_idx=0,
        end_idx=split_idx,
        broker=broker,
    )
    oos_metrics = _run_on_slice(
        strategy_name=strategy_name,
        timeframe=timeframe,
        best_params=best_params,
        market_data=market_data,
        start_idx=split_idx,
        end_idx=n_bars,
        broker=broker,
    )
    print(f"[3/4] Done in {time.perf_counter() - t2:.2f}s")

    _print_metrics(f"IN-SAMPLE (first 75%: {split_idx} bars)", is_metrics)
    _print_metrics(f"OUT-OF-SAMPLE (last 25%: {n_bars - split_idx} bars)", oos_metrics)

    if is_metrics["sharpe_ratio"] != 0:
        oos_is_ratio = oos_metrics["sharpe_ratio"] / is_metrics["sharpe_ratio"]
    else:
        oos_is_ratio = 0.0

    print(f"\n  OOS/IS Sharpe ratio: {oos_is_ratio:.3f}")
    if oos_is_ratio >= 0.5:
        print("  -> Moderate-to-good OOS retention (>=0.5)")
    elif oos_is_ratio > 0:
        print("  -> Weak OOS retention (0 < ratio < 0.5), possible overfitting")
    else:
        print("  -> Negative OOS Sharpe, likely overfit")

    is_bars, oos_bars, step_bars = _resolve_wfo_windows(timeframe)
    print(
        f"\n[4/4] Running walk-forward validation "
        f"(IS={is_bars}, OOS={oos_bars}, step={step_bars})..."
    )
    t3 = time.perf_counter()

    windows = generate_wfo_windows(
        n_bars,
        is_window_bars=is_bars,
        oos_window_bars=oos_bars,
        step_bars=step_bars,
    )

    oos_sharpes: list[float] = []
    oos_returns: list[float] = []
    oos_trades: list[int] = []

    if not windows:
        print("  Not enough data for walk-forward windows with current sizes.")
    else:
        print(f"  Generated {len(windows)} walk-forward windows")
        for idx, window in enumerate(windows, start=1):
            oos_metric = _run_on_slice(
                strategy_name=strategy_name,
                timeframe=timeframe,
                best_params=best_params,
                market_data=market_data,
                start_idx=window.oos_start,
                end_idx=window.oos_end,
                broker=broker,
            )
            oos_sharpes.append(float(oos_metric["sharpe_ratio"]))
            oos_returns.append(float(oos_metric["total_return"]))
            oos_trades.append(int(oos_metric["n_trades"]))
            print(
                f"  Window {idx}: OOS[{window.oos_start}:{window.oos_end}] "
                f"Sharpe={oos_metric['sharpe_ratio']:.4f}  "
                f"Return={oos_metric['total_return']:+.4f}  "
                f"Trades={oos_metric['n_trades']}"
            )

    print(f"[4/4] Done in {time.perf_counter() - t3:.2f}s")

    print(f"\n{'=' * 60}")
    print("  WALK-FORWARD OOS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Windows:         {len(oos_sharpes)}")
    if oos_sharpes:
        print(f"  Avg OOS Sharpe:  {sum(oos_sharpes) / len(oos_sharpes):.4f}")
        print(f"  Min OOS Sharpe:  {min(oos_sharpes):.4f}")
        print(f"  Max OOS Sharpe:  {max(oos_sharpes):.4f}")
        positive_count = sum(1 for value in oos_sharpes if value > 0)
        print(f"  Positive Sharpe: {positive_count}/{len(oos_sharpes)}")
        print(f"  Avg OOS Return:  {sum(oos_returns) / len(oos_returns):+.4f}")
        print(f"  Total OOS Trades: {sum(oos_trades)}")

    verdict = _verdict_from_wfo(oos_sharpes)
    print(f"\n{'=' * 60}")
    print("  VERDICT")
    print(f"{'=' * 60}")
    if verdict == "PASS":
        print("  PASS - Strategy shows robust OOS performance")
    elif verdict == "MARGINAL":
        print("  MARGINAL - Some OOS edge but inconsistent across windows")
    elif verdict == "FAIL":
        print("  FAIL - Likely overfit; OOS does not support IS results")
    else:
        print("  INSUFFICIENT DATA - Not enough WFO windows")

    _print_params(best_params)
    return {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "full_sharpe": float(full_metrics["sharpe_ratio"]),
        "oos_sharpe": float(oos_metrics["sharpe_ratio"]),
        "oos_is_ratio": float(oos_is_ratio),
        "wfo_avg_sharpe": float(sum(oos_sharpes) / len(oos_sharpes))
        if oos_sharpes
        else 0.0,
        "wfo_windows": len(oos_sharpes),
        "verdict": verdict,
    }


def main() -> None:
    broker = BrokerModel(
        commission_pct=BROKER_COMMISSION_PCT,
        fixed_fee=BROKER_FIXED_FEE,
        slippage_pct=BROKER_SLIPPAGE_PCT,
        max_volume_participation=MAX_VOLUME_PARTICIPATION,
        leverage=LEVERAGE,
    )

    summaries: list[dict[str, Any]] = []
    for config in VALIDATION_CONFIGS:
        summaries.append(_run_validation(config, broker))

    print(f"\n{'#' * 72}")
    print("### FINAL SUMMARY")
    print(f"{'#' * 72}")
    for summary in summaries:
        print(
            f"- {summary['strategy']} | {summary['symbol']} | {summary['timeframe']} "
            f"=> full_sharpe={summary['full_sharpe']:.4f}, "
            f"oos_sharpe={summary['oos_sharpe']:.4f}, "
            f"oos/is={summary['oos_is_ratio']:.3f}, "
            f"wfo_avg={summary['wfo_avg_sharpe']:.4f}, "
            f"windows={summary['wfo_windows']}, verdict={summary['verdict']}"
        )


if __name__ == "__main__":
    main()
