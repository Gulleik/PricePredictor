"""Run hyperparameter search and report best parameters for configured strategy."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.annualization import periods_per_year_from_freq
from src.analysis.optuna_integration import (
    optimize_strategy_parameters,
    persist_search_artifacts,
)
from src.analysis.sensitivity import build_sensitivity_matrix, save_sensitivity_heatmap
from src.analysis.wfo import generate_wfo_windows
from src.config import (
    BROKER_COMMISSION_PCT,
    BROKER_FIXED_FEE,
    BROKER_SLIPPAGE_PCT,
    DEFAULT_INIT_CASH,
    DEFAULT_TIMEFRAME,
    ENABLE_FRICTION_MODEL,
    ENABLE_NEXT_BAR_EXECUTION,
    FAST_WINDOWS,
    HYPERPARAM_SEARCH_STRATEGY,
    HYPERPARAM_SYMBOL,
    HYPERPARAM_TOP_N,
    MAX_VOLUME_PARTICIPATION,
    MEAN_REVERSION_BB_STD_VALUES,
    MEAN_REVERSION_BB_WINDOW_VALUES,
    MEAN_REVERSION_OVERBOUGHT_VALUES,
    MEAN_REVERSION_OVERSOLD_VALUES,
    MEAN_REVERSION_RSI_PERIOD_VALUES,
    MEAN_REVERSION_VOL_MAX_VALUES,
    OPTUNA_N_TRIALS,
    OPTUNA_SAMPLER,
    OPTUNA_SEED,
    OPTUNA_STARTUP_TRIALS,
    OPTUNA_STUDY_NAME,
    OPTUNA_TIMEOUT_SECONDS,
    ORB_BREAKOUT_BUFFER_VALUES,
    ORB_RANGE_BARS_VALUES,
    REGIME_LOOKBACK_FAST,
    REGIME_LOOKBACK_SLOW,
    REGIME_SIDEWAYS_BAND,
    RESULTS_DIR,
    SCAN_OBJECTIVE,
    SENSITIVITY_HEATMAP_OUTPUT_PATH,
    SENSITIVITY_MATRIX_OUTPUT_PATH,
    SLOW_WINDOWS,
    TREND_ATR_STOP_MULTIPLES,
    TREND_ATR_WINDOWS,
    TREND_EMA_FAST_WINDOWS,
    TREND_EMA_SLOW_WINDOWS,
    VOL_BREAKOUT_ATR_MIN_VALUES,
    VOL_BREAKOUT_ATR_WINDOWS,
    VOL_BREAKOUT_DONCHIAN_WINDOWS,
    WFO_ENABLED,
    WFO_IS_WINDOW_BARS,
    WFO_MODE,
    WFO_OOS_FRACTION,
    WFO_OOS_METRIC,
    WFO_PRESET,
    WFO_STEP_BARS,
)
from src.data import get_close_price_series, load_crypto_bars
from src.date_range import get_default_date_range
from src.models.broker import BrokerModel
from src.models.metrics import compute_advanced_metrics
from src.models.regime import classify_regimes
from src.strategies import get_strategy_module

OBJECTIVE_ACCESSORS = {
    "sharpe_ratio": "sharpe_ratio",
    "total_return": "total_return",
}

WFO_PRESET_WINDOWS = {
    "quick": (126, 30, 30),
    "balanced": (252, 63, 63),
    "robust": (365, 90, 90),
}


def _build_param_space_and_constraints(
    strategy_name: str,
) -> tuple[dict[str, list[Any]], Any | None]:
    """Build parameter space dict and constraint function for a strategy."""
    if strategy_name == "sma_crossover":
        return (
            {
                "fast": FAST_WINDOWS,
                "slow": SLOW_WINDOWS,
            },
            lambda p: p["fast"] < p["slow"],
        )
    elif strategy_name == "mean_reversion":
        return (
            {
                "rsi_period": MEAN_REVERSION_RSI_PERIOD_VALUES,
                "oversold": MEAN_REVERSION_OVERSOLD_VALUES,
                "overbought": MEAN_REVERSION_OVERBOUGHT_VALUES,
                "bb_window": MEAN_REVERSION_BB_WINDOW_VALUES,
                "bb_std": MEAN_REVERSION_BB_STD_VALUES,
                "vol_max_annualized": MEAN_REVERSION_VOL_MAX_VALUES,
            },
            None,  # No constraints
        )
    elif strategy_name == "trend_following":
        return (
            {
                "fast_window": TREND_EMA_FAST_WINDOWS,
                "slow_window": TREND_EMA_SLOW_WINDOWS,
                "atr_window": TREND_ATR_WINDOWS,
                "atr_stop_multiple": TREND_ATR_STOP_MULTIPLES,
            },
            lambda p: p["fast_window"] < p["slow_window"],
        )
    elif strategy_name == "volatility_breakout":
        return (
            {
                "donchian_window": VOL_BREAKOUT_DONCHIAN_WINDOWS,
                "atr_window": VOL_BREAKOUT_ATR_WINDOWS,
                "atr_min_fraction": VOL_BREAKOUT_ATR_MIN_VALUES,
            },
            None,
        )
    elif strategy_name == "orb":
        return (
            {
                "range_bars": ORB_RANGE_BARS_VALUES,
                "breakout_buffer": ORB_BREAKOUT_BUFFER_VALUES,
            },
            None,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def _build_optuna_config_snapshot() -> dict[str, Any]:
    return {
        "strategy": HYPERPARAM_SEARCH_STRATEGY,
        "symbol": HYPERPARAM_SYMBOL,
        "timeframe": DEFAULT_TIMEFRAME,
        "scan_objective": SCAN_OBJECTIVE,
        "optuna_sampler": OPTUNA_SAMPLER,
        "optuna_n_trials": OPTUNA_N_TRIALS,
        "optuna_timeout_seconds": OPTUNA_TIMEOUT_SECONDS,
        "optuna_seed": OPTUNA_SEED,
        "optuna_startup_trials": OPTUNA_STARTUP_TRIALS,
        "optuna_study_name": OPTUNA_STUDY_NAME,
        "wfo_enabled": WFO_ENABLED,
        "enable_next_bar_execution": ENABLE_NEXT_BAR_EXECUTION,
        "enable_friction_model": ENABLE_FRICTION_MODEL,
    }


def _metric_from_returns(
    returns: pd.Series | pd.DataFrame,
    objective: str,
    periods_per_year: int,
) -> pd.Series:
    if isinstance(returns, pd.Series):
        returns = returns.to_frame("metric")

    values: dict[Any, float] = {}
    for col in returns.columns:
        metrics = compute_advanced_metrics(
            returns[col],
            periods_per_year=periods_per_year,
        )
        if objective == "sortino_ratio":
            values[col] = metrics["sortino_ratio"]
        elif objective == "calmar_ratio":
            values[col] = metrics["calmar_ratio"]
        else:
            raise ValueError(
                f"Unknown SCAN_OBJECTIVE: {objective}. "
                "Use 'sharpe_ratio', 'total_return', 'sortino_ratio', "
                "or 'calmar_ratio'."
            )

    metric_series = pd.Series(values)
    if isinstance(metric_series.index, pd.Index) and len(metric_series) == 1:
        metric_series.index = pd.Index([metric_series.index[0]])
    return metric_series


def _trial_metric_series(study: Any) -> pd.Series:
    """Extract best metric per parameter combination from Optuna study."""
    values: dict[tuple | str, float] = {}
    for trial in study.trials:
        if trial.value is None or not np.isfinite(float(trial.value)):
            continue
        if trial.user_attrs.get("invalid_combo", False):
            continue

        # Create a hashable key from parameters
        params = trial.params
        if len(params) == 2 and "fast" in params and "slow" in params:
            # SMA-style: use tuple
            key = (int(params["fast"]), int(params["slow"]))
        elif len(params) == 1:
            # Single parameter
            key = next(iter(params.values()))
        else:
            # Multiple parameters: convert to tuple of sorted items
            key = tuple(sorted(params.items()))

        score = float(trial.value)
        existing = values.get(key)
        if existing is None or score > existing:
            values[key] = score
    return pd.Series(values)


def _get_metric_series(pf, objective: str) -> pd.Series:
    metric_method = OBJECTIVE_ACCESSORS.get(objective)
    if metric_method is not None:
        metric_series = getattr(pf, metric_method)()
    elif objective in {"sortino_ratio", "calmar_ratio"}:
        metric_series = _metric_from_returns(
            pf.returns(),
            objective,
            periods_per_year=periods_per_year_from_freq(DEFAULT_TIMEFRAME),
        )
    else:
        raise ValueError(
            f"Unknown SCAN_OBJECTIVE: {objective}. "
            "Use 'sharpe_ratio', 'total_return', 'sortino_ratio', or 'calmar_ratio'."
        )

    if not isinstance(metric_series, pd.Series):
        metric_series = pd.Series(metric_series)
    return metric_series


def _print_top_combinations(
    search_result: Any,
    objective: str,
) -> None:
    """Print top N parameter combinations from optimization result."""
    top_n = HYPERPARAM_TOP_N
    metric_series = _trial_metric_series(search_result.study)
    top_cols = metric_series.nlargest(top_n)
    print(f"Top {top_n} combinations:")
    for params, val in top_cols.items():
        if (
            isinstance(params, tuple)
            and len(params) == 2
            and isinstance(params[0], int)
        ):
            # SMA-style: (fast, slow) - both integers
            fast, slow = params
            print(f"  fast={fast}, slow={slow}: {objective}={val:.4f}")
        elif isinstance(params, tuple) and params and isinstance(params[0], tuple):
            # Multi-param: tuple of (key, value) pairs from sorted items
            param_str = ", ".join(f"{k}={v}" for k, v in params)
            print(f"  {param_str}: {objective}={val:.4f}")
        else:
            # Fallback for other formats (single param, etc.)
            print(f"  {params}: {objective}={val:.4f}")


def _emit_sensitivity_outputs(
    search_result: Any,
    objective: str,
    strategy_name: str,
) -> None:
    """Emit sensitivity matrix and heatmap if strategy supports it (i.e., SMA)."""
    if strategy_name != "sma_crossover":
        # Sensitivity heatmap is SMA-specific (2D parameter space)
        return

    metric_series = _trial_metric_series(search_result.study)
    matrix = build_sensitivity_matrix(
        metric_series,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
    )
    SENSITIVITY_MATRIX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(SENSITIVITY_MATRIX_OUTPUT_PATH)
    save_sensitivity_heatmap(
        matrix,
        SENSITIVITY_HEATMAP_OUTPUT_PATH,
        metric_name=objective,
    )
    print(f"Sensitivity matrix saved to: {SENSITIVITY_MATRIX_OUTPUT_PATH}")
    print(f"Sensitivity heatmap saved to: {SENSITIVITY_HEATMAP_OUTPUT_PATH}")


def _print_regime_breakdown(price: pd.Series, strategy_returns: pd.Series) -> None:
    regimes = classify_regimes(
        price,
        fast_window=REGIME_LOOKBACK_FAST,
        slow_window=REGIME_LOOKBACK_SLOW,
        sideways_band=REGIME_SIDEWAYS_BAND,
    )

    aligned_returns = strategy_returns.reindex(price.index).fillna(0.0)
    report_df = pd.DataFrame({"regime": regimes, "returns": aligned_returns})
    summary = report_df.groupby("regime", observed=True)["returns"].agg(
        ["count", "mean"]
    )

    print()
    print("Regime breakdown (bar returns):")
    for regime, row in summary.iterrows():
        count = int(row["count"])
        mean_return = float(row["mean"])
        print(f"  {regime}: count={count}, mean_return={mean_return:.6f}")


def _resolve_wfo_windows(n_bars: int) -> tuple[int, int, int]:
    """Resolve IS/OOS/step bars from WFO configuration mode."""
    if n_bars <= 0:
        raise ValueError("n_bars must be positive for WFO.")

    if WFO_MODE == "preset":
        preset = WFO_PRESET_WINDOWS.get(WFO_PRESET)
        if preset is None:
            raise ValueError(
                f"Unknown WFO_PRESET: {WFO_PRESET}. "
                "Use 'quick', 'balanced', or 'robust'."
            )
        return preset

    if WFO_MODE == "manual":
        if not 0 < WFO_OOS_FRACTION < 1:
            raise ValueError("WFO_OOS_FRACTION must be between 0 and 1 (exclusive).")
        oos_window_bars = max(2, int(round(WFO_IS_WINDOW_BARS * WFO_OOS_FRACTION)))
        return WFO_IS_WINDOW_BARS, oos_window_bars, WFO_STEP_BARS

    if WFO_MODE == "auto":
        if n_bars < 8:
            raise ValueError("WFO auto mode requires at least 8 bars.")

        is_window_bars = max(4, int(round(n_bars * 0.60)))
        oos_window_bars = max(2, int(round(is_window_bars * 0.25)))
        step_bars = oos_window_bars

        if is_window_bars + oos_window_bars > n_bars:
            is_window_bars = max(4, int(round(n_bars * 0.75)))
            oos_window_bars = max(2, n_bars - is_window_bars)
            step_bars = oos_window_bars

        return is_window_bars, oos_window_bars, step_bars

    raise ValueError(
        f"Unknown WFO_MODE: {WFO_MODE}. Use 'auto', 'preset', or 'manual'."
    )


def _run_single_pass(
    price: pd.Series,
    market_data: pd.DataFrame,
    *,
    friction_kwargs: dict,
    max_size_array: np.ndarray | None,
) -> None:
    """Run single-pass hyperparameter optimization."""
    strategy_module = get_strategy_module(HYPERPARAM_SEARCH_STRATEGY)
    param_space, param_constraints = _build_param_space_and_constraints(
        HYPERPARAM_SEARCH_STRATEGY
    )

    study_name = f"{OPTUNA_STUDY_NAME}_{HYPERPARAM_SEARCH_STRATEGY}"
    search_result = optimize_strategy_parameters(
        price,
        strategy_name=HYPERPARAM_SEARCH_STRATEGY,
        param_space=param_space,
        objective=SCAN_OBJECTIVE,
        n_trials=OPTUNA_N_TRIALS,
        timeout_seconds=OPTUNA_TIMEOUT_SECONDS,
        sampler_name=OPTUNA_SAMPLER,
        seed=OPTUNA_SEED,
        startup_trials=OPTUNA_STARTUP_TRIALS,
        study_name=study_name,
        init_cash=DEFAULT_INIT_CASH,
        portfolio_freq=DEFAULT_TIMEFRAME,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        friction_kwargs=friction_kwargs,
        max_size_array=max_size_array,
        param_constraints=param_constraints,
        market_data=market_data,
    )

    best_params = search_result.best_params
    params_str = ", ".join(f"{k}={v}" for k, v in sorted(best_params.items()))
    print(f"Hyperparameter search ({SCAN_OBJECTIVE})")
    print(
        "  Best: "
        f"{params_str} -> {SCAN_OBJECTIVE}="
        f"{search_result.best_objective_value:.4f}"
    )
    print(
        "  Metrics: "
        f"Sharpe={search_result.best_metrics['sharpe_ratio']:.4f}, "
        f"Sortino={search_result.best_metrics['sortino_ratio']:.4f}, "
        f"Calmar={search_result.best_metrics['calmar_ratio']:.4f}, "
        f"MaxDDDur(bars)={search_result.best_metrics['max_drawdown_duration']}"
    )
    print()

    # Run best parameters on full dataset
    run_args = [price]
    if HYPERPARAM_SEARCH_STRATEGY in {
        "trend_following",
        "volatility_breakout",
        "orb",
    }:
        run_args.extend([market_data["high"], market_data["low"]])

    best_pf = strategy_module.run(
        *run_args,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        max_size=max_size_array,
        portfolio_freq=DEFAULT_TIMEFRAME,
        **friction_kwargs,
        **best_params,
    )
    if isinstance(best_pf, tuple):
        best_pf = best_pf[0]

    print("Best parameter stats:")
    print(best_pf.stats())
    print()

    # Print top combinations
    _print_top_combinations(search_result, SCAN_OBJECTIVE)

    # Emit sensitivity outputs (SMA-specific)
    _emit_sensitivity_outputs(search_result, SCAN_OBJECTIVE, HYPERPARAM_SEARCH_STRATEGY)

    trials_path, summary_path = persist_search_artifacts(
        search_result,
        output_dir=RESULTS_DIR,
        objective=SCAN_OBJECTIVE,
        config_snapshot=_build_optuna_config_snapshot(),
    )
    print(f"Optuna trials saved to: {trials_path}")
    print(f"Run summary saved to: {summary_path}")

    strategy_returns = best_pf.returns()
    _print_regime_breakdown(price, strategy_returns)


def _run_wfo(
    price: pd.Series,
    market_data: pd.DataFrame,
    *,
    friction_kwargs: dict,
    max_size_array: np.ndarray | None,
) -> None:
    """Run walk-forward optimization."""
    strategy_module = get_strategy_module(HYPERPARAM_SEARCH_STRATEGY)
    param_space, param_constraints = _build_param_space_and_constraints(
        HYPERPARAM_SEARCH_STRATEGY
    )

    is_window_bars, oos_window_bars, step_bars = _resolve_wfo_windows(len(price))

    print(
        "WFO config: "
        f"mode={WFO_MODE}, is={is_window_bars}, oos={oos_window_bars}, step={step_bars}"
    )

    windows = generate_wfo_windows(
        len(price),
        is_window_bars=is_window_bars,
        oos_window_bars=oos_window_bars,
        step_bars=step_bars,
    )
    if not windows:
        raise ValueError(
            "No valid WFO windows generated. Increase data range or reduce "
            "WFO window sizes for current data length."
        )

    oos_by_param: dict[tuple | str, list[float]] = defaultdict(list)
    oos_values: list[float] = []

    for i, window in enumerate(windows, start=1):
        price_is = price.iloc[window.is_start : window.is_end]
        price_oos = price.iloc[window.oos_start : window.oos_end]

        max_size_is = None
        max_size_oos = None
        if max_size_array is not None:
            max_size_is = max_size_array[window.is_start : window.is_end]
            max_size_oos = max_size_array[window.oos_start : window.oos_end]

        # Slice market data for this window if needed
        market_data_is = None
        market_data_oos = None
        if HYPERPARAM_SEARCH_STRATEGY in {
            "trend_following",
            "volatility_breakout",
            "orb",
        }:
            market_data_is = market_data.iloc[window.is_start : window.is_end]
            market_data_oos = market_data.iloc[window.oos_start : window.oos_end]

        # Run in-sample scan with all parameter combinations
        scan_args_is = [price_is]
        if market_data_is is not None:
            scan_args_is.extend([market_data_is["high"], market_data_is["low"]])

        pf_is = strategy_module.run_scan(
            *scan_args_is,
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            **friction_kwargs,
            max_size=max_size_is,
            **{k: v for k, v in param_space.items()},
        )
        is_metric = _get_metric_series(pf_is, SCAN_OBJECTIVE)
        best_col = is_metric.idxmax()

        # For SMA, best_col is a tuple (fast, slow)
        # For other strategies, it may be different
        best_params = {}
        if isinstance(best_col, tuple):
            # Multi-param strategy (SMA, trend_following)
            param_names = list(param_space.keys())
            for j, param_name in enumerate(param_names):
                best_params[param_name] = best_col[j]
            param_key = best_col
        else:
            # Single param (shouldn't happen in 2+ param strategies)
            param_key = best_col

        # Run out-of-sample with best in-sample parameters
        scan_args_oos = [price_oos]
        if market_data_oos is not None:
            scan_args_oos.extend([market_data_oos["high"], market_data_oos["low"]])

        pf_oos = strategy_module.run_scan(
            *scan_args_oos,
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            **friction_kwargs,
            max_size=max_size_oos,
            **{k: [best_params.get(k, v[0])] for k, v in param_space.items()},
        )
        oos_metric_series = _get_metric_series(pf_oos, WFO_OOS_METRIC)
        oos_value = float(oos_metric_series.iloc[0])
        oos_values.append(oos_value)
        oos_by_param[param_key].append(oos_value)

        param_str = (
            ", ".join(f"{k}={v}" for k, v in best_params.items())
            if best_params
            else str(param_key)
        )
        print(
            f"WFO window {i}/{len(windows)}: "
            f"best_is=({param_str}), "
            f"oos_{WFO_OOS_METRIC}={oos_value:.4f}"
        )

    # Determine best parameter combination across windows
    if oos_by_param:
        grouped_summary = {
            key: float(np.mean(values)) for key, values in oos_by_param.items()
        }
        best_param = max(grouped_summary, key=grouped_summary.get)
        best_mean_oos = grouped_summary[best_param]

        print()
        print("Walk-forward optimization summary")
        print(
            "  Best OOS mean: "
            f"{best_param} -> "
            f"mean_{WFO_OOS_METRIC}={best_mean_oos:.4f}"
        )
    else:
        best_param = None
        best_mean_oos = None

    print(
        f"  Aggregate OOS {WFO_OOS_METRIC} across windows: "
        f"{float(np.mean(oos_values)):.4f}"
    )

    # Run scan on full dataset with all parameters for sensitivity analysis
    scan_args_full = [price]
    if HYPERPARAM_SEARCH_STRATEGY in {
        "trend_following",
        "volatility_breakout",
        "orb",
    }:
        scan_args_full.extend([market_data["high"], market_data["low"]])

    full_pf = strategy_module.run_scan(
        *scan_args_full,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        **friction_kwargs,
        max_size=max_size_array,
        **{k: v for k, v in param_space.items()},
    )
    full_metric_series = _get_metric_series(full_pf, SCAN_OBJECTIVE)
    _print_top_combinations_from_series(full_metric_series, SCAN_OBJECTIVE)
    # Note: sensitivity heatmap is SMA-specific, skipped for other strategies

    # Run best parameter on full dataset
    if best_param is not None:
        if isinstance(best_param, tuple):
            param_names = list(param_space.keys())
            best_params = {
                param_names[j]: best_param[j] for j in range(len(param_names))
            }
        else:
            best_params = {list(param_space.keys())[0]: best_param}

        run_args_best = [price]
        if HYPERPARAM_SEARCH_STRATEGY in {
            "trend_following",
            "volatility_breakout",
            "orb",
        }:
            run_args_best.extend([market_data["high"], market_data["low"]])

        best_pf = strategy_module.run(
            *run_args_best,
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            max_size=max_size_array,
            **friction_kwargs,
            **best_params,
        )
        if isinstance(best_pf, tuple):
            best_pf = best_pf[0]

        strategy_returns = best_pf.returns()
        _print_regime_breakdown(price, strategy_returns)


def _print_top_combinations_from_series(
    metric_series: pd.Series, objective: str
) -> None:
    """Print top combinations from a metric series."""
    top_n = HYPERPARAM_TOP_N
    top_cols = metric_series.nlargest(top_n)
    print(f"Top {top_n} combinations:")
    for params, val in top_cols.items():
        if isinstance(params, tuple):
            params_str = ", ".join(str(p) for p in params)
        else:
            params_str = str(params)
        print(f"  {params_str}: {objective}={val:.4f}")


def main() -> None:
    symbol = HYPERPARAM_SYMBOL
    start, end = get_default_date_range()
    market_data = load_crypto_bars(
        symbol,
        start=start,
        end=end,
        timeframe=DEFAULT_TIMEFRAME,
    )
    price = get_close_price_series(market_data)

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

    print(f"Optimizing strategy: {HYPERPARAM_SEARCH_STRATEGY}")
    print(f"Symbol: {symbol}, Data points: {len(price)}")
    print()

    if WFO_ENABLED:
        _run_wfo(
            price,
            market_data,
            friction_kwargs=friction_kwargs,
            max_size_array=max_size_array,
        )
    else:
        _run_single_pass(
            price,
            market_data,
            friction_kwargs=friction_kwargs,
            max_size_array=max_size_array,
        )


if __name__ == "__main__":
    main()
