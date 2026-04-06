"""Run hyperparameter search for SMA crossover and report best parameters."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.optuna_integration import (
    optimize_sma_parameters,
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
    HYPERPARAM_SYMBOL,
    HYPERPARAM_TOP_N,
    MAX_VOLUME_PARTICIPATION,
    OPTUNA_N_TRIALS,
    OPTUNA_SAMPLER,
    OPTUNA_SEED,
    OPTUNA_STARTUP_TRIALS,
    OPTUNA_STUDY_NAME,
    OPTUNA_TIMEOUT_SECONDS,
    REGIME_LOOKBACK_FAST,
    REGIME_LOOKBACK_SLOW,
    REGIME_SIDEWAYS_BAND,
    RESULTS_DIR,
    SCAN_OBJECTIVE,
    SENSITIVITY_HEATMAP_OUTPUT_PATH,
    SENSITIVITY_MATRIX_OUTPUT_PATH,
    SLOW_WINDOWS,
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
from src.strategies.sma_crossover import run, run_scan

OBJECTIVE_ACCESSORS = {
    "sharpe_ratio": "sharpe_ratio",
    "total_return": "total_return",
}

WFO_PRESET_WINDOWS = {
    "quick": (126, 30, 30),
    "balanced": (252, 63, 63),
    "robust": (365, 90, 90),
}


def _periods_per_year_from_freq(freq: str) -> int:
    normalized = freq.strip().lower()
    if normalized.endswith("h"):
        amount = int(normalized[:-1] or 1)
        return max(1, int(round((24 * 365) / amount)))
    if normalized.endswith("d"):
        amount = int(normalized[:-1] or 1)
        return max(1, int(round(252 / amount)))
    return 252


def _extract_scalar(value: Any) -> float:
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


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


def _build_optuna_config_snapshot() -> dict[str, Any]:
    return {
        "symbol": HYPERPARAM_SYMBOL,
        "timeframe": DEFAULT_TIMEFRAME,
        "scan_objective": SCAN_OBJECTIVE,
        "fast_windows": FAST_WINDOWS,
        "slow_windows": SLOW_WINDOWS,
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


def _trial_metric_series(study: Any) -> pd.Series:
    values: dict[tuple[int, int], float] = {}
    for trial in study.trials:
        fast = trial.params.get("fast")
        slow = trial.params.get("slow")
        if fast is None or slow is None or trial.value is None:
            continue
        if trial.user_attrs.get("invalid_combo", False):
            continue
        values[(int(fast), int(slow))] = float(trial.value)
    return pd.Series(values)


def _get_metric_series(pf, objective: str) -> pd.Series:
    metric_method = OBJECTIVE_ACCESSORS.get(objective)
    if metric_method is not None:
        metric_series = getattr(pf, metric_method)()
    elif objective in {"sortino_ratio", "calmar_ratio"}:
        metric_series = _metric_from_returns(
            pf.returns(),
            objective,
            periods_per_year=_periods_per_year_from_freq(DEFAULT_TIMEFRAME),
        )
    else:
        raise ValueError(
            f"Unknown SCAN_OBJECTIVE: {objective}. "
            "Use 'sharpe_ratio', 'total_return', 'sortino_ratio', or 'calmar_ratio'."
        )

    if not isinstance(metric_series, pd.Series):
        metric_series = pd.Series(metric_series)
    return metric_series


def _print_top_combinations(metric_series: pd.Series, objective: str) -> None:
    top_n = HYPERPARAM_TOP_N
    top_cols = metric_series.nlargest(top_n)
    print(f"Top {top_n} combinations:")
    for (fast, slow), val in top_cols.items():
        print(f"  fast={fast}, slow={slow}: {objective}={val:.4f}")


def _emit_sensitivity_outputs(metric_series: pd.Series, objective: str) -> None:
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
    *,
    friction_kwargs: dict,
    max_size_array: np.ndarray | None,
) -> None:
    search_result = optimize_sma_parameters(
        price,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
        objective=SCAN_OBJECTIVE,
        n_trials=OPTUNA_N_TRIALS,
        timeout_seconds=OPTUNA_TIMEOUT_SECONDS,
        sampler_name=OPTUNA_SAMPLER,
        seed=OPTUNA_SEED,
        startup_trials=OPTUNA_STARTUP_TRIALS,
        study_name=OPTUNA_STUDY_NAME,
        init_cash=DEFAULT_INIT_CASH,
        portfolio_freq=DEFAULT_TIMEFRAME,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        friction_kwargs=friction_kwargs,
        max_size_array=max_size_array,
    )

    best_fast = search_result.best_fast
    best_slow = search_result.best_slow
    best_value = search_result.best_objective_value

    print(f"Hyperparameter search ({SCAN_OBJECTIVE})")
    print(
        "  Best: "
        f"fast={best_fast}, slow={best_slow} -> "
        f"{SCAN_OBJECTIVE}={best_value:.4f}"
    )
    print(
        "  Metrics: "
        f"Sharpe={search_result.best_metrics['sharpe_ratio']:.4f}, "
        f"Sortino={search_result.best_metrics['sortino_ratio']:.4f}, "
        f"Calmar={search_result.best_metrics['calmar_ratio']:.4f}, "
        f"MaxDDDur(bars)={int(search_result.best_metrics['max_drawdown_duration'])}"
    )
    print()

    best_pf, _, _ = run(
        price,
        fast=best_fast,
        slow=best_slow,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        max_size=max_size_array,
        portfolio_freq=DEFAULT_TIMEFRAME,
        **friction_kwargs,
    )

    print("Best parameter stats:")
    print(best_pf.stats())
    print()

    metric_series = _trial_metric_series(search_result.study)
    if not metric_series.empty:
        _print_top_combinations(metric_series, SCAN_OBJECTIVE)
        _emit_sensitivity_outputs(metric_series, SCAN_OBJECTIVE)

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
    *,
    friction_kwargs: dict,
    max_size_array: np.ndarray | None,
) -> None:
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

    oos_by_param: dict[tuple[int, int], list[float]] = defaultdict(list)
    oos_values: list[float] = []

    for i, window in enumerate(windows, start=1):
        price_is = price.iloc[window.is_start : window.is_end]
        price_oos = price.iloc[window.oos_start : window.oos_end]

        max_size_is = None
        max_size_oos = None
        if max_size_array is not None:
            max_size_is = max_size_array[window.is_start : window.is_end]
            max_size_oos = max_size_array[window.oos_start : window.oos_end]

        pf_is = run_scan(
            price_is,
            fast_windows=FAST_WINDOWS,
            slow_windows=SLOW_WINDOWS,
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            **friction_kwargs,
            max_size=max_size_is,
        )
        is_metric = _get_metric_series(pf_is, SCAN_OBJECTIVE)
        best_col = is_metric.idxmax()
        best_fast, best_slow = best_col

        pf_oos = run_scan(
            price_oos,
            fast_windows=[int(best_fast)],
            slow_windows=[int(best_slow)],
            init_cash=DEFAULT_INIT_CASH,
            next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
            **friction_kwargs,
            max_size=max_size_oos,
        )
        oos_metric_series = _get_metric_series(pf_oos, WFO_OOS_METRIC)
        oos_value = float(oos_metric_series.iloc[0])
        oos_values.append(oos_value)
        oos_by_param[(int(best_fast), int(best_slow))].append(oos_value)

        print(
            f"WFO window {i}/{len(windows)}: "
            f"best_is=({best_fast},{best_slow}), "
            f"oos_{WFO_OOS_METRIC}={oos_value:.4f}"
        )

    grouped_summary = {
        key: float(np.mean(values)) for key, values in oos_by_param.items()
    }
    best_param = max(grouped_summary, key=grouped_summary.get)
    best_mean_oos = grouped_summary[best_param]

    print()
    print("Walk-forward optimization summary")
    print(
        "  Best OOS mean: "
        f"fast={best_param[0]}, slow={best_param[1]} -> "
        f"mean_{WFO_OOS_METRIC}={best_mean_oos:.4f}"
    )
    print(
        f"  Aggregate OOS {WFO_OOS_METRIC} across windows: "
        f"{float(np.mean(oos_values)):.4f}"
    )

    full_pf = run_scan(
        price,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        **friction_kwargs,
        max_size=max_size_array,
    )
    full_metric_series = _get_metric_series(full_pf, SCAN_OBJECTIVE)
    _print_top_combinations(full_metric_series, SCAN_OBJECTIVE)
    _emit_sensitivity_outputs(full_metric_series, SCAN_OBJECTIVE)

    best_pf, _, _ = run(
        price,
        fast=best_param[0],
        slow=best_param[1],
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        max_size=max_size_array,
        **friction_kwargs,
    )
    strategy_returns = best_pf.returns()
    _print_regime_breakdown(price, strategy_returns)


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
    if WFO_ENABLED:
        _run_wfo(price, friction_kwargs=friction_kwargs, max_size_array=max_size_array)
    else:
        _run_single_pass(
            price,
            friction_kwargs=friction_kwargs,
            max_size_array=max_size_array,
        )


if __name__ == "__main__":
    main()
