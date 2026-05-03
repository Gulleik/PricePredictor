"""Optuna-based Bayesian optimization utilities for strategy search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.analysis.annualization import periods_per_year_from_freq
from src.config import KELLY_FACTOR, StrategyName
from src.models.metrics import compute_advanced_metrics
from src.models.risk import estimate_kelly_from_portfolio
from src.strategies import get_strategy_module
from src.strategies.sma_crossover import run


@dataclass(frozen=True)
class SearchResult:
    """Structured output for one Optuna optimization run."""

    study: Any
    best_fast: int
    best_slow: int
    best_objective_value: float
    best_metrics: dict[str, float | int]


@dataclass(frozen=True)
class GenericSearchResult:
    """Generic structured output for strategy optimization runs."""

    study: Any
    best_params: dict[str, Any]
    best_objective_value: float
    best_metrics: dict[str, float | int]
    strategy_name: str


def _as_float(value: Any) -> float:
    """Normalize scalar-like values from VectorBT accessors."""
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def _build_sampler(sampler_name: str, seed: int, startup_trials: int) -> Any:
    """Build an Optuna sampler from config values."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - guarded runtime dependency
        raise RuntimeError(
            "Optuna is required for milestone 5 search. Install dependencies first."
        ) from exc

    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=startup_trials,
            multivariate=True,
        )

    raise ValueError(f"Unknown OPTUNA_SAMPLER: {sampler_name}. Use 'tpe' or 'random'.")


def _kelly_sized_run(
    *,
    strategy_module: Any,
    price: pd.Series,
    init_cash: float,
    next_bar_execution: bool,
    max_size_array: np.ndarray | None,
    portfolio_freq: str,
    friction_kwargs: dict[str, float],
    run_kwargs: dict[str, Any],
    trial_params: dict[str, Any],
    leverage: float = 1.0,
) -> Any:
    """Run strategy and evaluate only the Kelly-sized portfolio.

    A base run is used solely to estimate the Kelly fraction from realized trades.
    The returned portfolio is always the Kelly-sized re-run.
    """
    base_result = strategy_module.run(
        price,
        init_cash=init_cash,
        next_bar_execution=next_bar_execution,
        max_size=max_size_array,
        portfolio_freq=portfolio_freq,
        leverage=leverage,
        **friction_kwargs,
        **run_kwargs,
        **trial_params,
    )
    base_pf = base_result[0] if isinstance(base_result, tuple) else base_result

    try:
        raw_kelly = float(estimate_kelly_from_portfolio(base_pf))
    except (ValueError, AttributeError, TypeError):
        raw_kelly = 0.0

    scaled_kelly = float(np.clip(raw_kelly * KELLY_FACTOR, 0.0, 1.0))

    # If Kelly fraction is negligible, return the base portfolio to avoid
    # a zero-sized re-run that always produces Sharpe=0.
    if scaled_kelly < 1e-6:
        return base_pf

    position_sizes = (scaled_kelly * init_cash * leverage / price).astype(float)

    kelly_result = strategy_module.run(
        price,
        init_cash=init_cash,
        next_bar_execution=next_bar_execution,
        max_size=max_size_array,
        position_sizes=position_sizes,
        portfolio_freq=portfolio_freq,
        leverage=leverage,
        **friction_kwargs,
        **run_kwargs,
        **trial_params,
    )
    return kelly_result[0] if isinstance(kelly_result, tuple) else kelly_result


def optimize_strategy_parameters(
    price: pd.Series,
    *,
    strategy_name: StrategyName,
    param_space: dict[str, list[Any]],
    objective: str,
    n_trials: int,
    timeout_seconds: int,
    sampler_name: str,
    seed: int,
    startup_trials: int,
    study_name: str,
    init_cash: float,
    portfolio_freq: str,
    next_bar_execution: bool,
    friction_kwargs: dict[str, float],
    max_size_array: np.ndarray | None,
    param_constraints: Callable[[dict[str, Any]], bool] | None = None,
    market_data: pd.DataFrame | None = None,
    leverage: float = 1.0,
) -> GenericSearchResult:
    """Generic strategy optimizer using Optuna.

    Args:
        price: Close price series.
        strategy_name: Name of the strategy from StrategyName.
        param_space: Dict mapping parameter names to lists of valid values.
        objective: Metric to maximize ('sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'total_return').
        n_trials: Number of Optuna trials.
        timeout_seconds: Search timeout in seconds (0 = no limit).
        sampler_name: 'tpe' or 'random'.
        seed: Random seed for sampler.
        startup_trials: Number of startup trials for TPE.
        study_name: Optuna study name.
        init_cash: Initial portfolio cash.
        portfolio_freq: Portfolio frequency (e.g., '1h').
        next_bar_execution: Whether to shift signals forward by one bar.
        friction_kwargs: Friction model keyword arguments.
        max_size_array: Per-bar max position size constraints.
        param_constraints: Optional function(params_dict) -> bool for
            constraint validation.
        market_data: Optional full market data (OHLC); needed by some strategies.

    Returns:
        GenericSearchResult with best parameters and metrics.
    """
    if n_trials <= 0:
        raise ValueError("OPTUNA_N_TRIALS must be positive")
    if timeout_seconds < 0:
        raise ValueError("OPTUNA_TIMEOUT_SECONDS must be >= 0")

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - guarded runtime dependency
        raise RuntimeError(
            "Optuna is required for milestone 5 search. Install dependencies first."
        ) from exc

    strategy_module = get_strategy_module(strategy_name)
    periods_per_year = periods_per_year_from_freq(portfolio_freq)
    sampler = _build_sampler(sampler_name, seed, startup_trials)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )

    def _objective(trial: Any) -> float:
        # Suggest parameters based on param_space
        trial_params = {}
        for param_name, values in param_space.items():
            if isinstance(values[0], int):
                trial_params[param_name] = int(
                    trial.suggest_categorical(param_name, values)
                )
            else:
                trial_params[param_name] = trial.suggest_categorical(param_name, values)

        # Apply constraints if provided
        if param_constraints is not None and not param_constraints(trial_params):
            trial.set_user_attr("invalid_combo", True)
            return float("-inf")

        # Call strategy run() with trial parameters
        try:
            # Build arguments based on strategy
            run_kwargs: dict[str, Any] = {}
            if strategy_name in {
                "trend_following",
                "volatility_breakout",
                "orb",
                "ema_ribbon_scalp",
                "bb_rsi_mean_reversion",
                "momentum_scalp",
                "adaptive_momentum",
            }:
                if market_data is None:
                    raise ValueError(
                        f"Strategy '{strategy_name}' requires market_data (OHLC)"
                    )
                run_kwargs["high"] = market_data["high"]
                run_kwargs["low"] = market_data["low"]

            if strategy_name in {"momentum_scalp", "adaptive_momentum"}:
                if market_data is not None and "volume" in market_data.columns:
                    run_kwargs["volume"] = market_data["volume"]

            pf = _kelly_sized_run(
                strategy_module=strategy_module,
                price=price,
                init_cash=init_cash,
                next_bar_execution=next_bar_execution,
                max_size_array=max_size_array,
                portfolio_freq=portfolio_freq,
                friction_kwargs=friction_kwargs,
                run_kwargs=run_kwargs,
                trial_params=trial_params,
                leverage=leverage,
            )
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("-inf")

        # Compute metrics
        returns = pf.returns()
        metrics = compute_advanced_metrics(returns, periods_per_year=periods_per_year)
        total_return = _as_float(pf.total_return())

        objective_map = {
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "calmar_ratio": metrics["calmar_ratio"],
            "total_return": total_return,
        }
        if objective not in objective_map:
            raise ValueError(
                f"Unknown objective: {objective}. "
                "Use 'sharpe_ratio', 'total_return', 'sortino_ratio', "
                "or 'calmar_ratio'."
            )

        # Store metrics as trial attributes
        for key, value in metrics.items():
            if key == "max_drawdown_duration":
                trial.set_user_attr(key, int(value))
            else:
                trial.set_user_attr(key, float(value))
        trial.set_user_attr("total_return", float(total_return))

        return float(objective_map[objective])

    study.optimize(_objective, n_trials=n_trials, timeout=timeout_seconds or None)

    # Filter valid trials
    valid_trials = [
        trial
        for trial in study.trials
        if not bool(trial.user_attrs.get("invalid_combo", False))
        and not trial.user_attrs.get("error")
        and trial.value is not None
        and np.isfinite(float(trial.value))
    ]
    if not valid_trials:
        invalid_count = sum(
            1
            for trial in study.trials
            if bool(trial.user_attrs.get("invalid_combo", False))
        )
        error_messages = [
            str(trial.user_attrs.get("error", ""))
            for trial in study.trials
            if trial.user_attrs.get("error")
        ]
        unique_errors = list(dict.fromkeys(error_messages))
        error_preview = "; ".join(unique_errors[:3]) if unique_errors else "none"
        raise ValueError(
            "Optuna search produced no valid trials. "
            "Increase OPTUNA_N_TRIALS, relax constraints, or expand parameter ranges. "
            "invalid_combo_trials="
            f"{invalid_count}, error_trials={len(error_messages)}, "
            f"sample_errors=[{error_preview}]"
        )

    best_trial = max(valid_trials, key=lambda trial: float(trial.value))
    best_metrics = {
        "sharpe_ratio": float(best_trial.user_attrs.get("sharpe_ratio", 0.0)),
        "sortino_ratio": float(best_trial.user_attrs.get("sortino_ratio", 0.0)),
        "calmar_ratio": float(best_trial.user_attrs.get("calmar_ratio", 0.0)),
        "max_drawdown": float(best_trial.user_attrs.get("max_drawdown", 0.0)),
        "max_drawdown_duration": int(
            best_trial.user_attrs.get("max_drawdown_duration", 0)
        ),
        "total_return": float(best_trial.user_attrs.get("total_return", 0.0)),
    }

    return GenericSearchResult(
        study=study,
        best_params=dict(best_trial.params),
        best_objective_value=float(best_trial.value),
        best_metrics=best_metrics,
        strategy_name=strategy_name,
    )


def optimize_sma_parameters(
    price: pd.Series,
    *,
    fast_windows: list[int],
    slow_windows: list[int],
    objective: str,
    n_trials: int,
    timeout_seconds: int,
    sampler_name: str,
    seed: int,
    startup_trials: int,
    study_name: str,
    init_cash: float,
    portfolio_freq: str,
    next_bar_execution: bool,
    friction_kwargs: dict[str, float],
    max_size_array: np.ndarray | None,
) -> SearchResult:
    """Optimize SMA parameters with Optuna and return best trial details."""
    if n_trials <= 0:
        raise ValueError("OPTUNA_N_TRIALS must be positive")
    if timeout_seconds < 0:
        raise ValueError("OPTUNA_TIMEOUT_SECONDS must be >= 0")

    has_valid_pair = any(fast < slow for fast in fast_windows for slow in slow_windows)
    if not has_valid_pair:
        raise ValueError(
            "No valid fast/slow combinations in search space. "
            "Ensure FAST_WINDOWS contains values smaller than SLOW_WINDOWS."
        )

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - guarded runtime dependency
        raise RuntimeError(
            "Optuna is required for milestone 5 search. Install dependencies first."
        ) from exc

    periods_per_year = periods_per_year_from_freq(portfolio_freq)
    sampler = _build_sampler(sampler_name, seed, startup_trials)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )

    def _objective(trial: Any) -> float:
        fast = int(trial.suggest_categorical("fast", fast_windows))
        slow = int(trial.suggest_categorical("slow", slow_windows))

        if fast >= slow:
            trial.set_user_attr("invalid_combo", True)
            return float("-inf")

        pf = _kelly_sized_run(
            strategy_module=SimpleNamespace(run=run),
            price=price,
            init_cash=init_cash,
            next_bar_execution=next_bar_execution,
            max_size_array=max_size_array,
            portfolio_freq=portfolio_freq,
            friction_kwargs=friction_kwargs,
            run_kwargs={},
            trial_params={"fast": fast, "slow": slow},
        )

        returns = pf.returns()
        metrics = compute_advanced_metrics(returns, periods_per_year=periods_per_year)
        total_return = _as_float(pf.total_return())

        objective_map = {
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "calmar_ratio": metrics["calmar_ratio"],
            "total_return": total_return,
        }
        if objective not in objective_map:
            raise ValueError(
                f"Unknown SCAN_OBJECTIVE: {objective}. "
                "Use 'sharpe_ratio', 'total_return', 'sortino_ratio', "
                "or 'calmar_ratio'."
            )

        for key, value in metrics.items():
            if key == "max_drawdown_duration":
                trial.set_user_attr(key, int(value))
            else:
                trial.set_user_attr(key, float(value))
        trial.set_user_attr("total_return", float(total_return))

        return float(objective_map[objective])

    study.optimize(_objective, n_trials=n_trials, timeout=timeout_seconds or None)

    valid_trials = [
        trial
        for trial in study.trials
        if not bool(trial.user_attrs.get("invalid_combo", False))
        and trial.value is not None
        and np.isfinite(float(trial.value))
    ]
    if not valid_trials:
        raise ValueError(
            "Optuna search produced no valid trials. "
            "Increase OPTUNA_N_TRIALS or narrow FAST_WINDOWS/SLOW_WINDOWS "
            "to reduce invalid fast>=slow samples."
        )

    best_trial = max(valid_trials, key=lambda trial: float(trial.value))
    best_metrics = {
        "sharpe_ratio": float(best_trial.user_attrs.get("sharpe_ratio", 0.0)),
        "sortino_ratio": float(best_trial.user_attrs.get("sortino_ratio", 0.0)),
        "calmar_ratio": float(best_trial.user_attrs.get("calmar_ratio", 0.0)),
        "max_drawdown": float(best_trial.user_attrs.get("max_drawdown", 0.0)),
        "max_drawdown_duration": int(
            best_trial.user_attrs.get("max_drawdown_duration", 0)
        ),
        "total_return": float(best_trial.user_attrs.get("total_return", 0.0)),
    }

    return SearchResult(
        study=study,
        best_fast=int(best_trial.params["fast"]),
        best_slow=int(best_trial.params["slow"]),
        best_objective_value=float(best_trial.value),
        best_metrics=best_metrics,
    )


def persist_search_artifacts(
    search_result: SearchResult | GenericSearchResult,
    *,
    output_dir: Path,
    objective: str,
    config_snapshot: dict[str, Any],
    save_trials: bool = True,
) -> Path | None:
    """Persist an Optuna trials CSV for traceable research runs.

    This intentionally does not publish summary JSON artifacts.
    """
    _ = objective
    _ = config_snapshot
    if not save_trials:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    trials_path = output_dir / f"{timestamp}_optuna_trials.csv"

    study = getattr(search_result, "study", None)
    rows: list[dict[str, Any]] = []
    if study is not None:
        for trial in getattr(study, "trials", []):
            row: dict[str, Any] = {
                "number": int(getattr(trial, "number", len(rows))),
                "value": (
                    float(trial.value)
                    if getattr(trial, "value", None) is not None
                    and np.isfinite(float(trial.value))
                    else np.nan
                ),
            }
            for key, value in getattr(trial, "params", {}).items():
                row[f"param_{key}"] = value
            for key, value in getattr(trial, "user_attrs", {}).items():
                row[f"attr_{key}"] = value
            rows.append(row)

    pd.DataFrame(rows).to_csv(trials_path, index=False)
    return trials_path


def persist_leaderboard(
    results: list[tuple[str, str, str, GenericSearchResult]],
    *,
    output_dir: Path,
    objective: str,
) -> Path:
    """Persist a ranked CSV leaderboard across batch combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leaderboard_path = output_dir / f"{timestamp}_leaderboard.csv"

    rows: list[dict[str, Any]] = []
    for strategy, symbol, timeframe, result in results:
        rows.append(
            {
                "strategy": strategy,
                "symbol": symbol,
                "timeframe": timeframe,
                "objective": objective,
                "objective_score": float(result.best_objective_value),
                "sharpe_ratio": float(result.best_metrics.get("sharpe_ratio", 0.0)),
                "sortino_ratio": float(result.best_metrics.get("sortino_ratio", 0.0)),
                "calmar_ratio": float(result.best_metrics.get("calmar_ratio", 0.0)),
                "max_drawdown": float(result.best_metrics.get("max_drawdown", 0.0)),
                "max_drawdown_duration": int(
                    result.best_metrics.get("max_drawdown_duration", 0)
                ),
                "total_return": float(result.best_metrics.get("total_return", 0.0)),
                "best_params": json.dumps(result.best_params, sort_keys=True),
            }
        )

    leaderboard_df = pd.DataFrame(rows)
    if not leaderboard_df.empty:
        leaderboard_df = leaderboard_df.sort_values(
            by="objective_score", ascending=False
        )
    leaderboard_df.to_csv(leaderboard_path, index=False)
    return leaderboard_path


def persist_batch_config_snapshot(
    *,
    output_dir: Path,
    snapshot: dict[str, Any],
) -> Path:
    """Persist batch-level configuration used by a matrix run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = output_dir / f"{timestamp}_batch_config.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot_path
