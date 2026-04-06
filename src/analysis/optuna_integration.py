"""Optuna-based Bayesian optimization utilities for strategy search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.annualization import periods_per_year_from_freq
from src.models.metrics import compute_advanced_metrics
from src.strategies.sma_crossover import run


@dataclass(frozen=True)
class SearchResult:
    """Structured output for one Optuna optimization run."""

    study: Any
    best_fast: int
    best_slow: int
    best_objective_value: float
    best_metrics: dict[str, float | int]


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

        pf, _, _ = run(
            price,
            fast=fast,
            slow=slow,
            init_cash=init_cash,
            next_bar_execution=next_bar_execution,
            max_size=max_size_array,
            portfolio_freq=portfolio_freq,
            **friction_kwargs,
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
    search_result: SearchResult,
    *,
    output_dir: Path,
    objective: str,
    config_snapshot: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist trial table and summary JSON for traceable research runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    trials_path = output_dir / f"{timestamp}_optuna_trials.csv"
    summary_path = output_dir / f"{timestamp}_optuna_summary.json"

    trial_df = search_result.study.trials_dataframe()
    trial_df.to_csv(trials_path, index=False)

    summary = {
        "timestamp_utc": timestamp,
        "objective": objective,
        "best_fast": search_result.best_fast,
        "best_slow": search_result.best_slow,
        "best_objective_value": search_result.best_objective_value,
        "best_metrics": search_result.best_metrics,
        "config": config_snapshot,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return trials_path, summary_path
