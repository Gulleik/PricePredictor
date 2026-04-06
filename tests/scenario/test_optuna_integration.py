"""Scenario tests for Optuna-backed hyperparameter integration."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import optuna_integration

pytestmark = pytest.mark.scenario


class _DummyPortfolio:
    def __init__(self, returns: pd.Series) -> None:
        self._returns = returns

    def returns(self) -> pd.Series:
        return self._returns

    def total_return(self) -> float:
        return float((1.0 + self._returns).prod() - 1.0)


class _FakeTrial:
    def __init__(self, params: dict[str, int]) -> None:
        self.params = params
        self.user_attrs: dict[str, float | bool] = {}
        self.value: float | None = None

    def suggest_categorical(self, name: str, choices: list[int]) -> int:
        value = self.params[name]
        assert value in choices
        return value

    def set_user_attr(self, key: str, value: float | bool) -> None:
        self.user_attrs[key] = value


class _FakeStudy:
    def __init__(self) -> None:
        self.trials: list[_FakeTrial] = []
        self.best_trial: _FakeTrial | None = None

    def optimize(self, objective, n_trials: int, timeout: int | None = None) -> None:
        _ = timeout
        trial_params = [
            {"fast": 10, "slow": 30},
            {"fast": 20, "slow": 20},
            {"fast": 15, "slow": 40},
        ]
        for params in trial_params[:n_trials]:
            trial = _FakeTrial(params)
            value = objective(trial)
            trial.value = float(value)
            self.trials.append(trial)

        valid_trials = [
            trial
            for trial in self.trials
            if not bool(trial.user_attrs.get("invalid_combo", False))
        ]
        self.best_trial = max(valid_trials, key=lambda trial: float(trial.value))


class _FakeSampler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_optuna_optimizer_returns_best_params(monkeypatch) -> None:
    """Optimizer should select best valid parameter pair and store metrics."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    price = pd.Series(np.linspace(100, 120, num=len(idx)), index=idx)

    returns_by_param: dict[tuple[int, int], pd.Series] = {
        (10, 30): pd.Series([0.01] * len(idx), index=idx),
        (15, 40): pd.Series([0.005, -0.001] * (len(idx) // 2), index=idx),
    }

    def fake_run(*args, **kwargs):
        fast = kwargs["fast"]
        slow = kwargs["slow"]
        pf = _DummyPortfolio(returns_by_param[(fast, slow)])
        return pf, None, None

    monkeypatch.setattr(optuna_integration, "run", fake_run)

    fake_optuna = SimpleNamespace(
        create_study=lambda **kwargs: _FakeStudy(),
        samplers=SimpleNamespace(
            TPESampler=lambda **kwargs: _FakeSampler(**kwargs),
            RandomSampler=lambda **kwargs: _FakeSampler(**kwargs),
        ),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    result = optuna_integration.optimize_sma_parameters(
        price,
        fast_windows=[10, 15, 20],
        slow_windows=[20, 30, 40],
        objective="sharpe_ratio",
        n_trials=3,
        timeout_seconds=5,
        sampler_name="tpe",
        seed=42,
        startup_trials=2,
        study_name="test-study",
        init_cash=10_000.0,
        portfolio_freq="1D",
        next_bar_execution=True,
        friction_kwargs={"fees": 0.0, "fixed_fees": 0.0, "slippage": 0.0},
        max_size_array=None,
    )

    assert result.best_fast == 10
    assert result.best_slow == 30
    assert "sortino_ratio" in result.best_metrics
    assert "calmar_ratio" in result.best_metrics
