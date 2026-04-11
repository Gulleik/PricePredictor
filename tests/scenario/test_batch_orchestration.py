"""Scenario tests for Milestone 7 batch orchestration artifacts."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.optuna_integration import (
    GenericSearchResult,
    persist_batch_config_snapshot,
    persist_leaderboard,
)

pytestmark = pytest.mark.scenario


def test_leaderboard_columns_and_ordering(tmp_path) -> None:
    """Leaderboard CSV should include key columns and be sorted by objective."""
    results = [
        (
            "sma_crossover",
            "BTC/USD",
            "1h",
            GenericSearchResult(
                study=SimpleNamespace(trials=[]),
                best_params={"fast": 10, "slow": 30},
                best_objective_value=0.8,
                best_metrics={
                    "sharpe_ratio": 0.8,
                    "sortino_ratio": 0.7,
                    "calmar_ratio": 0.5,
                    "max_drawdown": -0.2,
                    "max_drawdown_duration": 12,
                    "total_return": 0.12,
                },
                strategy_name="sma_crossover",
            ),
        ),
        (
            "orb",
            "ETH/USD",
            "4h",
            GenericSearchResult(
                study=SimpleNamespace(trials=[]),
                best_params={"range_bars": 3, "breakout_buffer": 0.001},
                best_objective_value=1.2,
                best_metrics={
                    "sharpe_ratio": 1.2,
                    "sortino_ratio": 1.0,
                    "calmar_ratio": 0.6,
                    "max_drawdown": -0.15,
                    "max_drawdown_duration": 8,
                    "total_return": 0.18,
                },
                strategy_name="orb",
            ),
        ),
    ]

    leaderboard_path = persist_leaderboard(
        results,
        output_dir=tmp_path,
        objective="sharpe_ratio",
    )
    df = pd.read_csv(leaderboard_path)

    assert list(df.columns) == [
        "strategy",
        "symbol",
        "timeframe",
        "objective",
        "objective_score",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration",
        "total_return",
        "best_params",
    ]
    assert df.iloc[0]["strategy"] == "orb"
    assert float(df.iloc[0]["objective_score"]) == pytest.approx(1.2)
    assert float(df.iloc[1]["objective_score"]) == pytest.approx(0.8)


def test_batch_config_snapshot_persisted(tmp_path) -> None:
    """Batch config snapshot should be written as JSON."""
    snapshot_path = persist_batch_config_snapshot(
        output_dir=tmp_path,
        snapshot={
            "batch_mode": "quick",
            "hyperparam_strategies": ["sma_crossover"],
            "hyperparam_symbols": ["BTC/USD"],
            "hyperparam_timeframes": ["1h"],
            "scan_objective": "sharpe_ratio",
        },
    )

    assert snapshot_path.exists()
    content = snapshot_path.read_text(encoding="utf-8")
    assert '"batch_mode": "quick"' in content
