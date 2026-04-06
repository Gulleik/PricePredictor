"""Tests for run_hyperparameter_search script behavior."""

from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

import run_hyperparameter_search
from src.analysis.optuna_integration import GenericSearchResult

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_sensitivity_outputs(monkeypatch, tmp_path) -> None:
    """Keep integration tests hermetic by isolating generated artifacts."""
    matrix_path = tmp_path / "sensitivity_matrix.csv"
    heatmap_path = tmp_path / "sensitivity_heatmap.png"
    monkeypatch.setattr(
        run_hyperparameter_search,
        "SENSITIVITY_MATRIX_OUTPUT_PATH",
        matrix_path,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "SENSITIVITY_HEATMAP_OUTPUT_PATH",
        heatmap_path,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "save_sensitivity_heatmap",
        lambda *args, **kwargs: None,
    )


class _DummyTrial:
    """Mock Optuna trial for testing."""

    def __init__(self, params: dict, value: float) -> None:
        self.params = params
        self.value = value
        self.user_attrs = {"invalid_combo": False}


class _DummyTrial:
    """Mock Optuna trial for testing."""

    def __init__(self, params: dict, value: float) -> None:
        self.params = params
        self.value = value
        self.user_attrs = {"invalid_combo": False}


class _DummyRunPortfolio:
    """Mock portfolio result from run() function."""

    def __init__(self, returns: pd.Series) -> None:
        self._returns = returns

    def returns(self) -> pd.Series:
        return self._returns

    def stats(self) -> dict[str, float]:
        return {"sharpe_ratio": 1.0}


def _dummy_run_result(
    idx_price: pd.DatetimeIndex,
) -> tuple[_DummyRunPortfolio, None, None]:
    """Create dummy result tuple to match run() signature."""
    return _DummyRunPortfolio(pd.Series(0.0, index=idx_price)), None, None


def test_main_uses_config_values(monkeypatch, capsys) -> None:
    """main should use configured symbol and top-n values."""
    monkeypatch.setattr(run_hyperparameter_search, "HYPERPARAM_SYMBOL", "ETH/USD")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "HYPERPARAM_SEARCH_STRATEGY",
        "sma_crossover",
    )
    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", False)
    monkeypatch.setattr(run_hyperparameter_search, "HYPERPARAM_TOP_N", 2)
    monkeypatch.setattr(run_hyperparameter_search, "SCAN_OBJECTIVE", "sharpe_ratio")
    monkeypatch.setattr(run_hyperparameter_search, "ENABLE_NEXT_BAR_EXECUTION", True)
    monkeypatch.setattr(run_hyperparameter_search, "ENABLE_FRICTION_MODEL", True)
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )

    idx_price = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")

    def fake_load_crypto_bars(symbol, start, end, timeframe):
        assert symbol == run_hyperparameter_search.HYPERPARAM_SYMBOL
        assert start == "2024-01-01T00:00:00+00:00"
        assert end == "2025-01-01T00:00:00+00:00"
        assert timeframe == run_hyperparameter_search.DEFAULT_TIMEFRAME
        # Return OHLC data for strategies needing high/low
        return pd.DataFrame(
            {
                "open": range(99, 107),
                "high": range(101, 109),
                "low": range(98, 106),
                "close": range(100, 108),
            },
            index=idx_price,
        )

    class _DummyStudy:
        def __init__(self) -> None:
            self.trials = [
                _DummyTrial({"fast": 10, "slow": 40}, 1.2),
                _DummyTrial({"fast": 5, "slow": 30}, 0.9),
            ]

    def fake_optimize_strategy_parameters(price, **kwargs):
        return GenericSearchResult(
            study=_DummyStudy(),
            best_params={"fast": 10, "slow": 40},
            best_objective_value=1.2,
            best_metrics={
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.1,
                "calmar_ratio": 0.8,
                "max_drawdown_duration": 4,
            },
            strategy_name="sma_crossover",
        )

    # Create a mock strategy module with a run method
    mock_strategy_module = mock.MagicMock()
    mock_strategy_module.run.return_value = _dummy_run_result(idx_price)

    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        fake_load_crypto_bars,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_strategy_module",
        lambda strategy_name: mock_strategy_module,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "optimize_strategy_parameters",
        fake_optimize_strategy_parameters,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "persist_search_artifacts",
        lambda *args, **kwargs: (
            run_hyperparameter_search.RESULTS_DIR / "trials.csv",
            run_hyperparameter_search.RESULTS_DIR / "summary.json",
        ),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "classify_regimes",
        lambda *args, **kwargs: pd.Series("sideways", index=idx_price),
    )

    run_hyperparameter_search.main()

    out = capsys.readouterr().out
    assert "Hyperparameter search (sharpe_ratio)" in out
    assert "Top 2 combinations:" in out
    assert "Sensitivity heatmap saved to:" in out
    assert "Optuna trials saved to:" in out


def test_main_raises_for_unknown_objective(monkeypatch) -> None:
    """main should fail fast when objective is invalid."""
    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", False)
    monkeypatch.setattr(run_hyperparameter_search, "SCAN_OBJECTIVE", "invalid_metric")

    idx_price = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")

    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "open": range(99, 104),
                "high": range(101, 106),
                "low": range(98, 103),
                "close": range(100, 105),
            },
            index=idx_price,
        ),
    )

    def raise_invalid_objective(price, **kwargs):
        raise ValueError("Unknown SCAN_OBJECTIVE")

    monkeypatch.setattr(
        run_hyperparameter_search,
        "optimize_strategy_parameters",
        raise_invalid_objective,
    )

    with pytest.raises(ValueError, match="Unknown SCAN_OBJECTIVE"):
        run_hyperparameter_search.main()


def test_main_runs_wfo_mode(monkeypatch, capsys) -> None:
    """main should run WFO branch and print aggregate OOS summary."""
    idx_price = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")

    monkeypatch.setattr(
        run_hyperparameter_search,
        "HYPERPARAM_SEARCH_STRATEGY",
        "sma_crossover",
    )
    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", True)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_MODE", "manual")
    monkeypatch.setattr(run_hyperparameter_search, "FAST_WINDOWS", [5, 10])
    monkeypatch.setattr(run_hyperparameter_search, "SLOW_WINDOWS", [30, 40])
    monkeypatch.setattr(run_hyperparameter_search, "WFO_IS_WINDOW_BARS", 20)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_OOS_FRACTION", 0.5)
    monkeypatch.setattr(run_hyperparameter_search, "SCAN_OBJECTIVE", "sharpe_ratio")
    monkeypatch.setattr(run_hyperparameter_search, "WFO_OOS_METRIC", "sharpe_ratio")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "open": [n - 1 for n in range(100, 140)],
                "high": [n + 1 for n in range(100, 140)],
                "low": [n - 2 for n in range(100, 140)],
                "close": list(range(100, 140)),
            },
            index=idx_price,
        ),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "generate_wfo_windows",
        lambda *args, **kwargs: [
            SimpleNamespace(is_start=0, is_end=20, oos_start=20, oos_end=30),
            SimpleNamespace(is_start=10, is_end=30, oos_start=30, oos_end=40),
        ],
    )

    call_count = {"value": 0}

    def fake_optimize_strategy_parameters(price, **kwargs):
        call_count["value"] += 1
        # Alternate between IS and OOS windows
        if call_count["value"] in {1, 3}:
            # IS scans return multiple param combinations
            return GenericSearchResult(
                study=type(
                    "Study",
                    (),
                    {
                        "trials": [
                            _DummyTrial({"fast": 5, "slow": 30}, 1.2),
                            _DummyTrial({"fast": 10, "slow": 40}, 0.8),
                        ]
                    },
                )(),
                best_params={"fast": 5, "slow": 30},
                best_objective_value=1.2,
                best_metrics={"sharpe_ratio": 1.2},
                strategy_name="sma_crossover",
            )
        else:
            # OOS scan
            return GenericSearchResult(
                study=type(
                    "Study",
                    (),
                    {"trials": [_DummyTrial({"fast": 5, "slow": 30}, 0.6)]},
                )(),
                best_params={"fast": 5, "slow": 30},
                best_objective_value=0.6,
                best_metrics={"sharpe_ratio": 0.6},
                strategy_name="sma_crossover",
            )

    # Create a mock strategy module with run and run_scan methods
    mock_strategy_module = mock.MagicMock()
    mock_strategy_module.run.return_value = _dummy_run_result(idx_price)

    # Mock run_scan to return a portfolio with metric accessors
    def mock_run_scan(*args, **kwargs):
        # Create a dummy portfolio that mimics VectorBT results
        # For SMA with 2 params, this should return a portfolio where
        # fast/slow combinations can be accessed via portfolio.sharpe_ratio()
        class _DummyScanPortfolio:
            def sharpe_ratio(self):
                # Return metrics indexed by (fast, slow) parameters
                return pd.Series(
                    {
                        (5, 30): 1.2,
                        (10, 40): 0.8,
                    },
                    name="sharpe_ratio",
                )

            def total_return(self):
                return pd.Series(
                    {
                        (5, 30): 1.2,
                        (10, 40): 0.8,
                    },
                    name="total_return",
                )

        return _DummyScanPortfolio()

    mock_strategy_module.run_scan.side_effect = mock_run_scan

    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_strategy_module",
        lambda strategy_name: mock_strategy_module,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "optimize_strategy_parameters",
        fake_optimize_strategy_parameters,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "classify_regimes",
        lambda *args, **kwargs: pd.Series("sideways", index=idx_price),
    )

    run_hyperparameter_search.main()

    out = capsys.readouterr().out
    assert "Walk-forward optimization summary" in out
    assert "Aggregate OOS sharpe_ratio across windows" in out


def test_main_raises_for_invalid_oos_fraction(monkeypatch) -> None:
    """WFO should fail fast when OOS fraction is outside (0, 1)."""
    idx_price = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")

    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", True)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_MODE", "manual")
    monkeypatch.setattr(run_hyperparameter_search, "WFO_OOS_FRACTION", 1.2)
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {"close": range(100, 150)},
            index=idx_price,
        ),
    )

    with pytest.raises(ValueError, match="WFO_OOS_FRACTION"):
        run_hyperparameter_search.main()


def test_main_runs_wfo_auto_mode(monkeypatch, capsys) -> None:
    """Auto mode should derive windows from available bar count."""
    idx_price = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")

    monkeypatch.setattr(
        run_hyperparameter_search,
        "HYPERPARAM_SEARCH_STRATEGY",
        "sma_crossover",
    )
    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", True)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_MODE", "auto")
    monkeypatch.setattr(run_hyperparameter_search, "FAST_WINDOWS", [5, 10])
    monkeypatch.setattr(run_hyperparameter_search, "SLOW_WINDOWS", [30, 40])
    monkeypatch.setattr(run_hyperparameter_search, "SCAN_OBJECTIVE", "sharpe_ratio")
    monkeypatch.setattr(run_hyperparameter_search, "WFO_OOS_METRIC", "sharpe_ratio")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "open": [n - 1 for n in range(100, 200)],
                "high": [n + 1 for n in range(100, 200)],
                "low": [n - 2 for n in range(100, 200)],
                "close": list(range(100, 200)),
            },
            index=idx_price,
        ),
    )

    captured = {"calls": []}

    def fake_generate_wfo_windows(
        n_bars,
        *,
        is_window_bars,
        oos_window_bars,
        step_bars,
    ):
        captured["calls"].append((n_bars, is_window_bars, oos_window_bars, step_bars))
        return [SimpleNamespace(is_start=0, is_end=60, oos_start=60, oos_end=75)]

    monkeypatch.setattr(
        run_hyperparameter_search,
        "generate_wfo_windows",
        fake_generate_wfo_windows,
    )

    call_count = {"value": 0}

    def fake_optimize_strategy_parameters(price, **kwargs):
        call_count["value"] += 1
        # One IS scan, one OOS scan, one full-grid scan
        if call_count["value"] in {1}:
            return GenericSearchResult(
                study=type(
                    "Study",
                    (),
                    {
                        "trials": [
                            _DummyTrial({"fast": 5, "slow": 30}, 1.2),
                            _DummyTrial({"fast": 10, "slow": 40}, 0.8),
                        ]
                    },
                )(),
                best_params={"fast": 5, "slow": 30},
                best_objective_value=1.2,
                best_metrics={"sharpe_ratio": 1.2},
                strategy_name="sma_crossover",
            )
        else:
            return GenericSearchResult(
                study=type(
                    "Study",
                    (),
                    {"trials": [_DummyTrial({"fast": 5, "slow": 30}, 0.6)]},
                )(),
                best_params={"fast": 5, "slow": 30},
                best_objective_value=0.6,
                best_metrics={"sharpe_ratio": 0.6},
                strategy_name="sma_crossover",
            )

    # Create a mock strategy module with run and run_scan methods
    mock_strategy_module = mock.MagicMock()
    mock_strategy_module.run.return_value = _dummy_run_result(idx_price)

    # Mock run_scan to return a portfolio with metric accessors
    def mock_run_scan(*args, **kwargs):
        class _DummyScanPortfolio:
            def sharpe_ratio(self):
                return pd.Series(
                    {
                        (5, 30): 1.2,
                        (10, 40): 0.8,
                    },
                    name="sharpe_ratio",
                )

            def total_return(self):
                return pd.Series(
                    {
                        (5, 30): 1.2,
                        (10, 40): 0.8,
                    },
                    name="total_return",
                )

        return _DummyScanPortfolio()

    mock_strategy_module.run_scan.side_effect = mock_run_scan

    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_strategy_module",
        lambda strategy_name: mock_strategy_module,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "optimize_strategy_parameters",
        fake_optimize_strategy_parameters,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "classify_regimes",
        lambda *args, **kwargs: pd.Series("sideways", index=idx_price),
    )

    run_hyperparameter_search.main()

    assert captured["calls"]
    _, is_bars, oos_bars, step_bars = captured["calls"][0]
    assert is_bars == 60
    assert oos_bars == 15
    assert step_bars == 15

    out = capsys.readouterr().out
    assert "WFO config: mode=auto" in out


def test_main_raises_for_unknown_wfo_mode(monkeypatch) -> None:
    """main should fail fast on unsupported WFO mode values."""
    idx_price = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")

    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", True)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_MODE", "invalid")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {"close": range(100, 120)},
            index=idx_price,
        ),
    )

    with pytest.raises(ValueError, match="Unknown WFO_MODE"):
        run_hyperparameter_search.main()


def test_main_raises_for_unknown_wfo_preset(monkeypatch) -> None:
    """Preset mode should fail fast when configured preset name is invalid."""
    idx_price = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")

    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", True)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_MODE", "preset")
    monkeypatch.setattr(run_hyperparameter_search, "WFO_PRESET", "invalid")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {"close": range(100, 140)},
            index=idx_price,
        ),
    )

    with pytest.raises(ValueError, match="Unknown WFO_PRESET"):
        run_hyperparameter_search.main()


def test_main_raises_when_wfo_generates_no_windows(monkeypatch) -> None:
    """WFO should raise when generated window list is empty."""
    idx_price = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")

    monkeypatch.setattr(run_hyperparameter_search, "WFO_ENABLED", True)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_MODE", "manual")
    monkeypatch.setattr(run_hyperparameter_search, "WFO_IS_WINDOW_BARS", 20)
    monkeypatch.setattr(run_hyperparameter_search, "WFO_OOS_FRACTION", 0.5)
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: pd.DataFrame(
            {"close": range(100, 140)},
            index=idx_price,
        ),
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "generate_wfo_windows",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(ValueError, match="No valid WFO windows generated"):
        run_hyperparameter_search.main()


def test_trial_metric_series_keeps_best_value_for_duplicate_params() -> None:
    """Duplicate trials for same pair should keep the best observed objective."""

    class _DummyTrial:
        def __init__(
            self,
            fast: int,
            slow: int,
            value: float,
            *,
            invalid_combo: bool = False,
        ) -> None:
            self.params = {"fast": fast, "slow": slow}
            self.value = value
            self.user_attrs = {"invalid_combo": invalid_combo}

    class _DummyStudy:
        def __init__(self) -> None:
            self.trials = [
                _DummyTrial(10, 40, 0.70),
                _DummyTrial(10, 40, 1.10),
                _DummyTrial(5, 30, 0.90),
                _DummyTrial(20, 20, 5.00, invalid_combo=True),
            ]

    metric_series = run_hyperparameter_search._trial_metric_series(_DummyStudy())

    assert float(metric_series.loc[(10, 40)]) == pytest.approx(1.10)
    assert float(metric_series.loc[(5, 30)]) == pytest.approx(0.90)
    assert (20, 20) not in metric_series.index
