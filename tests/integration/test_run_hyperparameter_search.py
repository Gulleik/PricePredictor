"""Tests for run_hyperparameter_search script behavior."""

from types import SimpleNamespace

import pandas as pd
import pytest

import run_hyperparameter_search

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


class _DummyPortfolioSlice:
    def stats(self):
        return {"sharpe_ratio": 1.0}


class _DummyRunPortfolio:
    def __init__(self, returns: pd.Series) -> None:
        self._returns = returns

    def returns(self) -> pd.Series:
        return self._returns

    def stats(self) -> dict[str, float]:
        return {"sharpe_ratio": 1.0}


class _DummyPortfolio:
    def __init__(self, metric_series: pd.Series) -> None:
        self.metric_series = metric_series

    def sharpe_ratio(self) -> pd.Series:
        return self.metric_series

    def total_return(self) -> pd.Series:
        return self.metric_series

    def __getitem__(self, key):
        _ = key
        return _DummyPortfolioSlice()


def _dummy_run_result(
    idx_price: pd.DatetimeIndex,
) -> tuple[_DummyRunPortfolio, None, None]:
    return _DummyRunPortfolio(pd.Series(0.0, index=idx_price)), None, None


def test_main_uses_config_values(monkeypatch, capsys) -> None:
    """main should use configured symbol and top-n values."""
    monkeypatch.setattr(run_hyperparameter_search, "HYPERPARAM_SYMBOL", "ETH/USD")
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
        return pd.DataFrame({"close": range(100, 108)}, index=idx_price)

    class _DummyTrial:
        def __init__(self, fast: int, slow: int, value: float) -> None:
            self.params = {"fast": fast, "slow": slow}
            self.value = value
            self.user_attrs = {"invalid_combo": False}

    class _DummyStudy:
        def __init__(self) -> None:
            self.trials = [
                _DummyTrial(10, 40, 1.2),
                _DummyTrial(5, 30, 0.9),
            ]

    class _DummySearchResult:
        def __init__(self) -> None:
            self.study = _DummyStudy()
            self.best_fast = 10
            self.best_slow = 40
            self.best_objective_value = 1.2
            self.best_metrics = {
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.1,
                "calmar_ratio": 0.8,
                "max_drawdown_duration": 4.0,
            }

    def fake_optimize(
        price,
        *,
        fast_windows,
        slow_windows,
        objective,
        n_trials,
        timeout_seconds,
        sampler_name,
        seed,
        startup_trials,
        study_name,
        init_cash,
        portfolio_freq,
        next_bar_execution,
        friction_kwargs,
        max_size_array,
    ):
        assert isinstance(price, pd.Series)
        assert fast_windows == run_hyperparameter_search.FAST_WINDOWS
        assert slow_windows == run_hyperparameter_search.SLOW_WINDOWS
        assert objective == run_hyperparameter_search.SCAN_OBJECTIVE
        assert n_trials == run_hyperparameter_search.OPTUNA_N_TRIALS
        assert timeout_seconds == run_hyperparameter_search.OPTUNA_TIMEOUT_SECONDS
        assert sampler_name == run_hyperparameter_search.OPTUNA_SAMPLER
        assert seed == run_hyperparameter_search.OPTUNA_SEED
        assert startup_trials == run_hyperparameter_search.OPTUNA_STARTUP_TRIALS
        assert study_name == run_hyperparameter_search.OPTUNA_STUDY_NAME
        assert init_cash == run_hyperparameter_search.DEFAULT_INIT_CASH
        assert portfolio_freq == run_hyperparameter_search.DEFAULT_TIMEFRAME
        assert next_bar_execution == run_hyperparameter_search.ENABLE_NEXT_BAR_EXECUTION
        _ = max_size_array

        if run_hyperparameter_search.ENABLE_FRICTION_MODEL:
            assert friction_kwargs["fees"] > 0
            assert friction_kwargs["fixed_fees"] > 0
            assert friction_kwargs["slippage"] > 0
        else:
            assert friction_kwargs["fees"] == 0
            assert friction_kwargs["fixed_fees"] == 0
            assert friction_kwargs["slippage"] == 0

        return _DummySearchResult()

    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        fake_load_crypto_bars,
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "optimize_sma_parameters",
        fake_optimize,
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
        "run",
        lambda *args, **kwargs: _dummy_run_result(idx_price),
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
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: "price",
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "optimize_sma_parameters",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Unknown SCAN_OBJECTIVE")
        ),
    )

    with pytest.raises(ValueError, match="Unknown SCAN_OBJECTIVE"):
        run_hyperparameter_search.main()


def test_main_runs_wfo_mode(monkeypatch, capsys) -> None:
    """main should run WFO branch and print aggregate OOS summary."""
    idx_price = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")

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
            {"close": range(100, 140)},
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

    metric_is = pd.Series(
        [1.2, 0.8],
        index=pd.MultiIndex.from_tuples([(5, 30), (10, 40)]),
    )
    metric_oos = pd.Series([0.6], index=pd.MultiIndex.from_tuples([(5, 30)]))
    call_count = {"value": 0}

    def fake_run_scan(*args, **kwargs):
        call_count["value"] += 1
        # Two IS scans, two OOS scans, one full-grid scan
        if call_count["value"] in {1, 3, 5}:
            return _DummyPortfolio(metric_is)
        return _DummyPortfolio(metric_oos)

    monkeypatch.setattr(run_hyperparameter_search, "run_scan", fake_run_scan)
    monkeypatch.setattr(
        run_hyperparameter_search,
        "run",
        lambda *args, **kwargs: _dummy_run_result(idx_price),
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
            {"close": range(100, 200)},
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

    metric_is = pd.Series(
        [1.2, 0.8],
        index=pd.MultiIndex.from_tuples([(5, 30), (10, 40)]),
    )
    metric_oos = pd.Series([0.6], index=pd.MultiIndex.from_tuples([(5, 30)]))
    call_count = {"value": 0}

    def fake_run_scan(*args, **kwargs):
        call_count["value"] += 1
        # One IS scan, one OOS scan, one full-grid scan
        if call_count["value"] in {1, 3}:
            return _DummyPortfolio(metric_is)
        return _DummyPortfolio(metric_oos)

    monkeypatch.setattr(run_hyperparameter_search, "run_scan", fake_run_scan)
    monkeypatch.setattr(
        run_hyperparameter_search,
        "run",
        lambda *args, **kwargs: _dummy_run_result(idx_price),
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
