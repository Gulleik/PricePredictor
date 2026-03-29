"""Tests for run_hyperparameter_search script behavior."""

import pandas as pd
import pytest

import run_hyperparameter_search


class _DummyPortfolioSlice:
    def stats(self):
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


def test_main_uses_config_values(monkeypatch, capsys) -> None:
    """main should use configured symbol and top-n values."""
    idx = pd.MultiIndex.from_tuples([(5, 30), (10, 40), (15, 60)])
    metric_series = pd.Series([1.2, 0.7, 0.5], index=idx)
    pf = _DummyPortfolio(metric_series)

    monkeypatch.setattr(run_hyperparameter_search, "HYPERPARAM_SYMBOL", "ETH/USD")
    monkeypatch.setattr(run_hyperparameter_search, "HYPERPARAM_TOP_N", 2)
    monkeypatch.setattr(run_hyperparameter_search, "SCAN_OBJECTIVE", "sharpe_ratio")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )

    def fake_load_crypto_bars(symbol, start, end, timeframe):
        assert symbol == run_hyperparameter_search.HYPERPARAM_SYMBOL
        assert start == "2024-01-01T00:00:00+00:00"
        assert end == "2025-01-01T00:00:00+00:00"
        assert timeframe == run_hyperparameter_search.DEFAULT_TIMEFRAME
        return "price-series"

    def fake_run_scan(price, fast_windows, slow_windows, init_cash):
        assert price == "price-series"
        assert fast_windows == run_hyperparameter_search.FAST_WINDOWS
        assert slow_windows == run_hyperparameter_search.SLOW_WINDOWS
        assert init_cash == run_hyperparameter_search.DEFAULT_INIT_CASH
        return pf

    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        fake_load_crypto_bars,
    )
    monkeypatch.setattr(run_hyperparameter_search, "run_scan", fake_run_scan)

    run_hyperparameter_search.main()

    out = capsys.readouterr().out
    assert "Hyperparameter search (sharpe_ratio)" in out
    assert "Top 2 combinations:" in out


def test_main_raises_for_unknown_objective(monkeypatch) -> None:
    """main should fail fast when objective is invalid."""
    monkeypatch.setattr(run_hyperparameter_search, "SCAN_OBJECTIVE", "invalid_metric")
    monkeypatch.setattr(
        run_hyperparameter_search,
        "load_crypto_bars",
        lambda *args, **kwargs: "price",
    )
    monkeypatch.setattr(
        run_hyperparameter_search,
        "run_scan",
        lambda *args, **kwargs: _DummyPortfolio(
            pd.Series([1.0], index=pd.MultiIndex.from_tuples([(5, 30)]))
        ),
    )

    with pytest.raises(ValueError, match="Unknown SCAN_OBJECTIVE"):
        run_hyperparameter_search.main()
