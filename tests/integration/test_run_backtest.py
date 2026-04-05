"""Tests for run_backtest script behavior."""

from types import SimpleNamespace

import pytest

import run_backtest

pytestmark = pytest.mark.integration


class _DummyFigure:
    def __init__(self) -> None:
        self.shown = False

    def show(self) -> None:
        self.shown = True


class _DummyPlotter:
    def __init__(self, figure: _DummyFigure) -> None:
        self.figure = figure

    def plot(self, **kwargs):
        _ = kwargs
        return self.figure


class _DummyPrice:
    def __init__(self, figure: _DummyFigure) -> None:
        self.vbt = _DummyPlotter(figure)

    def __len__(self) -> int:
        return 3


class _DummyPortfolio:
    def __init__(self) -> None:
        self.positions_plotted = False

    def stats(self):
        return {"total_return": 0.1}

    def plot_positions(self, **kwargs) -> None:
        _ = kwargs
        self.positions_plotted = True


class _NoPlotPrice:
    def __len__(self) -> int:
        return 3

    @property
    def vbt(self):
        raise AssertionError(
            "Plotting should not be used when chart rendering is disabled"
        )


def test_main_uses_config_values_and_runs_flow(monkeypatch, capsys) -> None:
    """main should use module config constants and complete orchestration flow."""
    import pandas as pd

    figure = _DummyFigure()
    price = _DummyPrice(figure)
    pf = _DummyPortfolio()

    # Create  MA mocks with methods needed for Kelly computation
    fast_ma = SimpleNamespace(
        ma=SimpleNamespace(vbt=_DummyPlotter(figure)),
        ma_crossed_above=lambda x: pd.Series([False, False, False]),
        ma_crossed_below=lambda x: pd.Series([False, False, False]),
    )
    slow_ma = SimpleNamespace(
        ma=SimpleNamespace(vbt=_DummyPlotter(figure)),
        ma_crossed_above=lambda x: pd.Series([False, False, False]),
        ma_crossed_below=lambda x: pd.Series([False, False, False]),
    )

    monkeypatch.setattr(run_backtest, "BACKTEST_SYMBOL", "LTC/USD")
    monkeypatch.setattr(run_backtest, "BACKTEST_FAST_WINDOW", 7)
    monkeypatch.setattr(run_backtest, "BACKTEST_SLOW_WINDOW", 21)
    monkeypatch.setattr(run_backtest, "BACKTEST_RENDER_CHART", True)
    monkeypatch.setattr(run_backtest, "DEFAULT_INIT_CASH", 1234.0)
    monkeypatch.setattr(run_backtest, "ENABLE_NEXT_BAR_EXECUTION", True)
    monkeypatch.setattr(run_backtest, "ENABLE_FRICTION_MODEL", True)
    monkeypatch.setattr(run_backtest, "KELLY_FACTOR", 0.25)
    monkeypatch.setattr(
        run_backtest,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )

    def fake_load_crypto_bars(symbol, start, end, timeframe):
        assert symbol == run_backtest.BACKTEST_SYMBOL
        assert start == "2024-01-01T00:00:00+00:00"
        assert end == "2025-01-01T00:00:00+00:00"
        assert timeframe == run_backtest.DEFAULT_TIMEFRAME
        return price

    def fake_sma_run(
        price_arg,
        fast,
        slow,
        init_cash,
        next_bar_execution=False,
        fees=0,
        fixed_fees=0,
        slippage=0,
        max_size=None,
        position_sizes=None,
    ):
        assert price_arg is price
        assert fast == run_backtest.BACKTEST_FAST_WINDOW
        assert slow == run_backtest.BACKTEST_SLOW_WINDOW
        assert init_cash == run_backtest.DEFAULT_INIT_CASH
        assert next_bar_execution == run_backtest.ENABLE_NEXT_BAR_EXECUTION
        # Friction parameters are now centralized in broker model
        if run_backtest.ENABLE_FRICTION_MODEL:
            assert fees > 0, "When friction enabled, fees should be > 0"
            assert fixed_fees > 0, "When friction enabled, fixed_fees should be > 0"
            assert slippage > 0, "When friction enabled, slippage should be > 0"
        else:
            assert fees == 0
            assert fixed_fees == 0
            assert slippage == 0
        return pf, fast_ma, slow_ma

    monkeypatch.setattr(run_backtest, "load_crypto_bars", fake_load_crypto_bars)
    monkeypatch.setattr(run_backtest, "sma_run", fake_sma_run)
    # Mock Kelly functions to avoid issues with dummy data
    monkeypatch.setattr(
        run_backtest,
        "estimate_conservative_kelly",
        lambda e, ex, p: 0.0,
    )
    monkeypatch.setattr(
        run_backtest,
        "generate_position_sizes",
        lambda e, p, k, c: None,
    )

    run_backtest.main()

    out = capsys.readouterr().out
    assert "Loading market data for LTC/USD" in out
    assert "Running SMA crossover backtest" in out
    assert "Rendering chart" in out
    assert pf.positions_plotted
    assert figure.shown


def test_main_skips_chart_when_disabled(monkeypatch, capsys) -> None:
    """main should skip plotting when chart rendering is disabled in config."""
    import pandas as pd

    price = _NoPlotPrice()
    pf = _DummyPortfolio()

    # Mock MA objects with methods for Kelly computation
    fast_ma = SimpleNamespace(
        ma=SimpleNamespace(vbt=object()),
        ma_crossed_above=lambda x: pd.Series([False, False, False]),
        ma_crossed_below=lambda x: pd.Series([False, False, False]),
    )
    slow_ma = SimpleNamespace(
        ma=SimpleNamespace(vbt=object()),
        ma_crossed_above=lambda x: pd.Series([False, False, False]),
        ma_crossed_below=lambda x: pd.Series([False, False, False]),
    )

    monkeypatch.setattr(run_backtest, "BACKTEST_SYMBOL", "LTC/USD")
    monkeypatch.setattr(run_backtest, "BACKTEST_FAST_WINDOW", 7)
    monkeypatch.setattr(run_backtest, "BACKTEST_SLOW_WINDOW", 21)
    monkeypatch.setattr(run_backtest, "BACKTEST_RENDER_CHART", False)
    monkeypatch.setattr(run_backtest, "DEFAULT_INIT_CASH", 1234.0)
    monkeypatch.setattr(run_backtest, "ENABLE_NEXT_BAR_EXECUTION", True)
    monkeypatch.setattr(run_backtest, "ENABLE_FRICTION_MODEL", False)
    monkeypatch.setattr(run_backtest, "KELLY_FACTOR", 0.25)
    monkeypatch.setattr(
        run_backtest,
        "get_default_date_range",
        lambda: ("2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
    )

    monkeypatch.setattr(
        run_backtest,
        "load_crypto_bars",
        lambda *args, **kwargs: price,
    )
    monkeypatch.setattr(
        run_backtest,
        "sma_run",
        lambda *args, **kwargs: (pf, fast_ma, slow_ma),
    )
    # Mock Kelly functions
    monkeypatch.setattr(
        run_backtest,
        "estimate_conservative_kelly",
        lambda e, ex, p: 0.0,
    )
    monkeypatch.setattr(
        run_backtest,
        "generate_position_sizes",
        lambda e, p, k, c: None,
    )

    run_backtest.main()

    out = capsys.readouterr().out
    assert "Chart rendering disabled by config" in out
    assert not pf.positions_plotted
