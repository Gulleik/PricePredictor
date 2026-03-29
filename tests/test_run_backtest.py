"""Tests for run_backtest script behavior."""

from types import SimpleNamespace

import run_backtest


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
    figure = _DummyFigure()
    price = _DummyPrice(figure)
    pf = _DummyPortfolio()
    fast_ma = SimpleNamespace(ma=SimpleNamespace(vbt=_DummyPlotter(figure)))
    slow_ma = SimpleNamespace(ma=SimpleNamespace(vbt=_DummyPlotter(figure)))

    monkeypatch.setattr(run_backtest, "BACKTEST_SYMBOL", "LTC/USD")
    monkeypatch.setattr(run_backtest, "BACKTEST_FAST_WINDOW", 7)
    monkeypatch.setattr(run_backtest, "BACKTEST_SLOW_WINDOW", 21)
    monkeypatch.setattr(run_backtest, "BACKTEST_RENDER_CHART", True)
    monkeypatch.setattr(run_backtest, "DEFAULT_INIT_CASH", 1234.0)
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

    def fake_sma_run(price_arg, fast, slow, init_cash):
        assert price_arg is price
        assert fast == run_backtest.BACKTEST_FAST_WINDOW
        assert slow == run_backtest.BACKTEST_SLOW_WINDOW
        assert init_cash == run_backtest.DEFAULT_INIT_CASH
        return pf, fast_ma, slow_ma

    monkeypatch.setattr(run_backtest, "load_crypto_bars", fake_load_crypto_bars)
    monkeypatch.setattr(run_backtest, "sma_run", fake_sma_run)

    run_backtest.main()

    out = capsys.readouterr().out
    assert "Loading market data for LTC/USD" in out
    assert "Running SMA crossover backtest" in out
    assert "Rendering chart" in out
    assert pf.positions_plotted
    assert figure.shown


def test_main_skips_chart_when_disabled(monkeypatch, capsys) -> None:
    """main should skip plotting when chart rendering is disabled in config."""
    price = _NoPlotPrice()
    pf = _DummyPortfolio()

    monkeypatch.setattr(run_backtest, "BACKTEST_SYMBOL", "LTC/USD")
    monkeypatch.setattr(run_backtest, "BACKTEST_FAST_WINDOW", 7)
    monkeypatch.setattr(run_backtest, "BACKTEST_SLOW_WINDOW", 21)
    monkeypatch.setattr(run_backtest, "BACKTEST_RENDER_CHART", False)
    monkeypatch.setattr(run_backtest, "DEFAULT_INIT_CASH", 1234.0)
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
        lambda *args, **kwargs: (pf, object(), object()),
    )

    run_backtest.main()

    out = capsys.readouterr().out
    assert "Chart rendering disabled by config" in out
    assert not pf.positions_plotted
