"""Tests for Kelly Criterion and position sizing utilities."""

import numpy as np
import pandas as pd
import pytest

from src.models.risk import (
    compute_kelly_fraction,
    estimate_kelly_from_portfolio,
    generate_position_sizes,
)

pytestmark = pytest.mark.unit


class TestComputeKellyFraction:
    """Test Kelly Criterion formula computation."""

    def test_kelly_breakeven(self) -> None:
        """Test that breakeven strategy (50% win, 1:1 payoff) yields 0 Kelly."""
        kelly = compute_kelly_fraction(
            win_rate=0.5,
            avg_win=100.0,
            avg_loss=100.0,
        )
        assert kelly == 0.0, "Breakeven strategy should yield Kelly = 0"

    def test_kelly_profitable(self) -> None:
        """Test profitable strategy with 55% win rate, 1.1:1 payoff."""
        kelly = compute_kelly_fraction(
            win_rate=0.55,
            avg_win=110.0,
            avg_loss=100.0,
        )
        assert kelly > 0, "Profitable strategy should yield positive Kelly"
        assert kelly <= 1.0, "Kelly should be clipped to [0, 1]"

    def test_kelly_conservative_factor(self) -> None:
        """Test conservative scaling with kelly_factor < 1."""
        kelly_full = compute_kelly_fraction(
            win_rate=0.6,
            avg_win=150.0,
            avg_loss=100.0,
            kelly_factor=1.0,
        )
        kelly_quarter = compute_kelly_fraction(
            win_rate=0.6,
            avg_win=150.0,
            avg_loss=100.0,
            kelly_factor=0.25,
        )
        assert kelly_quarter < kelly_full, "Conservative factor should reduce Kelly"
        assert abs(kelly_quarter - kelly_full * 0.25) < 1e-6, (
            "Quarter Kelly should be ~25%"
        )

    def test_kelly_losing_strategy(self) -> None:
        """Test that losing strategy returns 0 (not negative)."""
        kelly = compute_kelly_fraction(
            win_rate=0.4,
            avg_win=80.0,
            avg_loss=100.0,
        )
        assert kelly == 0.0, "Losing strategy should yield Kelly = 0 (clipped)"

    def test_kelly_invalid_win_rate(self) -> None:
        """Test validation of win_rate bounds."""
        with pytest.raises(ValueError, match="win_rate must be in"):
            compute_kelly_fraction(win_rate=1.5, avg_win=100, avg_loss=100)
        with pytest.raises(ValueError, match="win_rate must be in"):
            compute_kelly_fraction(win_rate=-0.1, avg_win=100, avg_loss=100)

    def test_kelly_invalid_avg_win(self) -> None:
        """Test validation of avg_win (must be non-negative)."""
        with pytest.raises(ValueError, match="avg_win must be"):
            compute_kelly_fraction(win_rate=0.5, avg_win=-10, avg_loss=100)

    def test_kelly_invalid_avg_loss(self) -> None:
        """Test validation of avg_loss (must be positive)."""
        with pytest.raises(ValueError, match="avg_loss must be"):
            compute_kelly_fraction(win_rate=0.5, avg_win=100, avg_loss=0)
        with pytest.raises(ValueError, match="avg_loss must be"):
            compute_kelly_fraction(win_rate=0.5, avg_win=100, avg_loss=-10)

    def test_kelly_zero_avg_win(self) -> None:
        """Test that zero average win yields 0 Kelly."""
        kelly = compute_kelly_fraction(
            win_rate=1.0,
            avg_win=0.0,
            avg_loss=100.0,
        )
        assert kelly == 0.0, "Zero average win should yield Kelly = 0"


class TestGeneratePositionSizes:
    """Test position size generation with Kelly Criterion."""

    def test_position_sizing_at_entries(self) -> None:
        """Test that position sizes are computed at entry signals."""
        price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        entries = pd.Series([True, False, True, False, False])

        sizes = generate_position_sizes(
            entries=entries,
            price=price,
            kelly_fraction=0.25,
            init_cash=10000.0,
        )

        # At entry bars (T=0, T=2), size should be kelly_frac * init_cash / price
        assert not np.isnan(sizes.iloc[0]), "First bar is entry, should have size"
        assert np.isnan(sizes.iloc[1]), "Non-entry bar should be NaN"
        assert not np.isnan(sizes.iloc[2]), "Third bar is entry, should have size"
        assert np.isnan(sizes.iloc[3]), "Non-entry bar should be NaN"
        assert np.isnan(sizes.iloc[4]), "Non-entry bar should be NaN"

    def test_position_sizing_calculation(self) -> None:
        """Test correct Kelly position size calculation."""
        price = pd.Series([100.0, 100.0])
        entries = pd.Series([True, False])
        kelly_frac = 0.1
        init_cash = 10000.0

        expected_size = kelly_frac * init_cash / 100.0  # 0.1 * 10000 / 100 = 10

        sizes = generate_position_sizes(
            entries=entries,
            price=price,
            kelly_fraction=kelly_frac,
            init_cash=init_cash,
        )

        assert sizes.iloc[0] == expected_size, f"Size should be {expected_size}"

    def test_position_sizing_no_leverage(self) -> None:
        """Test that position notional never exceeds init_cash."""
        price = pd.Series([1.0, 1.0])  # Very cheap price
        entries = pd.Series([True, False])
        kelly_frac = 1.0  # Full Kelly
        init_cash = 10000.0

        sizes = generate_position_sizes(
            entries=entries,
            price=price,
            kelly_fraction=kelly_frac,
            init_cash=init_cash,
        )

        notional = sizes.iloc[0] * price.iloc[0]
        assert notional <= init_cash, "Entry notional should not exceed init_cash"

    def test_position_sizing_low_price_asset_units(self) -> None:
        """Low-priced assets should allow higher unit counts at same notional."""
        price = pd.Series([0.5, 0.5])
        entries = pd.Series([True, False])

        sizes = generate_position_sizes(
            entries=entries,
            price=price,
            kelly_fraction=1.0,
            init_cash=10000.0,
        )

        # Full Kelly at $0.5 allows 20,000 units while staying at $10,000 notional.
        assert sizes.iloc[0] == 20000.0
        assert sizes.iloc[0] * price.iloc[0] == 10000.0

    def test_position_sizing_all_entries(self) -> None:
        """Test sizing when all bars are entries."""
        price = pd.Series([100.0, 200.0, 300.0])
        entries = pd.Series([True, True, True])

        sizes = generate_position_sizes(
            entries=entries,
            price=price,
            kelly_fraction=0.5,
            init_cash=10000.0,
        )

        assert all(~sizes.isna()), "All bars are entries, all should have sizes"
        assert sizes.iloc[0] != sizes.iloc[1], "Sizes should differ by price"

    def test_position_sizing_no_entries(self) -> None:
        """Test sizing when no bars are entries."""
        price = pd.Series([100.0, 100.0, 100.0])
        entries = pd.Series([False, False, False])

        sizes = generate_position_sizes(
            entries=entries,
            price=price,
            kelly_fraction=0.5,
            init_cash=10000.0,
        )

        assert all(sizes.isna()), "No entry signals, all should be NaN"

    def test_position_sizing_invalid_kelly(self) -> None:
        """Test validation of kelly_fraction bounds."""
        price = pd.Series([100.0])
        entries = pd.Series([True])

        with pytest.raises(ValueError, match="kelly_fraction must be"):
            generate_position_sizes(entries, price, kelly_fraction=1.5, init_cash=10000)
        with pytest.raises(ValueError, match="kelly_fraction must be"):
            generate_position_sizes(
                entries, price, kelly_fraction=-0.1, init_cash=10000
            )

    def test_position_sizing_invalid_cash(self) -> None:
        """Test validation of init_cash."""
        price = pd.Series([100.0])
        entries = pd.Series([True])

        with pytest.raises(ValueError, match="init_cash must be"):
            generate_position_sizes(entries, price, kelly_fraction=0.5, init_cash=-1000)
        with pytest.raises(ValueError, match="init_cash must be"):
            generate_position_sizes(entries, price, kelly_fraction=0.5, init_cash=0)

    def test_position_sizing_length_mismatch(self) -> None:
        """Test validation of input lengths."""
        price = pd.Series([100.0, 100.0])
        entries = pd.Series([True, False, True])  # Mismatched length

        with pytest.raises(ValueError, match="same length"):
            generate_position_sizes(entries, price, kelly_fraction=0.5, init_cash=10000)


class TestEstimateKellyFromPortfolio:
    """Test Kelly estimation from portfolio trade records."""

    def test_estimate_from_trade_pnl(self) -> None:
        """Estimator should compute Kelly from realized trade PnL."""

        records = pd.DataFrame(
            {
                "pnl": [100.0, -20.0, 80.0, -10.0],
            }
        )
        portfolio = type(
            "PortfolioMock",
            (),
            {"trades": type("TradesMock", (), {"records": records})()},
        )()

        kelly = estimate_kelly_from_portfolio(portfolio)

        # win_rate=0.5, avg_win=90, avg_loss=15 => Kelly ~= 0.4166667
        assert kelly == pytest.approx(0.4166666667, rel=1e-6)

    def test_estimate_returns_zero_on_missing_trade_data(self) -> None:
        """Estimator should safely return 0.0 if trade records are unavailable."""
        portfolio = object()
        assert estimate_kelly_from_portfolio(portfolio) == 0.0

    def test_estimate_returns_zero_on_all_wins_or_losses(self) -> None:
        """Estimator requires both winners and losers for stable Kelly calculation."""
        wins_only = pd.DataFrame({"pnl": [10.0, 5.0, 1.0]})
        losses_only = pd.DataFrame({"pnl": [-10.0, -5.0, -1.0]})

        wins_portfolio = type(
            "PortfolioMock",
            (),
            {"trades": type("TradesMock", (), {"records": wins_only})()},
        )()
        losses_portfolio = type(
            "PortfolioMock",
            (),
            {"trades": type("TradesMock", (), {"records": losses_only})()},
        )()

        assert estimate_kelly_from_portfolio(wins_portfolio) == 0.0
        assert estimate_kelly_from_portfolio(losses_portfolio) == 0.0
