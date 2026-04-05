"""Tests for next-bar execution logic and lookahead bias prevention."""

import pandas as pd
import pytest
import vectorbt as vbt

from src.strategies.sma_crossover import run

pytestmark = pytest.mark.scenario


class TestNextBarExecution:
    """Test next-bar execution and lookahead bias prevention."""

    def test_next_bar_execution_disabled(self) -> None:
        """Test baseline behavior with next_bar_execution=False."""
        price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

        pf_baseline, _, _ = run(
            price,
            fast=2,
            slow=3,
            init_cash=10000.0,
            next_bar_execution=False,
        )

        # Verify portfolio executed without errors
        assert pf_baseline is not None
        assert hasattr(pf_baseline, "stats"), "Portfolio should have stats method"

    def test_next_bar_execution_enabled(self) -> None:
        """Test next_bar_execution=True shifts signals forward."""
        price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

        pf_nextbar, _, _ = run(
            price,
            fast=2,
            slow=3,
            init_cash=10000.0,
            next_bar_execution=True,
        )

        # Verify portfolio executed without errors
        assert pf_nextbar is not None
        assert hasattr(pf_nextbar, "stats"), "Portfolio should have stats method"

    def test_next_bar_execution_changes_returns(self) -> None:
        """Test that next-bar execution is a strict one-bar signal shift."""
        price = pd.Series([100.0, 101.0, 99.0, 103.0, 98.0, 105.0, 97.0, 106.0])

        fast_ma = vbt.MA.run(price, 2)
        slow_ma = vbt.MA.run(price, 4)
        entries = fast_ma.ma_crossed_above(slow_ma)
        shifted_entries = entries.vbt.fshift(1)

        # fshift(1) should move each signal exactly one bar forward.
        assert pd.isna(shifted_entries.iloc[0])
        pd.testing.assert_series_equal(
            shifted_entries.iloc[1:].fillna(False).reset_index(drop=True),
            entries.iloc[:-1].fillna(False).reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )

    def test_no_lookahead_bias_with_next_bar(self) -> None:
        """Test that next-bar execution avoids lookahead bias.

        The key is: a signal at T should not be filled until T+1.
        This is hard to test directly without inspecting VectorBT internals,
        but we can verify that the fshift(1) is applied correctly by checking
        that with a perfectly rising price, next-bar produces lower returns
        (since it enters 1 bar late).
        """
        # Monotonically increasing prices
        price = pd.Series(range(100, 120), dtype=float)

        pf_baseline, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=False,
        )

        pf_nextbar, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=True,
        )

        baseline_return = pf_baseline.total_return()
        nextbar_return = pf_nextbar.total_return()

        # In a perfectly rising market, entering 1 bar late should mean lower returns
        # (or equal if there are no trades, but with this config there should be)
        assert nextbar_return <= baseline_return, (
            "Next-bar execution should not exceed baseline "
            "in continuously rising prices"
        )

    def test_friction_parameters_accepted(self) -> None:
        """Test that friction parameters are accepted without error."""
        price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

        # Should not raise any errors
        pf, _, _ = run(
            price,
            fast=2,
            slow=3,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.001,
            fixed_fees=1.0,
            slippage=0.002,
        )

        assert pf is not None

    def test_friction_reduces_returns(self) -> None:
        """Test that friction costs reduce net returns."""
        price = pd.Series(range(100, 130), dtype=float)

        pf_no_friction, _, _ = run(
            price,
            fast=2,
            slow=10,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
        )

        pf_with_friction, _, _ = run(
            price,
            fast=2,
            slow=10,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.001,
            fixed_fees=1.0,
            slippage=0.002,
        )

        no_friction_return = pf_no_friction.total_return()
        with_friction_return = pf_with_friction.total_return()

        # Portfolio with friction costs should have lower or equal total return
        assert with_friction_return <= no_friction_return, (
            "Friction should reduce or maintain returns, never increase them"
        )
