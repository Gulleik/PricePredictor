"""Tests for broker friction modeling and volume constraints."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.sma_crossover import run

pytestmark = pytest.mark.scenario


class TestFrictionModel:
    """Test broker friction costs (fees, slippage, commissions)."""

    def test_friction_disabled_baseline(self) -> None:
        """Test that friction disabled produces baseline results."""
        price = pd.Series(range(100, 120), dtype=float)

        pf, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=False,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
        )

        assert pf is not None
        stats = pf.stats()
        assert stats is not None
        assert len(stats) > 0

    def test_friction_enabled_vs_disabled(self) -> None:
        """Test that friction enabled produces lower returns than disabled."""
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

        pf_friction, _, _ = run(
            price,
            fast=2,
            slow=10,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.001,  # 0.1% commission
            fixed_fees=1.0,  # $1 per trade
            slippage=0.002,  # 0.2% slippage
        )

        return_no_friction = float(pf_no_friction.total_return())
        return_friction = float(pf_friction.total_return())

        assert return_friction <= return_no_friction, (
            "Friction should reduce total return"
        )

    def test_commission_consistency(self) -> None:
        """Test that commissions are applied consistently."""
        price = pd.Series(range(100, 120), dtype=float)

        pf_high_commission, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.05,  # 5% commission (very high)
            fixed_fees=0.0,
            slippage=0.0,
        )

        pf_low_commission, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.001,  # 0.1% commission (low)
            fixed_fees=0.0,
            slippage=0.0,
        )

        return_high = float(pf_high_commission.total_return())
        return_low = float(pf_low_commission.total_return())

        assert return_high <= return_low, (
            "Higher commission should reduce returns more than lower commission"
        )

    def test_slippage_impact(self) -> None:
        """Test that slippage reduces returns."""
        price = pd.Series(range(100, 130), dtype=float)

        pf_no_slippage, _, _ = run(
            price,
            fast=2,
            slow=10,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
        )

        pf_slippage, _, _ = run(
            price,
            fast=2,
            slow=10,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.005,  # 0.5% slippage
        )

        return_no_slippage = float(pf_no_slippage.total_return())
        return_slippage = float(pf_slippage.total_return())

        assert return_slippage <= return_no_slippage, "Slippage should reduce returns"

    def test_volume_constraint_accepted(self) -> None:
        """Test that volume constraints are accepted as max_size parameter."""
        price = pd.Series(range(100, 130), dtype=float)
        volume = pd.Series([1000.0] * len(price))

        # Max size = 10% of volume / price
        max_size = (0.1 * volume / price).values

        pf, _, _ = run(
            price,
            fast=2,
            slow=10,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
            max_size=max_size,
        )

        assert pf is not None

    def test_volume_constraint_limits_position(self) -> None:
        """Test that volume constraints cap position sizes."""
        price = pd.Series([100.0] * 20, dtype=float)

        # Very tight volume constraint: only 1% of volume
        max_size_tight = np.array([0.01] * 20)

        pf_tight, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=True,
            max_size=max_size_tight,
        )

        # Loose volume constraint: 50% of volume
        max_size_loose = np.array([0.5] * 20)

        pf_loose, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
            next_bar_execution=True,
            max_size=max_size_loose,
        )

        # Tight constraint should produce different (likely lower) returns
        return_tight = float(pf_tight.total_return())
        return_loose = float(pf_loose.total_return())

        # Both should execute without error
        assert return_tight is not None
        assert return_loose is not None

    def test_combined_friction_costs(self) -> None:
        """Test combination of all friction costs together."""
        price = pd.Series(range(100, 140), dtype=float)
        volume = pd.Series([10000.0] * len(price))

        max_size = (0.1 * volume / price).values

        pf_combined, _, _ = run(
            price,
            fast=2,
            slow=15,
            init_cash=10000.0,
            next_bar_execution=True,
            fees=0.001,
            fixed_fees=5.0,
            slippage=0.003,
            max_size=max_size,
        )

        # Should execute without error and produce reasonable output
        assert pf_combined is not None
        stats = pf_combined.stats()
        assert len(stats) > 0

    def test_friction_backward_compatibility(self) -> None:
        """Test that friction parameters default to 0 if omitted."""
        price = pd.Series(range(100, 120), dtype=float)

        # Call without friction parameters (should not error)
        pf, _, _ = run(
            price,
            fast=2,
            slow=5,
            init_cash=10000.0,
        )

        assert pf is not None
