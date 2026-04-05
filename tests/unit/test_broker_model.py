"""Tests for broker model abstraction."""

import numpy as np
import pandas as pd
import pytest

from src.models.broker import BrokerModel, build_broker_model_from_config

pytestmark = pytest.mark.unit


class TestBrokerModel:
    """Test BrokerModel friction and volume constraint logic."""

    def test_broker_model_initialization(self) -> None:
        """Test BrokerModel stores parameters correctly."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        assert broker.commission_pct == 0.001
        assert broker.fixed_fee == 1.0
        assert broker.slippage_pct == 0.002
        assert broker.max_volume_participation == 0.1

    def test_build_friction_kwargs_enabled(self) -> None:
        """Test friction kwargs when enabled."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        kwargs = broker.build_friction_kwargs(enable_friction=True)

        assert kwargs["fees"] == 0.001
        assert kwargs["fixed_fees"] == 1.0
        assert kwargs["slippage"] == 0.002

    def test_build_friction_kwargs_disabled(self) -> None:
        """Test friction kwargs when disabled (all zero)."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        kwargs = broker.build_friction_kwargs(enable_friction=False)

        assert kwargs["fees"] == 0.0
        assert kwargs["fixed_fees"] == 0.0
        assert kwargs["slippage"] == 0.0

    def test_compute_max_size_array_enabled(self) -> None:
        """Test max size computation when enabled."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        market_data = pd.DataFrame({
            "volume": [1000.0, 2000.0, 1500.0],
            "close": [100.0, 101.0, 102.0],
        })

        max_size = broker.compute_max_size_array(market_data, enable=True)

        expected = np.array([100.0, 200.0, 150.0])  # 0.1 * volume
        np.testing.assert_array_almost_equal(max_size, expected)

    def test_compute_max_size_array_disabled(self) -> None:
        """Test max size returns None when disabled."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        market_data = pd.DataFrame({
            "volume": [1000.0, 2000.0],
            "close": [100.0, 101.0],
        })

        max_size = broker.compute_max_size_array(market_data, enable=False)

        assert max_size is None

    def test_compute_max_size_array_no_volume_column(self) -> None:
        """Test max size returns None when volume column is missing."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        market_data = pd.DataFrame({
            "close": [100.0, 101.0],
        })

        max_size = broker.compute_max_size_array(market_data, enable=True)

        assert max_size is None

    def test_compute_max_size_array_series_input(self) -> None:
        """Test max size returns None when input is Series."""
        broker = BrokerModel(
            commission_pct=0.001,
            fixed_fee=1.0,
            slippage_pct=0.002,
            max_volume_participation=0.1,
        )

        price_series = pd.Series([100.0, 101.0, 102.0])

        max_size = broker.compute_max_size_array(price_series, enable=True)

        assert max_size is None

    def test_build_broker_model_from_config(self) -> None:
        """Test factory function creates BrokerModel from config."""
        from types import SimpleNamespace

        config = SimpleNamespace(
            BROKER_COMMISSION_PCT=0.002,
            BROKER_FIXED_FEE=2.0,
            BROKER_SLIPPAGE_PCT=0.003,
            MAX_VOLUME_PARTICIPATION=0.15,
        )

        broker = build_broker_model_from_config(config)

        assert broker.commission_pct == 0.002
        assert broker.fixed_fee == 2.0
        assert broker.slippage_pct == 0.003
        assert broker.max_volume_participation == 0.15
