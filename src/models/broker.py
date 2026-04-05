"""Broker model and execution abstraction for friction costs and volume constraints."""

from typing import Any

import numpy as np
import pandas as pd


class BrokerModel:
    """Centralized configuration for broker friction costs and constraints."""

    def __init__(
        self,
        commission_pct: float,
        fixed_fee: float,
        slippage_pct: float,
        max_volume_participation: float,
    ) -> None:
        """
        Initialize broker model with friction and volume parameters.

        Args:
            commission_pct: Percentage commission per trade (e.g., 0.001 for 0.1%).
            fixed_fee: Fixed fee in currency per trade (e.g., 1.0 for $1).
            slippage_pct: Percentage slippage (bid-ask spread, e.g., 0.002 for 0.2%).
            max_volume_participation: Max fraction of bar volume for a position (e.g., 0.1).
        """
        self.commission_pct = commission_pct
        self.fixed_fee = fixed_fee
        self.slippage_pct = slippage_pct
        self.max_volume_participation = max_volume_participation

    def build_friction_kwargs(self, enable_friction: bool) -> dict[str, Any]:
        """
        Build VectorBT Portfolio.from_signals() friction kwargs.

        Args:
            enable_friction: If False, returns zero friction costs.

        Returns:
            Dictionary with keys: fees, fixed_fees, slippage.
        """
        if enable_friction:
            return {
                "fees": self.commission_pct,
                "fixed_fees": self.fixed_fee,
                "slippage": self.slippage_pct,
            }
        else:
            return {
                "fees": 0.0,
                "fixed_fees": 0.0,
                "slippage": 0.0,
            }

    def compute_max_size_array(
        self, market_data: pd.DataFrame | pd.Series, enable: bool = True
    ) -> np.ndarray | None:
        """
        Compute max position sizes based on volume constraints.

        Calculates max_size as a fraction of each bar's volume, used to prevent
        unrealistic position sizing that exceeds available liquidity.

        Args:
            market_data: DataFrame with "volume" column, or Series (returns None).
            enable: If False, returns None (no volume constraint).

        Returns:
            Array of max position sizes, or None if constraints disabled/unavailable.
        """
        if not enable:
            return None

        if isinstance(market_data, pd.Series):
            return None

        if not hasattr(market_data, "columns") or "volume" not in market_data.columns:
            return None

        # Max size = max_volume_participation * bar_volume
        max_size_array = (
            self.max_volume_participation * market_data["volume"]
        ).values

        return max_size_array


def build_broker_model_from_config(config_module: Any) -> BrokerModel:
    """
    Factory function to build BrokerModel from config module.

    Extracts broker settings from config and creates a BrokerModel instance.

    Args:
        config_module: Module with broker config attributes (e.g., src.config).

    Returns:
        Configured BrokerModel instance.
    """
    return BrokerModel(
        commission_pct=config_module.BROKER_COMMISSION_PCT,
        fixed_fee=config_module.BROKER_FIXED_FEE,
        slippage_pct=config_module.BROKER_SLIPPAGE_PCT,
        max_volume_participation=config_module.MAX_VOLUME_PARTICIPATION,
    )
