"""Broker model and execution abstraction for friction costs and volume constraints."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrokerModel:
    """Centralized configuration for broker friction costs and constraints."""

    commission_pct: float
    fixed_fee: float
    slippage_pct: float
    max_volume_participation: float

    def build_friction_kwargs(self, enable_friction: bool) -> dict[str, float]:
        """
        Build VectorBT Portfolio.from_signals() friction kwargs.

        Args:
            enable_friction: If False, returns zero friction costs.

        Returns:
            Dictionary with keys: fees, fixed_fees, slippage.
        """
        if not enable_friction:
            return {"fees": 0.0, "fixed_fees": 0.0, "slippage": 0.0}

        return {
            "fees": self.commission_pct,
            "fixed_fees": self.fixed_fee,
            "slippage": self.slippage_pct,
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
        max_size_array = (self.max_volume_participation * market_data["volume"]).values

        return max_size_array


class BrokerConfig(Protocol):
    """Protocol for config modules used to initialize BrokerModel."""

    BROKER_COMMISSION_PCT: float
    BROKER_FIXED_FEE: float
    BROKER_SLIPPAGE_PCT: float
    MAX_VOLUME_PARTICIPATION: float


def build_broker_model_from_config(config_module: BrokerConfig) -> BrokerModel:
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
