"""Strategy registry and dispatch helpers."""

from collections.abc import Mapping
from typing import Any

from src.config import StrategyName
from src.strategies import (
    mean_reversion,
    momentum_scalp,
    orb,
    sma_crossover,
    trend_following,
    vectorbt_scalping,
)
from src.strategies import volatility_breakout as vol_breakout

STRATEGY_REGISTRY: Mapping[StrategyName, Any] = {
    "sma_crossover": sma_crossover,
    "mean_reversion": mean_reversion,
    "trend_following": trend_following,
    "volatility_breakout": vol_breakout,
    "orb": orb,
    "ema_ribbon_scalp": vectorbt_scalping,
    "bb_rsi_mean_reversion": vectorbt_scalping,
    "momentum_scalp": momentum_scalp,
}


def get_strategy_module(strategy_name: StrategyName) -> Any:
    """Resolve configured strategy name to a strategy module."""
    if strategy_name not in STRATEGY_REGISTRY:
        supported = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(
            f"Unknown ACTIVE_STRATEGY: {strategy_name}. Supported: {supported}."
        )
    return STRATEGY_REGISTRY[strategy_name]
