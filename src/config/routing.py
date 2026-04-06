"""Strategy routing configuration."""

from typing import Literal

StrategyName = Literal[
    "sma_crossover",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "orb",
]

ACTIVE_STRATEGY: StrategyName = "mean_reversion"
ENABLED_STRATEGIES: tuple[StrategyName, ...] = (
    "sma_crossover",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "orb",
)
