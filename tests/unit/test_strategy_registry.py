"""Unit tests for strategy registry and configuration wiring."""

import pytest

from src.strategies import STRATEGY_REGISTRY, get_strategy_module

pytestmark = pytest.mark.unit


def test_registry_contains_milestone6_strategies() -> None:
    """Registry should expose all configured strategy families."""
    expected = {
        "sma_crossover",
        "mean_reversion",
        "trend_following",
        "volatility_breakout",
        "orb",
    }
    assert expected.issubset(set(STRATEGY_REGISTRY))


def test_get_strategy_module_raises_for_unknown_name() -> None:
    """Resolver should fail fast for unknown strategy names."""
    with pytest.raises(ValueError, match="Unknown ACTIVE_STRATEGY"):
        get_strategy_module("not_a_strategy")
