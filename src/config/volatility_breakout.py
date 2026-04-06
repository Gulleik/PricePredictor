"""Volatility breakout strategy configuration."""

# Single-run defaults
DONCHIAN_WINDOW = 20
ATR_WINDOW = 14
USE_ATR_FILTER = True
ATR_MIN_FRACTION = 0.005

# Hyperparameter ranges
DONCHIAN_WINDOWS = [15, 20, 30, 40]
ATR_WINDOWS = [10, 14, 20]
ATR_MIN_VALUES = [0.003, 0.005, 0.008]
