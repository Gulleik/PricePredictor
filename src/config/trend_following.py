"""Trend-following strategy configuration."""

# Single-run defaults
EMA_FAST_WINDOW = 12
EMA_SLOW_WINDOW = 26
ATR_WINDOW = 14
ATR_STOP_MULTIPLE = 2.0

# Hyperparameter ranges
EMA_FAST_WINDOWS = [8, 10, 12, 15]
EMA_SLOW_WINDOWS = [20, 26, 35, 50]
ATR_WINDOWS = [10, 14, 20]
ATR_STOP_MULTIPLES = [1.5, 2.0, 2.5, 3.0]
