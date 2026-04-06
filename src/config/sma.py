"""SMA crossover strategy configuration."""

# Single-run SMA defaults
BACKTEST_FAST_WINDOW = 10
BACKTEST_SLOW_WINDOW = 25

# Hyperparameter search defaults for SMA
FAST_WINDOWS = [5, 10, 15, 20, 25]
SLOW_WINDOWS = [30, 40, 50, 60]
