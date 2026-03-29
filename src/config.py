"""Default configuration for backtesting."""

from pathlib import Path
from typing import Literal

# Default symbols for crypto backtesting
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD"]

# Entry-script symbols
BACKTEST_SYMBOL = DEFAULT_SYMBOLS[0]
HYPERPARAM_SYMBOL = DEFAULT_SYMBOLS[0]

# Default date range settings
DATE_REFRESH_CADENCE: Literal["month", "week", "day"] = "week"
DEFAULT_LOOKBACK_DAYS = 365

# Default timeframe for bars
DEFAULT_TIMEFRAME = "1d"

# Default initial capital for backtests
DEFAULT_INIT_CASH = 10_000.0

# Single-run SMA defaults
BACKTEST_FAST_WINDOW = 5
BACKTEST_SLOW_WINDOW = 15
BACKTEST_RENDER_CHART = True

# Hyperparameter search defaults
SCAN_OBJECTIVE = "sharpe_ratio"
FAST_WINDOWS = [5, 10, 15, 20, 25]
SLOW_WINDOWS = [30, 40, 50, 60]
HYPERPARAM_TOP_N = 5

# Data cache settings
CACHE_ENABLED = True
CACHE_DIR = Path("data")
