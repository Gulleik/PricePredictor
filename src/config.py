"""Default configuration for backtesting."""

# Default symbols for crypto backtesting
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD"]

# Default date range (use relative dates for fresh data)
DEFAULT_START = "1 year ago UTC"
DEFAULT_END = "now UTC"

# Default timeframe for bars
DEFAULT_TIMEFRAME = "1d"

# Default initial capital for backtests
DEFAULT_INIT_CASH = 10_000.0

# Hyperparameter search defaults
SCAN_OBJECTIVE = "sharpe_ratio"
FAST_WINDOWS = [5, 10, 15, 20, 25]
SLOW_WINDOWS = [30, 40, 50, 60]
