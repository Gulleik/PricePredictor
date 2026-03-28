"""Default configuration for backtesting."""

# Default symbols for crypto backtesting
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD"]

# Entry-script symbols
BACKTEST_SYMBOL = DEFAULT_SYMBOLS[0]
HYPERPARAM_SYMBOL = DEFAULT_SYMBOLS[0]

# Default date range (use relative dates for fresh data)
DEFAULT_START = "1 year ago UTC"
DEFAULT_END = "now UTC"

# Default timeframe for bars
DEFAULT_TIMEFRAME = "1d"

# Default initial capital for backtests
DEFAULT_INIT_CASH = 10_000.0

# Single-run SMA defaults
BACKTEST_FAST_WINDOW = 5
BACKTEST_SLOW_WINDOW = 15

# Hyperparameter search defaults
SCAN_OBJECTIVE = "sharpe_ratio"
FAST_WINDOWS = [5, 10, 15, 20, 25]
SLOW_WINDOWS = [30, 40, 50, 60]
HYPERPARAM_TOP_N = 5
