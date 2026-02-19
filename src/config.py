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
