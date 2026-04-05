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
DEFAULT_TIMEFRAME = "1D"

# Default initial capital for backtests
DEFAULT_INIT_CASH = 10_000.0

# Single-run SMA defaults
BACKTEST_FAST_WINDOW = 5
BACKTEST_SLOW_WINDOW = 15
BACKTEST_RENDER_CHART = False

# Hyperparameter search defaults
SCAN_OBJECTIVE = "sharpe_ratio"
FAST_WINDOWS = [5, 10, 15, 20, 25]
SLOW_WINDOWS = [30, 40, 50, 60]
HYPERPARAM_TOP_N = 5

# Data cache settings
CACHE_ENABLED = True
CACHE_DIR = Path("data")

# Data integrity settings (Milestone 2)
ENFORCE_UTC_INDEX = True
AUDIT_SURVIVORSHIP = True
SURVIVORSHIP_MAX_GAP_FRACTION = 0.05
SURVIVORSHIP_FAIL_FAST = False

# Execution fidelity & risk settings (Milestone 3)
ENABLE_FRICTION_MODEL = True           # Toggle friction costs on/off
BROKER_COMMISSION_PCT = 0.001          # 0.1% per-trade commission
BROKER_FIXED_FEE = 1.0                 # $1 flat fee per trade
BROKER_SLIPPAGE_PCT = 0.002            # 0.2% slippage (bid-ask spread)
MAX_VOLUME_PARTICIPATION = 0.10        # 10% of bar volume max position size
KELLY_FACTOR = 0.25                    # Conservative: 25% of theoretical Kelly
