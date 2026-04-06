"""Shared configuration used by all strategies and scripts."""

from pathlib import Path
from typing import Literal

# Default symbols for crypto backtesting
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD"]

# Entry-script symbols
BACKTEST_SYMBOL = DEFAULT_SYMBOLS[0]
HYPERPARAM_SYMBOL = DEFAULT_SYMBOLS[0]

# Default date range settings
DATE_REFRESH_CADENCE: Literal["month", "week", "day"] = "week"
DEFAULT_LOOKBACK_DAYS = 365  # 1 year of daily data

# Default timeframe for bars
DEFAULT_TIMEFRAME = "1h"

# Default initial capital for backtests
DEFAULT_INIT_CASH = 10_000.0

# Backtest rendering
BACKTEST_RENDER_CHART = False

# Hyperparameter search defaults
SCAN_OBJECTIVE: Literal[
    "sharpe_ratio",
    "total_return",
    "sortino_ratio",
    "calmar_ratio",
] = "sharpe_ratio"
HYPERPARAM_TOP_N = 5
HYPERPARAM_SEARCH_STRATEGY: Literal[
    "sma_crossover",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "orb",
] = "orb"  # Strategy to optimize in hyperparameter search

# Milestone 5: systematic search and metrics defaults
OPTUNA_SAMPLER: Literal["tpe", "random"] = "tpe"
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT_SECONDS = 600
OPTUNA_SEED = 42
# Study name prefix; strategy name is appended at runtime.
OPTUNA_STUDY_NAME = "strategy_search"
OPTUNA_STARTUP_TRIALS = 10
RESULTS_DIR = Path("results")

# Milestone 4: validation and sensitivity defaults
WFO_ENABLED = False
WFO_MODE: Literal["auto", "preset", "manual"] = "auto"
WFO_PRESET: Literal["quick", "balanced", "robust"] = "balanced"
WFO_IS_WINDOW_BARS = 252
WFO_OOS_FRACTION = 0.25
WFO_STEP_BARS = 63
WFO_OOS_METRIC: Literal["sharpe_ratio", "total_return"] = "sharpe_ratio"

REGIME_LOOKBACK_FAST = 50
REGIME_LOOKBACK_SLOW = 200
REGIME_SIDEWAYS_BAND = 0.01

SENSITIVITY_HEATMAP_OUTPUT_PATH = Path("data/sensitivity_heatmap.png")
SENSITIVITY_MATRIX_OUTPUT_PATH = Path("data/sensitivity_matrix.csv")

# Data cache settings
CACHE_ENABLED = True
CACHE_DIR = Path("data")

# Data integrity settings (Milestone 2)
ENFORCE_UTC_INDEX = True
AUDIT_SURVIVORSHIP = True
SURVIVORSHIP_MAX_GAP_FRACTION = 0.05
SURVIVORSHIP_FAIL_FAST = False

# Execution fidelity & risk settings (Milestone 3)
ENABLE_NEXT_BAR_EXECUTION = True  # Shift signals to next bar (lookahead safety)
ENABLE_FRICTION_MODEL = True  # Toggle friction costs on/off
BROKER_COMMISSION_PCT = 0.001  # 0.1% per-trade commission
BROKER_FIXED_FEE = 1.0  # $1 flat fee per trade
BROKER_SLIPPAGE_PCT = 0.002  # 0.2% slippage (bid-ask spread)
MAX_VOLUME_PARTICIPATION = 0.10  # 10% of bar volume max position size
KELLY_FACTOR = 0.25  # Conservative: 25% of theoretical Kelly
