"""Shared configuration used by all strategies and scripts."""

from pathlib import Path
from typing import Literal

StrategyName = Literal[
    "sma_crossover",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "orb",
    "ema_ribbon_scalp",
    "bb_rsi_mean_reversion",
    "momentum_scalp",
    "adaptive_momentum",
]

ACTIVE_STRATEGY: StrategyName = "adaptive_momentum"
ENABLED_STRATEGIES: tuple[StrategyName, ...] = (
    "sma_crossover",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "orb",
    "ema_ribbon_scalp",
    "bb_rsi_mean_reversion",
    "momentum_scalp",
    "adaptive_momentum",
)

# Default symbols for crypto backtesting
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]

# Entry-script symbols
BACKTEST_SYMBOL = DEFAULT_SYMBOLS[1]

# Default date range settings
DATE_REFRESH_CADENCE: Literal["month", "week", "day"] = "week"
DEFAULT_LOOKBACK_DAYS = 365  # 1 year of daily data

# Default timeframe for bars, aleternatives: "1h", "15m", "5m"
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
] = "total_return"
HYPERPARAM_TOP_N = 5

# Milestone 7: full-matrix hyperparameter batch search
HYPERPARAM_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
HYPERPARAM_TIMEFRAMES = ["1h", "15m", "5m"]
HYPERPARAM_STRATEGIES: list[StrategyName] = [
    "sma_crossover",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "orb",
    "ema_ribbon_scalp",
    "bb_rsi_mean_reversion",
]

BATCH_MODE: Literal["quick", "full"] = "quick"
BATCH_QUICK_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
BATCH_QUICK_TIMEFRAMES = ["1h"]
BATCH_QUICK_STRATEGIES: list[StrategyName] = [
    "bb_rsi_mean_reversion",
]

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
WFO_ENABLED = True
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
BROKER_COMMISSION_PCT = 0.00055  # 0.055% per-trade commission
BROKER_FIXED_FEE = 0.0  # $0 flat fee per trade
BROKER_SLIPPAGE_PCT = 0.002  # 0.2% slippage (bid-ask spread)
MAX_VOLUME_PARTICIPATION = 0.10  # 10% of bar volume max position size
KELLY_FACTOR = 0.5  # Conservative: 50% of theoretical Kelly
LEVERAGE = 50.0  # Leverage multiplier applied to position sizes (1.0 = no leverage)
SAVE_OPTUNA_TRIALS = False  # Set to True to persist per-run Optuna trial CSVs
