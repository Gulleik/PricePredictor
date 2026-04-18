"""Configuration for momentum scalping strategy with multi-TP levels."""

# Single-run backtest defaults
BACKTEST_EMA_FAST = 8
BACKTEST_EMA_MEDIUM = 13
BACKTEST_EMA_SLOW = 21
BACKTEST_RSI_PERIOD = 7
BACKTEST_VOL_THRESHOLD = 1.5
BACKTEST_ATR_WINDOW = 14
BACKTEST_SL_ATR_MULTIPLE = 1.0
BACKTEST_TP1_MULTIPLE = 1.5
BACKTEST_TP2_MULTIPLE = 3.0
BACKTEST_TP3_TRAIL_MULTIPLE = 1.5
BACKTEST_ENTRY_COOLDOWN_BARS = 2

# TP allocations (fixed, not tuned)
TP1_ALLOCATION = 0.40  # 40% of position
TP2_ALLOCATION = 0.30  # 30% of position
TP3_ALLOCATION = 0.30  # 30% of position

# RSI levels (fixed)
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

# MACD fixed at standard settings (no tuning)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume analysis window (fixed)
VOLUME_WINDOW = 20

# Hyperparameter search ranges (Optuna)
EMA_FAST_WINDOWS = [5, 8, 10]
EMA_MEDIUM_WINDOWS = [13, 21]
EMA_SLOW_WINDOWS = [34, 55]
RSI_PERIOD_VALUES = [5, 7, 9]
VOL_THRESHOLD_VALUES = [1.2, 1.5, 2.0]
ATR_WINDOW_VALUES = [10, 14]
SL_ATR_MULTIPLE_VALUES = [0.75, 1.0, 1.5]
TP1_MULTIPLE_VALUES = [1.0, 1.5, 2.0]
TP2_MULTIPLE_VALUES = [2.5, 3.0, 3.5]
TP3_TRAIL_MULTIPLE_VALUES = [1.0, 1.5, 2.0]
ENTRY_COOLDOWN_BARS_VALUES = [0, 1, 2, 3]

# Constraint: ema_fast < ema_medium < ema_slow (enforced in strategy optimization)
