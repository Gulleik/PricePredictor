"""Configuration for adaptive momentum strategy with scored signals."""

# Single-run backtest defaults
BACKTEST_EMA_FAST = 8
BACKTEST_EMA_MEDIUM = 13
BACKTEST_EMA_SLOW = 21
BACKTEST_RSI_PERIOD = 7
BACKTEST_VOL_THRESHOLD = 1.5
BACKTEST_SIGNAL_THRESHOLD = 0.5
BACKTEST_ADX_PERIOD = 14
BACKTEST_ADX_THRESHOLD = 20.0
BACKTEST_ATR_PERCENTILE_MIN = 30.0
BACKTEST_ATR_PERCENTILE_WINDOW = 100
BACKTEST_ATR_WINDOW = 14
BACKTEST_SL_ATR_MULTIPLE = 1.5
BACKTEST_TP1_MULTIPLE = 2.0
BACKTEST_TP2_TRAIL_MULTIPLE = 1.5
BACKTEST_ENTRY_COOLDOWN_BARS = 2

# TP allocations (fixed, not tuned)
TP1_ALLOCATION = 0.60  # 60% of position
TP2_ALLOCATION = 0.40  # 40% of position

# RSI levels (fixed)
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

# MACD defaults
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume analysis window (fixed)
VOLUME_WINDOW = 20

# Hyperparameter search ranges (Optuna)
# Narrowed based on 2026-04-19 optimizer findings:
# - Low signal/ADX thresholds dominate (fewer entry gates = more trades)
# - Tight SL (1.0) and high cooldown (3) consistently best
# - Small indicator periods (fast=5, adx=10, atr=10) preferred
EMA_FAST_WINDOWS = [5, 8]
EMA_MEDIUM_WINDOWS = [13, 21]
EMA_SLOW_WINDOWS = [34, 55]
RSI_PERIOD_VALUES = [7, 9]
VOL_THRESHOLD_VALUES = [1.2, 1.5, 2.0]
SIGNAL_THRESHOLD_VALUES = [0.3, 0.4, 0.5]
ADX_PERIOD_VALUES = [10, 14]
ADX_THRESHOLD_VALUES = [15.0, 20.0]
ATR_PERCENTILE_MIN_VALUES = [20.0, 30.0]
ATR_WINDOW_VALUES = [10, 14]
SL_ATR_MULTIPLE_VALUES = [0.75, 1.0, 1.5]
TP1_MULTIPLE_VALUES = [1.5, 2.0, 2.5]
TP2_TRAIL_MULTIPLE_VALUES = [1.0, 1.5]
ENTRY_COOLDOWN_BARS_VALUES = [2, 3, 4]
MACD_FAST_VALUES = [8, 12]
MACD_SLOW_VALUES = [21, 26]
MACD_SIGNAL_VALUES = [7, 9]

# Constraint: ema_fast < ema_medium < ema_slow (enforced in strategy optimization)
# Constraint: macd_fast < macd_slow (enforced in strategy optimization)
