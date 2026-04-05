"""Run hyperparameter search for SMA crossover and report best parameters."""

from src.config import (
    DEFAULT_INIT_CASH,
    DEFAULT_TIMEFRAME,
    ENABLE_NEXT_BAR_EXECUTION,
    ENABLE_FRICTION_MODEL,
    FAST_WINDOWS,
    HYPERPARAM_SYMBOL,
    HYPERPARAM_TOP_N,
    SCAN_OBJECTIVE,
    SLOW_WINDOWS,
)
from src.data import load_crypto_bars
from src.date_range import get_default_date_range
from src.models.broker import build_broker_model_from_config
from src.strategies.sma_crossover import run_scan
import src.config


def main() -> None:
    symbol = HYPERPARAM_SYMBOL
    start, end = get_default_date_range()
    market_data = load_crypto_bars(
        symbol,
        start=start,
        end=end,
        timeframe=DEFAULT_TIMEFRAME,
    )
    price = (
        market_data["close"]
        if hasattr(market_data, "columns") and "close" in market_data.columns
        else market_data
    )

    # Initialize broker model for friction and volume constraints
    broker = build_broker_model_from_config(src.config)
    max_size_array = broker.compute_max_size_array(market_data, enable=ENABLE_FRICTION_MODEL)
    friction_kwargs = broker.build_friction_kwargs(enable_friction=ENABLE_FRICTION_MODEL)

    pf = run_scan(
        price,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        **friction_kwargs,
        max_size=max_size_array,
    )

    if SCAN_OBJECTIVE == "sharpe_ratio":
        metric_series = pf.sharpe_ratio()
    elif SCAN_OBJECTIVE == "total_return":
        metric_series = pf.total_return()
    else:
        raise ValueError(
            f"Unknown SCAN_OBJECTIVE: {SCAN_OBJECTIVE}. "
            "Use 'sharpe_ratio' or 'total_return'."
        )

    best_col = metric_series.idxmax()
    best_fast, best_slow = best_col
    best_value = metric_series.max()

    print(f"Hyperparameter search ({SCAN_OBJECTIVE})")
    print(
        "  Best: "
        f"fast={best_fast}, slow={best_slow} -> "
        f"{SCAN_OBJECTIVE}={best_value:.4f}"
    )
    print()
    print("Best parameter stats:")
    print(pf[best_col].stats())
    print()

    top_n = HYPERPARAM_TOP_N
    top_cols = metric_series.nlargest(top_n)
    print(f"Top {top_n} combinations:")
    for (fast, slow), val in top_cols.items():
        print(f"  fast={fast}, slow={slow}: {SCAN_OBJECTIVE}={val:.4f}")


if __name__ == "__main__":
    main()
