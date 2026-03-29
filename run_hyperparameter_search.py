"""Run hyperparameter search for SMA crossover and report best parameters."""

from src.config import (
    DEFAULT_END,
    DEFAULT_INIT_CASH,
    DEFAULT_START,
    DEFAULT_TIMEFRAME,
    FAST_WINDOWS,
    HYPERPARAM_SYMBOL,
    HYPERPARAM_TOP_N,
    SCAN_OBJECTIVE,
    SLOW_WINDOWS,
)
from src.data import load_crypto_bars
from src.strategies.sma_crossover import run_scan


def main() -> None:
    symbol = HYPERPARAM_SYMBOL
    price = load_crypto_bars(
        symbol,
        start=DEFAULT_START,
        end=DEFAULT_END,
        timeframe=DEFAULT_TIMEFRAME,
    )

    pf = run_scan(
        price,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
        init_cash=DEFAULT_INIT_CASH,
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
