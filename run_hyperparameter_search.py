"""Run hyperparameter search for SMA crossover and report best parameters."""

from src.config import (
    BROKER_COMMISSION_PCT,
    BROKER_FIXED_FEE,
    BROKER_SLIPPAGE_PCT,
    DEFAULT_INIT_CASH,
    DEFAULT_TIMEFRAME,
    ENABLE_FRICTION_MODEL,
    ENABLE_NEXT_BAR_EXECUTION,
    FAST_WINDOWS,
    HYPERPARAM_SYMBOL,
    HYPERPARAM_TOP_N,
    MAX_VOLUME_PARTICIPATION,
    SCAN_OBJECTIVE,
    SLOW_WINDOWS,
)
from src.data import get_close_price_series, load_crypto_bars
from src.date_range import get_default_date_range
from src.models.broker import BrokerModel
from src.strategies.sma_crossover import run_scan

OBJECTIVE_ACCESSORS = {
    "sharpe_ratio": "sharpe_ratio",
    "total_return": "total_return",
}


def main() -> None:
    symbol = HYPERPARAM_SYMBOL
    start, end = get_default_date_range()
    market_data = load_crypto_bars(
        symbol,
        start=start,
        end=end,
        timeframe=DEFAULT_TIMEFRAME,
    )
    price = get_close_price_series(market_data)

    # Initialize broker model for friction and volume constraints
    broker = BrokerModel(
        commission_pct=BROKER_COMMISSION_PCT,
        fixed_fee=BROKER_FIXED_FEE,
        slippage_pct=BROKER_SLIPPAGE_PCT,
        max_volume_participation=MAX_VOLUME_PARTICIPATION,
    )
    max_size_array = broker.compute_max_size_array(
        market_data,
        enable=ENABLE_FRICTION_MODEL,
    )
    friction_kwargs = broker.build_friction_kwargs(
        enable_friction=ENABLE_FRICTION_MODEL
    )

    pf = run_scan(
        price,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
        init_cash=DEFAULT_INIT_CASH,
        next_bar_execution=ENABLE_NEXT_BAR_EXECUTION,
        **friction_kwargs,
        max_size=max_size_array,
    )

    metric_method = OBJECTIVE_ACCESSORS.get(SCAN_OBJECTIVE)
    if metric_method is None:
        raise ValueError(
            f"Unknown SCAN_OBJECTIVE: {SCAN_OBJECTIVE}. "
            "Use 'sharpe_ratio' or 'total_return'."
        )
    metric_series = getattr(pf, metric_method)()

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
