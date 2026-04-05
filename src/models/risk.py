"""Kelly Criterion and position sizing utilities for risk management."""

import numpy as np
import pandas as pd


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_factor: float = 1.0,
) -> float:
    """
    Compute optimal Kelly Criterion fraction for position sizing.

    The Kelly formula is: f* = (p * b - q) / b
    where:
        p = win rate (0 to 1)
        b = average win / average loss (ratio)
        q = 1 - p (loss rate)

    For a profitable strategy: f* > 0
    For a negative expectancy: f* < 0 (don't trade)

    Args:
        win_rate: Fraction of winning trades (e.g., 0.55 for 55%).
        avg_win: Average profit per winning trade.
        avg_loss: Average loss per losing trade (positive value).
        kelly_factor: Scaling factor for conservative/aggressive sizing
                      (e.g., 0.25 for quarter-Kelly, 1.0 for full Kelly).

    Returns:
        Optimal fraction of bankroll to risk per trade, scaled by kelly_factor.
        Clipped to [0, 1] to prevent impossible values.

    Raises:
        ValueError: If inputs are invalid (negative values, avg_loss <= 0).
    """
    if not 0 <= win_rate <= 1:
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
    if avg_win < 0:
        raise ValueError(f"avg_win must be >= 0, got {avg_win}")
    if avg_loss <= 0:
        raise ValueError(f"avg_loss must be > 0, got {avg_loss}")
    if kelly_factor <= 0:
        raise ValueError(f"kelly_factor must be > 0, got {kelly_factor}")

    # Avoid division by zero
    if avg_win == 0:
        return 0.0

    # Kelly formula: f* = (p * b - q) / b
    # where b = avg_win / avg_loss
    b = avg_win / avg_loss
    loss_rate = 1.0 - win_rate
    kelly_fraction = (win_rate * b - loss_rate) / b

    # Apply conservative scaling factor
    scaled_fraction = kelly_fraction * kelly_factor

    # Clip to valid range [0, 1]
    return float(np.clip(scaled_fraction, 0.0, 1.0))


def generate_position_sizes(
    entries: pd.Series,
    price: pd.Series,
    kelly_fraction: float,
    init_cash: float,
) -> pd.Series:
    """
    Generate position size array for Kelly Criterion sizing.

    Produces a Series where:
        - Entry bars have position size = kelly_fraction * init_cash / entry_price
        - Non-entry bars have NaN (VectorBT interprets as "hold position")
        - Notional per entry is bounded by init_cash (no leverage)

    Args:
        entries: Boolean Series indicating entry signals (True = entry, False = hold).
        price: Close price Series (aligned with entries).
        kelly_fraction: Optimal Kelly fraction from compute_kelly_fraction().
        init_cash: Initial bankroll in currency units.

    Returns:
        Series with position size per bar; NaN for non-entry bars.

    Raises:
        ValueError: If kelly_fraction not in [0, 1] or init_cash <= 0.
    """
    if not 0 <= kelly_fraction <= 1:
        raise ValueError(f"kelly_fraction must be in [0, 1], got {kelly_fraction}")
    if init_cash <= 0:
        raise ValueError(f"init_cash must be > 0, got {init_cash}")

    if len(entries) != len(price):
        raise ValueError(
            f"entries and price must have same length, "
            f"got {len(entries)} and {len(price)}"
        )

    # Initialize with NaN (hold existing position)
    position_sizes = pd.Series(np.nan, index=price.index, dtype=float)

    # Set position size only at entry signals
    entry_mask = entries.astype(bool)
    if entry_mask.any():
        entry_prices = price[entry_mask]
        kelly_position = (kelly_fraction * init_cash) / entry_prices
        position_sizes[entry_mask] = kelly_position

    return position_sizes


def estimate_conservative_kelly(
    entries: pd.Series,
    exits: pd.Series,
    price: pd.Series,
) -> float:
    """
    Estimate a conservative Kelly fraction from signal-based trade metrics.

    Uses a simple heuristic: count signal pairs as trades, measure price moves
    between entry and exit, and compute win rate from positive moves.

    Args:
        entries: Boolean Series indicating entry signals.
        exits: Boolean Series indicating exit signals.
        price: Price Series (close prices).

    Returns:
        Conservative Kelly fraction; bounded to [0, 0.25] for safety.
        Returns 0.05 if insufficient data or no trades detected.
    """
    entry_indices = entries[entries].index.tolist()
    exit_indices = exits[exits].index.tolist()

    if not entry_indices or not exit_indices:
        return 0.05  # Default conservative estimate

    # Pair entries with following exits
    trades = []
    for entry_idx in entry_indices:
        following_exits = [e for e in exit_indices if e > entry_idx]
        if following_exits:
            exit_idx = following_exits[0]
            entry_price = price.loc[entry_idx]
            exit_price = price.loc[exit_idx]
            profit = exit_price - entry_price
            trades.append(profit)

    if not trades:
        return 0.05

    # Simple win rate: fraction of positive trades
    trades_arr = np.array(trades)
    wins = (trades_arr > 0).sum()
    losses = (trades_arr < 0).sum()

    if losses == 0:
        # No losses; use modest kelly
        return 0.10
    if wins == 0:
        # No wins; zero kelly
        return 0.0

    win_rate = wins / len(trades_arr)
    avg_win = trades_arr[trades_arr > 0].mean()
    avg_loss = np.abs(trades_arr[trades_arr < 0]).mean()

    kelly_frac = compute_kelly_fraction(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        kelly_factor=1.0,  # Raw Kelly before scaling
    )

    # Cap at 0.25 for extreme cases
    return float(np.clip(kelly_frac, 0.0, 0.25))
