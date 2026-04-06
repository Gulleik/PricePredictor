"""Shared helpers for strategy execution and signal handling."""

from typing import Any

import numpy as np
import pandas as pd


def sanitize_max_size(
    max_size: Any,
    price_index: pd.Index,
) -> tuple[Any | None, pd.Series | None]:
    """Return max_size safe for VectorBT and validity mask for tradable bars."""
    if max_size is None:
        return None, None

    arr = np.asarray(max_size, dtype=float)
    if arr.ndim == 0:
        value = float(arr)
        if not np.isfinite(value) or value <= 0:
            mask = pd.Series(False, index=price_index)
            return None, mask
        return value, None

    if arr.ndim != 1:
        return max_size, None

    if len(arr) != len(price_index):
        raise ValueError(
            "max_size length must match price length when provided as 1D array"
        )

    valid = np.isfinite(arr) & (arr > 0)
    valid_mask = pd.Series(valid, index=price_index)
    safe_arr = np.where(valid, arr, np.finfo(float).tiny)
    return safe_arr, valid_mask


def apply_valid_mask(signals: Any, valid_mask: pd.Series) -> Any:
    """Apply row-wise tradability mask to Series/DataFrame boolean signals."""
    if isinstance(signals, pd.Series):
        return signals & valid_mask
    if isinstance(signals, pd.DataFrame):
        mask_2d = np.broadcast_to(valid_mask.to_numpy().reshape(-1, 1), signals.shape)
        return signals.where(mask_2d, False)
    return signals


def apply_next_bar_execution(entries: Any, exits: Any) -> tuple[Any, Any]:
    """Shift signal execution forward by one bar to avoid same-bar fills."""
    shifted_entries = entries.vbt.fshift(1).fillna(False).astype(bool)
    shifted_exits = exits.vbt.fshift(1).fillna(False).astype(bool)
    return shifted_entries, shifted_exits


def extract_ohlc(
    market_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Extract open/high/low/close series from market data frame."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(market_data.columns):
        raise ValueError(
            f"market_data missing required OHLC columns. Expected {required}, "
            f"got {set(market_data.columns)}"
        )

    return (
        market_data["open"],
        market_data["high"],
        market_data["low"],
        market_data["close"],
    )
