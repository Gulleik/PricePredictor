"""Data integrity validators for timezone, survivorship, and look-ahead checks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


class SurvivorshipAuditResult(dict[str, Any]):
    """Dictionary-like result payload for survivorship auditing."""


def _validate_utc_datetime(value: datetime, field_name: str) -> None:
    """Raise if a datetime is naive or not UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware and UTC")

    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC")


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Convert timeframe text like 1d/1h/15m into a timedelta."""
    if not timeframe:
        return timedelta(days=1)

    unit = timeframe[-1].lower()
    amount_text = timeframe[:-1] if len(timeframe) > 1 else "1"
    try:
        amount = max(int(amount_text), 1)
    except ValueError:
        return timedelta(days=1)

    if unit == "d":
        return timedelta(days=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    return timedelta(days=1)


def validate_utc_index(series: pd.Series, field_name: str = "price") -> None:
    """Raise if a series index is timezone-naive or not UTC."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{field_name} index must be a pandas DatetimeIndex")

    if series.index.tz is None:
        raise ValueError(f"{field_name} index must be timezone-aware and UTC")

    tz_name = str(series.index.tz)
    if tz_name.upper() != "UTC":
        raise ValueError(f"{field_name} index must be UTC, got {tz_name}")


def detect_data_gaps(
    series: pd.Series,
    *,
    timeframe: str = "1d",
) -> list[tuple[datetime, datetime]]:
    """Return (prev_ts, next_ts) pairs with larger-than-expected spacing."""
    if len(series.index) < 2:
        return []

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a pandas DatetimeIndex")

    expected = _timeframe_to_timedelta(timeframe)
    diffs = series.index.to_series().diff().dropna()
    gaps = diffs[diffs > expected]

    result: list[tuple[datetime, datetime]] = []
    for gap_ts in gaps.index:
        prev_ts = gap_ts - diffs.loc[gap_ts]
        result.append((prev_ts.to_pydatetime(), gap_ts.to_pydatetime()))
    return result


def audit_survivorship_bias(
    price: pd.Series,
    *,
    symbol: str,
    requested_start: datetime,
    requested_end: datetime,
    timeframe: str = "1d",
    max_gap_fraction: float = 0.05,
) -> SurvivorshipAuditResult:
    """Audit data continuity and range coverage for survivorship-bias risk signals."""
    validate_utc_index(price, field_name=f"{symbol} price")
    _validate_utc_datetime(requested_start, "requested_start")
    _validate_utc_datetime(requested_end, "requested_end")
    if requested_end < requested_start:
        raise ValueError(
            "requested_end must be greater than or equal to requested_start"
        )

    warnings: list[str] = []
    gaps = detect_data_gaps(price, timeframe=timeframe)

    if price.empty:
        warnings.append(f"{symbol}: no price data returned")
        return SurvivorshipAuditResult(
            {
                "gaps_count": 0,
                "gap_fraction": 1.0,
                "starts_late": True,
                "ends_early": True,
                "warnings": warnings,
            }
        )

    interval = _timeframe_to_timedelta(timeframe)
    total_window = max(requested_end - requested_start, interval)
    expected_bars = max(int(total_window / interval) + 1, 1)
    observed_bars = len(price)
    gap_fraction = max((expected_bars - observed_bars) / expected_bars, 0.0)

    starts_late = price.index[0].to_pydatetime() > (requested_start + interval)
    ends_early = price.index[-1].to_pydatetime() < (requested_end - interval)

    if starts_late:
        warnings.append(
            (
                f"{symbol}: data starts at {price.index[0].isoformat()}, "
                f"after requested start {requested_start.isoformat()}"
            )
        )
    if ends_early:
        warnings.append(
            (
                f"{symbol}: data ends at {price.index[-1].isoformat()}, "
                f"before requested end {requested_end.isoformat()}"
            )
        )
    if gaps:
        warnings.append(
            f"{symbol}: detected {len(gaps)} gap(s) larger than timeframe {timeframe}"
        )
    if gap_fraction > max_gap_fraction:
        warnings.append(
            (
                f"{symbol}: missing-bar fraction {gap_fraction:.2%} "
                f"exceeds threshold {max_gap_fraction:.2%}"
            )
        )

    return SurvivorshipAuditResult(
        {
            "gaps_count": len(gaps),
            "gap_fraction": gap_fraction,
            "starts_late": starts_late,
            "ends_early": ends_early,
            "warnings": warnings,
        }
    )


def validate_no_future_leakage(
    price: pd.Series,
    compute_signal: Callable[[pd.Series], pd.Series],
    *,
    perturb_from: int,
) -> None:
    """Raise if changing future prices alters earlier signal values."""
    if perturb_from <= 0 or perturb_from >= len(price):
        raise ValueError("perturb_from must be between 1 and len(price)-1")

    baseline = compute_signal(price).fillna(False)

    shocked = price.copy()
    tail_len = len(shocked) - perturb_from
    shocked.iloc[perturb_from:] = (
        pd.Series(
            range(tail_len, 0, -1), index=shocked.index[perturb_from:], dtype=float
        )
        * -1_000_000.0
    )
    candidate = compute_signal(shocked).fillna(False)

    if not baseline.iloc[:perturb_from].equals(candidate.iloc[:perturb_from]):
        raise ValueError("Signal depends on future data before perturbation point")
