"""Date range helpers for cache-stable default windows."""

from datetime import datetime, timedelta, timezone

from src.config import DATE_REFRESH_CADENCE, DEFAULT_LOOKBACK_DAYS


def _floor_utc(dt: datetime, cadence: str) -> datetime:
    """Round a UTC datetime down to the configured cadence boundary."""
    if cadence == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if cadence == "week":
        week_start = dt - timedelta(days=dt.weekday())
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    if cadence == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    raise ValueError(
        f"Unknown DATE_REFRESH_CADENCE: {cadence}. Use 'month', 'week', or 'day'."
    )


def get_default_date_range(now_utc: datetime | None = None) -> tuple[str, str]:
    """Return default start/end ISO timestamps based on cadence and lookback."""
    now = now_utc or datetime.now(timezone.utc)
    end_dt = _floor_utc(now, DATE_REFRESH_CADENCE)
    start_dt = end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start_dt.isoformat(), end_dt.isoformat()
