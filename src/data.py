"""Load crypto bars from Alpaca via alpaca-py CryptoHistoricalDataClient."""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dateparser import parse as parse_date
from dotenv import load_dotenv

from src.config import (
    AUDIT_SURVIVORSHIP,
    CACHE_DIR,
    CACHE_ENABLED,
    ENFORCE_UTC_INDEX,
    SURVIVORSHIP_FAIL_FAST,
    SURVIVORSHIP_MAX_GAP_FRACTION,
)
from src.validators import audit_survivorship_bias, validate_utc_index

load_dotenv()


def _cache_filename(
    symbol: str, start_dt: datetime, end_dt: datetime, timeframe: TimeFrame
) -> str:
    """Build a deterministic cache filename from request inputs."""
    key = (
        f"{symbol}|{start_dt.isoformat()}|{end_dt.isoformat()}|"
        f"{timeframe.amount}|{timeframe.unit.name}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return f"{safe_symbol}_{digest}.parquet"


def _get_cache_path(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    timeframe: TimeFrame,
) -> Path:
    """Return cache path under configured cache directory."""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / _cache_filename(symbol, start_dt, end_dt, timeframe)


def _load_from_cache(cache_path: Path) -> Union[pd.DataFrame, None]:
    """Load OHLCV bars from local cache if available."""
    if not CACHE_ENABLED or not cache_path.exists():
        return None

    cached_df: pd.DataFrame
    try:
        cached_df = pd.read_parquet(cache_path)
    except ImportError:
        cached_df = pd.read_pickle(cache_path)
    except Exception:
        # Support fallback caches written with pickle to a .parquet path.
        cached_df = pd.read_pickle(cache_path)

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(cached_df.columns):
        # Backward compatibility: older cache files may store close-only data.
        # Treat these files as cache misses so we can refresh to OHLCV format.
        if set(cached_df.columns) == {"close"}:
            print(
                f"Legacy cache format detected at {cache_path}; "
                "refreshing cache with OHLCV data."
            )
            return None
        raise ValueError(
            f"Cache file is missing required OHLCV columns. "
            f"Expected {required_cols}, got {set(cached_df.columns)}: {cache_path}"
        )
    return cached_df.sort_index()


def _save_to_cache(cache_path: Path, ohlcv: pd.DataFrame) -> None:
    """Persist OHLCV bars as parquet for fast repeat access."""
    if not CACHE_ENABLED:
        return
    sorted_ohlcv = ohlcv.sort_index()
    try:
        sorted_ohlcv.to_parquet(cache_path)
    except ImportError:
        sorted_ohlcv.to_pickle(cache_path)


def _parse_timeframe(timeframe: str) -> TimeFrame:
    """Convert string like '1d', '1h', '15m' to TimeFrame."""
    if not timeframe:
        return TimeFrame.Day
    unit = timeframe[-1].lower()
    try:
        amount = int(timeframe[:-1]) if len(timeframe) > 1 else 1
    except ValueError:
        return TimeFrame.Day
    if unit == "d":
        return TimeFrame(amount=max(amount, 1), unit=TimeFrameUnit.Day)
    if unit == "h":
        return TimeFrame(amount=max(1, min(amount, 23)), unit=TimeFrameUnit.Hour)
    if unit == "m":
        return TimeFrame(amount=max(1, min(amount, 59)), unit=TimeFrameUnit.Minute)
    return TimeFrame.Day


def _parse_datetime(value: Union[str, datetime]) -> datetime:
    """Parse flexible date string to timezone-aware datetime."""
    if isinstance(value, datetime):
        return value
    parsed = parse_date(
        value,
        settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True},
    )
    if parsed is None:
        raise ValueError(f"Could not parse datetime value: {value}")
    return parsed


def _run_data_integrity_checks(
    price: pd.Series,
    *,
    symbol: str,
    source_label: str,
    requested_start: datetime,
    requested_end: datetime,
    timeframe: str,
) -> None:
    """Run configured integrity checks without duplicating UTC validation."""
    if AUDIT_SURVIVORSHIP:
        report = audit_survivorship_bias(
            price,
            symbol=symbol,
            requested_start=requested_start,
            requested_end=requested_end,
            timeframe=timeframe,
            max_gap_fraction=SURVIVORSHIP_MAX_GAP_FRACTION,
        )
        for warning in report["warnings"]:
            print(f"Data integrity warning: {warning}")
        if SURVIVORSHIP_FAIL_FAST and report["warnings"]:
            raise ValueError(f"Survivorship audit failed for {symbol}")
        return

    if ENFORCE_UTC_INDEX:
        validate_utc_index(price, field_name=f"{symbol} {source_label} price")


def get_close_price_series(market_data: pd.DataFrame | pd.Series) -> pd.Series:
    """Return close price series from OHLCV data or pass through price series."""
    if hasattr(market_data, "columns") and "close" in market_data.columns:
        return market_data["close"]
    return market_data


def load_crypto_bars(
    symbol: str,
    start: str,
    end: str,
    *,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """
    Load historical crypto OHLCV bars from Alpaca.

    Args:
        symbol: Alpaca crypto symbol (e.g. "BTC/USD", "ETH/USD").
        start: Start date (e.g. "2024-01-01" or "1 year ago UTC").
        end: End date (e.g. "2024-12-31" or "now UTC").
        timeframe: Bar interval (e.g. "1d", "1h", "15m").

    Returns:
        OHLCV DataFrame with columns: open, high, low, close, volume.
    """
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    tf = _parse_timeframe(timeframe)
    cache_path = _get_cache_path(symbol, start_dt, end_dt, tf)

    cached_ohlcv = _load_from_cache(cache_path)
    if cached_ohlcv is not None:
        _run_data_integrity_checks(
            cached_ohlcv["close"],
            symbol=symbol,
            source_label="cached",
            requested_start=start_dt,
            requested_end=end_dt,
            timeframe=timeframe,
        )
        print(f"Using cache: {cache_path}")
        return cached_ohlcv

    if CACHE_ENABLED:
        print(f"Cache miss: fetching data from Alpaca for {symbol}")
    else:
        print(f"Cache disabled: fetching data from Alpaca for {symbol}")

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    client = CryptoHistoricalDataClient(
        api_key=key or "",
        secret_key=secret or "",
    )

    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_dt,
        end=end_dt,
    )
    bars = client.get_crypto_bars(request)

    if not bars.data:
        raise ValueError(f"No data returned for {symbol}")

    df = bars.df
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        df = df.loc[symbol]

    # Preserve full OHLCV; ensure required columns exist
    ohlcv = df[["open", "high", "low", "close", "volume"]].sort_index()

    _run_data_integrity_checks(
        ohlcv["close"],
        symbol=symbol,
        source_label="API",
        requested_start=start_dt,
        requested_end=end_dt,
        timeframe=timeframe,
    )

    _save_to_cache(cache_path, ohlcv)
    return ohlcv
