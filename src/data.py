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

from src.config import CACHE_DIR, CACHE_ENABLED

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


def _load_from_cache(cache_path: Path) -> Union[pd.Series, None]:
    """Load a close-price series from local cache if available."""
    if not CACHE_ENABLED or not cache_path.exists():
        return None

    cached_df = pd.read_parquet(cache_path)
    if "close" not in cached_df.columns:
        raise ValueError(f"Cache file is missing required 'close' column: {cache_path}")
    return cached_df["close"].sort_index()


def _save_to_cache(cache_path: Path, price: pd.Series) -> None:
    """Persist close-price series as parquet for fast repeat access."""
    if not CACHE_ENABLED:
        return
    price.sort_index().to_frame(name="close").to_parquet(cache_path)


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


def load_crypto_bars(
    symbol: str,
    start: str,
    end: str,
    *,
    timeframe: str = "1d",
) -> pd.Series:
    """
    Load historical crypto OHLCV bars from Alpaca.

    Args:
        symbol: Alpaca crypto symbol (e.g. "BTC/USD", "ETH/USD").
        start: Start date (e.g. "2024-01-01" or "1 year ago UTC").
        end: End date (e.g. "2024-12-31" or "now UTC").
        timeframe: Bar interval (e.g. "1d", "1h", "15m").

    Returns:
        Price series (Close) as pandas Series for use with VectorBT.
    """
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    tf = _parse_timeframe(timeframe)
    cache_path = _get_cache_path(symbol, start_dt, end_dt, tf)

    cached_price = _load_from_cache(cache_path)
    if cached_price is not None:
        print(f"Cache hit: {cache_path}")
        return cached_price

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
    price = df["close"].sort_index()
    _save_to_cache(cache_path, price)
    return price
