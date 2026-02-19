"""Load crypto bars from Alpaca via alpaca-py CryptoHistoricalDataClient."""

import os
from datetime import datetime

import pandas as pd
from dateparser import parse as parse_date
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()


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
        return TimeFrame(amount=1, unit=TimeFrameUnit.Day)
    if unit == "h":
        return TimeFrame(amount=min(amount, 23), unit=TimeFrameUnit.Hour)
    if unit == "m":
        return TimeFrame(amount=min(amount, 59), unit=TimeFrameUnit.Minute)
    return TimeFrame.Day


def _parse_datetime(value: str) -> datetime:
    """Parse flexible date string to timezone-aware datetime."""
    if isinstance(value, datetime):
        return value
    return parse_date(value, settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True})


def load_crypto_bars(
    symbol: str,
    start: str,
    end: str,
    *,
    timeframe: str = "1d",
):
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
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    client = CryptoHistoricalDataClient(
        api_key=key or "",
        secret_key=secret or "",
    )

    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    tf = _parse_timeframe(timeframe)

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
    return price
