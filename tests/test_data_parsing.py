"""Tests for parser helpers in src.data."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data import (
    _cache_filename,
    _parse_datetime,
    _parse_timeframe,
    load_crypto_bars,
)


def test_parse_datetime_returns_input_datetime() -> None:
    """A datetime input should be returned unchanged."""
    value = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _parse_datetime(value) is value


def test_parse_datetime_raises_for_invalid_string() -> None:
    """Invalid date text should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Could not parse datetime value"):
        _parse_datetime("definitely not a date")


def test_cache_filename_is_deterministic() -> None:
    """Same cache inputs should always produce the same filename."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    timeframe = _parse_timeframe("1d")

    first = _cache_filename("BTC/USD", start, end, timeframe)
    second = _cache_filename("BTC/USD", start, end, timeframe)

    assert first == second
    assert first.endswith(".parquet")


def test_load_crypto_bars_uses_cache_when_available(monkeypatch, tmp_path) -> None:
    """Existing cache should bypass API calls and return cached prices."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    expected = pd.Series([10.0, 11.0], index=index, name="close")
    cache_path = data_module._get_cache_path(
        "BTC/USD",
        _parse_datetime("2024-01-01"),
        _parse_datetime("2024-01-03"),
        _parse_timeframe("1d"),
    )
    expected.to_frame(name="close").to_parquet(cache_path)

    class _UnexpectedClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)
            raise AssertionError("API client should not be created on cache hit")

    monkeypatch.setattr(data_module, "CryptoHistoricalDataClient", _UnexpectedClient)

    actual = load_crypto_bars("BTC/USD", "2024-01-01", "2024-01-03", timeframe="1d")

    pd.testing.assert_series_equal(actual, expected, check_freq=False)


def test_load_crypto_bars_calls_api_on_cache_miss_and_saves(
    monkeypatch, tmp_path
) -> None:
    """Missing cache should call API once and write close prices to cache."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    response_df = pd.DataFrame({"close": [20.0, 21.0, 22.0]}, index=index)

    class _BarsResponse:
        def __init__(self, df: pd.DataFrame) -> None:
            self.data = {"BTC/USD": True}
            self.df = df

    class _Client:
        called = 0

        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def get_crypto_bars(self, request):
            _ = request
            _Client.called += 1
            return _BarsResponse(response_df)

    monkeypatch.setattr(data_module, "CryptoHistoricalDataClient", _Client)

    actual = load_crypto_bars("BTC/USD", "2024-01-01", "2024-01-04", timeframe="1d")

    assert _Client.called == 1
    pd.testing.assert_series_equal(actual, response_df["close"])

    expected_cache_path = data_module._get_cache_path(
        "BTC/USD",
        _parse_datetime("2024-01-01"),
        _parse_datetime("2024-01-04"),
        _parse_timeframe("1d"),
    )
    assert expected_cache_path.exists()
