"""Tests for parser helpers in src.data."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data import (
    _cache_filename,
    _load_from_cache,
    _parse_datetime,
    _parse_timeframe,
    _save_to_cache,
    load_crypto_bars,
)

pytestmark = pytest.mark.unit


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
    expected = pd.DataFrame(
        {
            "open": [9.5, 10.5],
            "high": [10.5, 11.5],
            "low": [9.0, 10.0],
            "close": [10.0, 11.0],
            "volume": [100.0, 120.0],
        },
        index=index,
    )
    cache_path = data_module._get_cache_path(
        "BTC/USD",
        _parse_datetime("2024-01-01"),
        _parse_datetime("2024-01-03"),
        _parse_timeframe("1d"),
    )
    data_module._save_to_cache(cache_path, expected)

    class _UnexpectedClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)
            raise AssertionError("API client should not be created on cache hit")

    monkeypatch.setattr(data_module, "CryptoHistoricalDataClient", _UnexpectedClient)

    actual = load_crypto_bars("BTC/USD", "2024-01-01", "2024-01-03", timeframe="1d")

    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_load_crypto_bars_calls_api_on_cache_miss_and_saves(
    monkeypatch, tmp_path
) -> None:
    """Missing cache should call API once and write close prices to cache."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    response_df = pd.DataFrame(
        {
            "open": [19.5, 20.5, 21.5],
            "high": [20.5, 21.5, 22.5],
            "low": [19.0, 20.0, 21.0],
            "close": [20.0, 21.0, 22.0],
            "volume": [200.0, 210.0, 220.0],
        },
        index=index,
    )

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
    pd.testing.assert_frame_equal(actual, response_df, check_freq=False)

    expected_cache_path = data_module._get_cache_path(
        "BTC/USD",
        _parse_datetime("2024-01-01"),
        _parse_datetime("2024-01-04"),
        _parse_timeframe("1d"),
    )
    assert expected_cache_path.exists()


def test_load_from_cache_returns_none_when_parquet_engine_missing(
    monkeypatch, tmp_path
) -> None:
    """Missing parquet engine should be treated as cache miss."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    cache_path = tmp_path / "bars.parquet"
    cache_path.write_bytes(b"not used")

    def _raise_import_error(*args, **kwargs):
        _ = (args, kwargs)
        raise ImportError("Missing optional dependency 'pyarrow'")

    monkeypatch.setattr(data_module.pd, "read_parquet", _raise_import_error)

    assert _load_from_cache(cache_path) is None


def test_load_from_cache_reraises_unexpected_parquet_errors(
    monkeypatch, tmp_path
) -> None:
    """Unexpected parquet read errors should not be silently masked."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    cache_path = tmp_path / "bars.parquet"
    cache_path.write_bytes(b"invalid parquet payload")

    def _raise_value_error(*args, **kwargs):
        _ = (args, kwargs)
        raise ValueError("Corrupt parquet file")

    monkeypatch.setattr(data_module.pd, "read_parquet", _raise_value_error)

    with pytest.raises(ValueError, match="Corrupt parquet file"):
        _load_from_cache(cache_path)


def test_save_to_cache_skips_write_when_parquet_engine_missing(
    monkeypatch, tmp_path
) -> None:
    """Cache save should not write pickle payloads to .parquet paths."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)

    index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    ohlcv = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10.0, 11.0],
        },
        index=index,
    )
    cache_path = tmp_path / "bars.parquet"

    to_pickle_called = {"called": False}

    def _raise_import_error(*args, **kwargs):
        _ = (args, kwargs)
        raise ImportError("Missing optional dependency 'pyarrow'")

    def _track_to_pickle(*args, **kwargs):
        _ = (args, kwargs)
        to_pickle_called["called"] = True

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_import_error)
    monkeypatch.setattr(pd.DataFrame, "to_pickle", _track_to_pickle)

    _save_to_cache(cache_path, ohlcv)

    assert to_pickle_called["called"] is False
    assert not cache_path.exists()
