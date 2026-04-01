"""Tests for UTC timezone integrity across the data pipeline."""

from datetime import timezone

import pandas as pd
import pytest

from src.validators import validate_utc_index


def test_validate_utc_index_accepts_utc_series() -> None:
    """UTC-indexed price series should pass validation."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0], index=index)

    validate_utc_index(series)


def test_validate_utc_index_rejects_naive_index() -> None:
    """Timezone-naive index should fail strict UTC validation."""
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([1.0, 2.0, 3.0], index=index)

    with pytest.raises(ValueError, match="timezone-aware"):
        validate_utc_index(series)


def test_validate_utc_index_rejects_non_utc_index() -> None:
    """Non-UTC timezone index should fail strict UTC validation."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="Europe/Oslo")
    series = pd.Series([1.0, 2.0, 3.0], index=index)

    with pytest.raises(ValueError, match="must be UTC"):
        validate_utc_index(series)


def test_load_crypto_bars_rejects_naive_cache_index(monkeypatch, tmp_path) -> None:
    """Cache hit should fail when cached index is not UTC-aware."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_module, "ENFORCE_UTC_INDEX", True)
    monkeypatch.setattr(data_module, "AUDIT_SURVIVORSHIP", False)

    start_dt = data_module._parse_datetime("2024-01-01")
    end_dt = data_module._parse_datetime("2024-01-03")
    cache_path = data_module._get_cache_path(
        "BTC/USD",
        start_dt,
        end_dt,
        data_module._parse_timeframe("1d"),
    )

    naive_index = pd.date_range("2024-01-01", periods=2, freq="D")
    pd.Series([100.0, 101.0], index=naive_index, name="close").to_frame(
        name="close"
    ).to_parquet(cache_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        data_module.load_crypto_bars("BTC/USD", "2024-01-01", "2024-01-03")


def test_parse_datetime_returns_aware_string_parse() -> None:
    """String date parsing should return timezone-aware UTC datetime."""
    import src.data as data_module

    parsed = data_module._parse_datetime("2024-01-01")
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def test_load_crypto_bars_cache_path_skips_duplicate_validation(
    monkeypatch, tmp_path
) -> None:
    """When auditing is enabled, cache path should not call direct UTC validation."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "CACHE_ENABLED", True)
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_module, "ENFORCE_UTC_INDEX", True)
    monkeypatch.setattr(data_module, "AUDIT_SURVIVORSHIP", True)

    start_dt = data_module._parse_datetime("2024-01-01")
    end_dt = data_module._parse_datetime("2024-01-03")
    cache_path = data_module._get_cache_path(
        "BTC/USD",
        start_dt,
        end_dt,
        data_module._parse_timeframe("1d"),
    )

    utc_index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    pd.Series([100.0, 101.0], index=utc_index, name="close").to_frame(
        name="close"
    ).to_parquet(cache_path)

    direct_validate_calls = {"count": 0}
    audit_calls = {"count": 0}

    def _count_direct_validate(series: pd.Series, field_name: str = "price") -> None:
        direct_validate_calls["count"] += 1

    def _fake_audit(*args, **kwargs):
        audit_calls["count"] += 1
        return {"warnings": []}

    monkeypatch.setattr(data_module, "validate_utc_index", _count_direct_validate)
    monkeypatch.setattr(data_module, "audit_survivorship_bias", _fake_audit)

    data_module.load_crypto_bars("BTC/USD", "2024-01-01", "2024-01-03")

    assert direct_validate_calls["count"] == 0
    assert audit_calls["count"] == 1
