"""Tests for survivorship-bias auditing helpers."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.validators import audit_survivorship_bias, detect_data_gaps


def test_detect_data_gaps_finds_missing_interval() -> None:
    """A missing timestamp interval should be reported as a gap."""
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"], utc=True)
    series = pd.Series([100.0, 101.0, 104.0], index=index)

    gaps = detect_data_gaps(series, timeframe="1d")

    assert len(gaps) == 1
    assert gaps[0][0] == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert gaps[0][1] == datetime(2024, 1, 5, tzinfo=timezone.utc)


def test_audit_survivorship_flags_early_end() -> None:
    """Audit should flag data that ends well before the requested window end."""
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    series = pd.Series([10.0, 11.0, 12.0, 13.0], index=index)

    report = audit_survivorship_bias(
        series,
        symbol="BTC/USD",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        timeframe="1d",
        max_gap_fraction=0.05,
    )

    assert report["ends_early"] is True
    assert any("before requested end" in warning for warning in report["warnings"])


def test_audit_survivorship_gap_fraction_threshold_warning() -> None:
    """Audit should warn when missing bar fraction exceeds threshold."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    series = pd.Series([10.0, 11.0, 12.0], index=index)

    report = audit_survivorship_bias(
        series,
        symbol="BTC/USD",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        timeframe="1d",
        max_gap_fraction=0.05,
    )

    assert report["gap_fraction"] > 0.05
    assert any("exceeds threshold" in warning for warning in report["warnings"])


def test_audit_survivorship_rejects_naive_requested_start() -> None:
    """Audit should raise when requested_start is timezone-naive."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    series = pd.Series([10.0, 11.0, 12.0], index=index)

    with pytest.raises(ValueError, match="requested_start"):
        audit_survivorship_bias(
            series,
            symbol="BTC/USD",
            requested_start=datetime(2024, 1, 1),
            requested_end=datetime(2024, 1, 3, tzinfo=timezone.utc),
            timeframe="1d",
        )


def test_audit_survivorship_rejects_non_utc_requested_end() -> None:
    """Audit should raise when requested_end is not UTC."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    series = pd.Series([10.0, 11.0, 12.0], index=index)

    with pytest.raises(ValueError, match="requested_end"):
        audit_survivorship_bias(
            series,
            symbol="BTC/USD",
            requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            requested_end=datetime.fromisoformat("2024-01-03T00:00:00+01:00"),
            timeframe="1d",
        )


def test_audit_survivorship_rejects_reversed_window() -> None:
    """Audit should raise when requested_end is before requested_start."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    series = pd.Series([10.0, 11.0, 12.0], index=index)

    with pytest.raises(ValueError, match="requested_end"):
        audit_survivorship_bias(
            series,
            symbol="BTC/USD",
            requested_start=datetime(2024, 1, 3, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            timeframe="1d",
        )
