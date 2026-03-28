"""Tests for parser helpers in src.data."""

from datetime import datetime, timezone

import pytest

from src.data import _parse_datetime


def test_parse_datetime_returns_input_datetime() -> None:
    """A datetime input should be returned unchanged."""
    value = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _parse_datetime(value) is value


def test_parse_datetime_raises_for_invalid_string() -> None:
    """Invalid date text should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Could not parse datetime value"):
        _parse_datetime("definitely not a date")