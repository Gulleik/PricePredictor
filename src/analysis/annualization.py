"""Shared annualization helpers for analysis workflows."""


def periods_per_year_from_freq(freq: str) -> int:
    """Infer annualization periods from configured portfolio frequency."""
    normalized = freq.strip().lower()
    if normalized.endswith("h"):
        amount = int(normalized[:-1] or 1)
        return max(1, int(round((24 * 365) / amount)))
    if normalized.endswith("d"):
        amount = int(normalized[:-1] or 1)
        return max(1, int(round(252 / amount)))
    return 252
