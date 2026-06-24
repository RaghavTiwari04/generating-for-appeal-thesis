"""Fetch daily FX rates and normalise prices to GBP.

Uses exchangerate.host (free, no key required) with a 24-hour disk cache.
Falls back to hardcoded approximate rates if network is unavailable so the
pipeline never blocks on FX fetch failures.

Usage:
    rate = get_rate("USD")      # USD → GBP
    gbp  = to_gbp(500, "USD")   # 500 USD cents → GBP
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from common.logging import get_logger

log = get_logger(__name__)

_CACHE_PATH = Path(".cache/fx_rates.json")
_CACHE_TTL_SEC = 86_400  # 24 hours
_API_URL = "https://api.frankfurter.app/latest?base=GBP"  # free, open-source ECB data

# Fallback rates (GBP base, rough 2024 values — updated by live fetch)
_FALLBACK: dict[str, float] = {
    "GBP": 1.0,
    "USD": 1.27,
    "EUR": 1.17,
    "CAD": 1.71,
    "AUD": 1.94,
}


def _load_cache() -> dict[str, float] | None:
    if not _CACHE_PATH.exists():
        return None
    age = time.time() - _CACHE_PATH.stat().st_mtime
    if age > _CACHE_TTL_SEC:
        return None
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return None


def _save_cache(rates: dict[str, float]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(rates))


def _fetch_live() -> dict[str, float] | None:
    try:
        resp = httpx.get(_API_URL, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        if not rates:
            return None
        # API returns X per 1 GBP, so GBP→X = rates[X]
        # We want X→GBP = 1/rates[X]
        return {ccy: 1.0 / float(v) for ccy, v in rates.items() if float(v) > 0}
    except Exception as e:
        log.warning(f"FX fetch failed: {e}. Using fallback rates.")
        return None


def _get_rates() -> dict[str, float]:
    cached = _load_cache()
    if cached:
        return cached
    live = _fetch_live()
    if live:
        _save_cache(live)
        return live
    log.warning("Using hardcoded fallback FX rates")
    return {k: 1.0 / v for k, v in _FALLBACK.items()}


def get_rate(from_currency: str) -> float:
    """Return conversion rate: 1 unit of `from_currency` in GBP."""
    if from_currency.upper() == "GBP":
        return 1.0
    rates = _get_rates()
    rate = rates.get(from_currency.upper())
    if rate is None:
        log.warning(f"Unknown currency {from_currency!r}; treating as GBP")
        return 1.0
    return rate


def to_gbp(minor_units: int | None, currency: str | None) -> float | None:
    """Convert minor units (pence/cents) in `currency` to GBP float.

    Returns None if either input is None or not a valid string.
    """
    if minor_units is None or currency is None:
        return None
    if not isinstance(currency, str):
        return None
    rate = get_rate(currency)
    return round(minor_units / 100.0 * rate, 4)


def normalise_price_column(df, minor_col: str = "price_minor_units",
                            currency_col: str = "currency",
                            out_col: str = "price_gbp") -> None:
    """Add `out_col` (GBP float) to df in-place."""

    df[out_col] = [
        to_gbp(m, c)
        for m, c in zip(df[minor_col], df[currency_col], strict=False)
    ]
