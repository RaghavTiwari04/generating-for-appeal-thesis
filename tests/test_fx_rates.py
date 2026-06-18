"""Unit tests for FX rate helpers (offline — no network calls)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from common.fx_rates import get_rate, normalise_price_column, to_gbp


class TestGetRate:
    def test_gbp_identity(self) -> None:
        assert get_rate("GBP") == 1.0

    def test_gbp_lowercase(self) -> None:
        assert get_rate("gbp") == 1.0

    def test_usd_uses_fallback(self) -> None:
        # Patch _load_cache and _fetch_live to return None → uses fallback
        with patch("common.fx_rates._load_cache", return_value=None), \
             patch("common.fx_rates._fetch_live", return_value=None):
            rate = get_rate("USD")
        # Fallback USD is ~1.27 GBP per 1 USD → rate should be ~1/1.27
        assert 0.7 < rate < 0.9

    def test_unknown_currency_returns_1(self) -> None:
        with patch("common.fx_rates._load_cache", return_value=None), \
             patch("common.fx_rates._fetch_live", return_value=None):
            rate = get_rate("XYZ")
        assert rate == 1.0

    def test_uses_cache_when_fresh(self) -> None:
        cached = {"USD": 0.80, "EUR": 0.86}
        with patch("common.fx_rates._load_cache", return_value=cached):
            rate = get_rate("USD")
        assert rate == pytest.approx(0.80)

    def test_eur_rate_reasonable(self) -> None:
        cached = {"EUR": 0.86}
        with patch("common.fx_rates._load_cache", return_value=cached):
            rate = get_rate("EUR")
        assert 0.7 < rate < 1.2


class TestToGbp:
    def test_none_minor_returns_none(self) -> None:
        assert to_gbp(None, "GBP") is None

    def test_none_currency_returns_none(self) -> None:
        assert to_gbp(399, None) is None

    def test_gbp_passthrough(self) -> None:
        result = to_gbp(399, "GBP")
        assert result == pytest.approx(3.99)

    def test_usd_conversion(self) -> None:
        # 500 USD cents = $5.00; at rate ~0.79 → ~£3.95
        with patch("common.fx_rates._load_cache", return_value={"USD": 0.79}):
            result = to_gbp(500, "USD")
        assert result == pytest.approx(3.95, abs=0.01)

    def test_zero_price(self) -> None:
        assert to_gbp(0, "GBP") == pytest.approx(0.0)


class TestNormalisePriceColumn:
    def test_adds_price_gbp_column(self) -> None:
        import pandas as pd
        df = pd.DataFrame({
            "price_minor_units": [399, 500, None],
            "currency": ["GBP", "GBP", "GBP"],
        })
        normalise_price_column(df)
        assert "price_gbp" in df.columns
        assert df["price_gbp"][0] == pytest.approx(3.99)

    def test_none_produces_none(self) -> None:
        import pandas as pd
        df = pd.DataFrame({
            "price_minor_units": [None],
            "currency": ["GBP"],
        })
        normalise_price_column(df)
        assert df["price_gbp"][0] is None
