"""Scraper parse-logic tests using fixture HTML (no network)."""

from __future__ import annotations

import pytest

from data.scrapers.base import ParsedListing
from data.scrapers.redbubble import RedbubbleScraper

REDBUBBLE_URL = "https://www.redbubble.com/people/artshop/works/987654321-birthday-watercolour"



class TestRedbubbleParser:
    def _parse(self, html: str) -> ParsedListing:
        return RedbubbleScraper().parse(html, REDBUBBLE_URL)

    def test_source_listing_id(self, redbubble_html: str) -> None:
        r = self._parse(redbubble_html)
        assert r.source_listing_id == "987654321"

    def test_title(self, redbubble_html: str) -> None:
        r = self._parse(redbubble_html)
        assert r.title and "Birthday" in r.title

    def test_price_usd(self, redbubble_html: str) -> None:
        r = self._parse(redbubble_html)
        assert r.price_minor_units == 450
        assert r.currency == "USD"

    def test_seller_id(self, redbubble_html: str) -> None:
        r = self._parse(redbubble_html)
        assert r.seller_id == "artshop"

    def test_image_urls(self, redbubble_html: str) -> None:
        r = self._parse(redbubble_html)
        assert len(r.image_urls) >= 1


class TestPriceParser:
    """Edge-case tests for the shared price parser."""

    def _p(self, text: str):
        from data.scrapers.parsing import _parse_price
        return _parse_price(text)

    def test_gbp_pence(self) -> None:
        assert self._p("£3.99") == (399, "GBP")

    def test_usd_cents(self) -> None:
        assert self._p("$12.50") == (1250, "USD")

    def test_eur(self) -> None:
        assert self._p("€5.00") == (500, "EUR")

    def test_none_input(self) -> None:
        assert self._p(None) == (None, None)

    def test_no_price_in_string(self) -> None:
        assert self._p("Add to cart") == (None, None)

    def test_whole_pounds(self) -> None:
        assert self._p("£4") == (400, "GBP")

    def test_with_surrounding_text(self) -> None:
        minor, currency = self._p("From £2.50 per card")
        assert minor == 250
        assert currency == "GBP"
