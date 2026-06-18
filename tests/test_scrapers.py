"""Scraper parse-logic tests using fixture HTML (no network)."""

from __future__ import annotations

import pytest

from data.scrapers.base import ParsedListing
from data.scrapers.etsy import EtsyScraper
from data.scrapers.redbubble import RedbubbleScraper

ETSY_URL = "https://www.etsy.com/listing/123456789/happy-birthday-mum"
REDBUBBLE_URL = "https://www.redbubble.com/people/artshop/works/987654321-birthday-watercolour"


class TestEtsyParser:
    def _parse(self, html: str) -> ParsedListing:
        return EtsyScraper().parse(html, ETSY_URL)

    def test_source_listing_id_extracted(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.source_listing_id == "123456789"

    def test_url_preserved(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.url == ETSY_URL

    def test_title_parsed(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.title and "Birthday" in r.title

    def test_description_parsed(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.description and "watercolour" in r.description.lower()

    def test_seller_id_parsed(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.seller_id == "FloralPaperCo"

    def test_price_in_pence(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.price_minor_units == 399
        assert r.currency == "GBP"

    def test_review_count(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.review_count == 127

    def test_review_avg(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.review_avg == pytest.approx(4.9)

    def test_favourite_count(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.favourite_count == 2341

    def test_bestseller_flag(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert r.is_bestseller is True

    def test_image_urls_extracted(self, etsy_html: str) -> None:
        r = self._parse(etsy_html)
        assert len(r.image_urls) >= 1
        assert "etsy" in r.image_urls[0]

    def test_no_bestseller_flag(self) -> None:
        html = "<html><body><h1 data-buy-box-listing-title>Card</h1></body></html>"
        r = self._parse(html)
        assert r.is_bestseller is False

    def test_missing_price_returns_none(self) -> None:
        html = "<html><body><h1 data-buy-box-listing-title>Card</h1></body></html>"
        r = self._parse(html)
        assert r.price_minor_units is None
        assert r.currency is None

    def test_usd_price(self) -> None:
        html = (
            '<html><body><h1 data-buy-box-listing-title>Card</h1>'
            '<div data-buy-box-region="price">$5.00</div></body></html>'
        )
        r = self._parse(html)
        assert r.price_minor_units == 500
        assert r.currency == "USD"

    def test_eur_price(self) -> None:
        html = (
            '<html><body><h1 data-buy-box-listing-title>Card</h1>'
            '<div data-buy-box-region="price">€4.20</div></body></html>'
        )
        r = self._parse(html)
        assert r.price_minor_units == 420
        assert r.currency == "EUR"


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
        from data.scrapers.etsy import _parse_price
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
