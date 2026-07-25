"""Shared HTML/JSON-LD parsing helpers used by every scraper.

These previously lived in `etsy.py` and were imported from there by the other
scrapers, which made an unused marketplace module load-bearing. They are
marketplace-agnostic, so they belong here.
"""

from __future__ import annotations

import json
import re

from selectolax.parser import HTMLParser


def _extract_jsonld(tree: HTMLParser, typ: str) -> dict | None:
    for node in tree.css("script[type='application/ld+json']"):
        try:
            data = json.loads(node.text())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == typ:
                    return item
        elif isinstance(data, dict):
            if data.get("@type") == typ:
                return data
            for item in data.get("@graph", []):
                if isinstance(item, dict) and item.get("@type") == typ:
                    return item
    return None


def _text(node) -> str | None:
    if node is None:
        return None
    return node.text(strip=True) or None


def _attr(node, key: str) -> str | None:
    if node is None:
        return None
    return node.attributes.get(key)


_PRICE_RE = re.compile(r"([£$€])\s*([0-9]+(?:[.,][0-9]{1,2})?)")
_CURRENCY_MAP = {"£": "GBP", "$": "USD", "€": "EUR"}


def _parse_price(text: str | None) -> tuple[int | None, str | None]:
    if not text:
        return None, None
    m = _PRICE_RE.search(text)
    if not m:
        return None, None
    symbol, amount = m.group(1), m.group(2).replace(",", ".")
    try:
        return round(float(amount) * 100), _CURRENCY_MAP.get(symbol)
    except ValueError:
        return None, None


def _to_int(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    digits = re.sub(r"[^0-9]", "", str(val))
    return int(digits) if digits else None


def _to_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, float):
        return val
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", str(val))
    return float(m.group(0)) if m else None
