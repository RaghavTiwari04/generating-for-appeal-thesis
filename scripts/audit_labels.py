"""Flag occasion labels that their own title contradicts.

3900 listings cannot be reviewed by hand, but most errors violate a checkable
invariant — a card labelled kids whose title says "29th", a card in general
whose title says "1st Birthday".

Deliberately independent of the classifier: it re-derives signals from the
title itself, so it can catch mistakes the classifier makes rather than
agreeing with it by construction.

Read-only.

    python -m scripts.audit_labels
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd
import typer

from common.db import engine
from common.occasions import MILESTONE_AGES

KID_AGE_MAX = 12

# Whole numbers only: "29th" must not read as "9th".
_ORDINAL_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\b")
_AGE_RE = re.compile(r"\bage\s+(\d{1,3})\b")
_YEARS_OLD_RE = re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b")
# In-laws are adults, so they must not count as a kid signal.
_IN_LAW_RE = re.compile(r"\b(son|daughter|mother|father|brother|sister)[\s-]+in[\s-]+law\b")

_KID_WORDS = [
    "kid", "kids", "child", "children", "son", "daughter", "nephew", "niece",
    "grandson", "granddaughter", "toddler", "baby", "boy", "girl", "little one",
    "first birthday", "second birthday", "third birthday",
]
_REL_WORDS = [
    "husband", "wife", "boyfriend", "girlfriend", "partner", "fiance", "fiancee",
    "hubby", "wifey", "soulmate", "other half", "love of my life",
]

_SQL = """
SELECT COALESCE(l.title, '') AS title, lf.occasion
FROM listings l
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id;
"""


def _ages(text: str) -> set[int]:
    out: set[int] = set()
    for pattern in (_ORDINAL_RE, _AGE_RE, _YEARS_OLD_RE):
        out.update(int(m) for m in pattern.findall(text))
    return out


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _signals(title: str) -> dict[str, bool]:
    t = _IN_LAW_RE.sub("inlaw", title.lower())
    ages = _ages(t)
    return {
        "kid_word": any(_has(t, w) for w in _KID_WORDS),
        "kid_age": any(a <= KID_AGE_MAX for a in ages),
        "adult_age": bool(ages) and all(a > KID_AGE_MAX for a in ages),
        "milestone_age": any(a in MILESTONE_AGES for a in ages),
        "rel_word": any(_has(t, w) for w in _REL_WORDS),
    }


def _violations(occasion: str | None, sig: dict[str, bool]) -> list[str]:
    out = []
    if occasion == "birthday/kids":
        if sig["adult_age"]:
            out.append("kids_but_adult_age")
        elif not (sig["kid_word"] or sig["kid_age"]):
            out.append("kids_but_no_kid_signal")
    elif occasion == "birthday/milestone":
        if not sig["milestone_age"]:
            out.append("milestone_but_no_milestone_age")
    elif occasion == "birthday/relationship":
        if not sig["rel_word"]:
            out.append("relationship_but_no_relationship_word")
    elif occasion == "birthday/general":
        if sig["kid_age"] or sig["kid_word"]:
            out.append("general_but_kid_signal")
        if sig["milestone_age"]:
            out.append("general_but_milestone_age")
        if sig["rel_word"]:
            out.append("general_but_relationship_word")
    return out


def main(examples: int = 8) -> None:
    df = pd.read_sql(_SQL, engine())
    if df.empty:
        print("No listings found.")
        return

    # SQL NULL can arrive as NaN rather than None depending on dtype.
    labels = [None if pd.isna(v) else v for v in df["occasion"]]

    dist: Counter = Counter(occ or "(none)" for occ in labels)
    flagged: dict[str, list[str]] = defaultdict(list)
    for title, occ in zip(df["title"], labels):
        for v in _violations(occ, _signals(title)):
            flagged[v].append(title)

    print(f"\n{len(df)} listings\n")
    print("distribution:")
    for occ, n in dist.most_common():
        print(f"  {occ:28s} {n}")

    total = sum(len(v) for v in flagged.values())
    print(f"\nviolations: {total}")
    if not flagged:
        print("  none — every label is consistent with its title")
        return
    for kind in sorted(flagged, key=lambda k: -len(flagged[k])):
        titles = flagged[kind]
        print(f"\n  {kind}: {len(titles)}")
        for t in titles[:examples]:
            print(f"      {t[:74]}")


if __name__ == "__main__":
    typer.run(main)
