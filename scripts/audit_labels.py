"""Flag occasion labels that their own title contradicts.

3900 listings cannot be reviewed by hand, but most errors violate a checkable
invariant — a card labelled kids whose title says "29th", a card in general
whose title says "1st Birthday". This recomputes labels with the current rules
and reports every violation, with examples, so rules can be fixed until the
counts reach zero.

Read-only.

    python -m scripts.audit_labels            # audit current rules
    python -m scripts.audit_labels --stored   # audit what is in the database
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
import typer

from common.db import engine
from data.features.occasion_classifier import (
    KID_AGE_MAX,
    _ages_in,
    _has_phrase,
    _IN_LAW_RE,
    pick_best_occasion,
    weak_label,
)
from common.occasions import MILESTONE_AGES

_SQL = """
SELECT COALESCE(l.title, '') AS title, lf.occasion AS stored
FROM listings l
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id;
"""

_KID_WORDS = [
    "kid", "kids", "child", "children", "son", "daughter", "nephew", "niece",
    "grandson", "granddaughter", "toddler", "baby", "boy", "girl", "little one",
    "first birthday", "second birthday", "third birthday",
]
_REL_WORDS = [
    "husband", "wife", "boyfriend", "girlfriend", "partner", "fiance", "fiancee",
    "hubby", "wifey", "soulmate", "other half", "love of my life",
]


def _signals(title: str) -> dict[str, bool]:
    # Same in-law neutralisation the classifier applies, or the audit would
    # flag "Son in Law" as a kid signal and report a false violation.
    t = _IN_LAW_RE.sub("inlaw", title.lower())
    ages = _ages_in(t)
    return {
        "kid_word": any(_has_phrase(t, w) for w in _KID_WORDS),
        "kid_age": any(a <= KID_AGE_MAX for a in ages),
        "adult_age": bool(ages) and all(a > KID_AGE_MAX for a in ages),
        "milestone_age": any(a in MILESTONE_AGES for a in ages),
        "rel_word": any(_has_phrase(t, w) for w in _REL_WORDS),
    }


def _violations(occasion: str | None, sig: dict[str, bool]) -> list[str]:
    """Checks that hold regardless of how the rules are written."""
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
        # Under-matching: a stronger subtype signal was present but missed.
        if sig["kid_age"] or sig["kid_word"]:
            out.append("general_but_kid_signal")
        if sig["milestone_age"]:
            out.append("general_but_milestone_age")
        if sig["rel_word"]:
            out.append("general_but_relationship_word")
    return out


def main(stored: bool = typer.Option(False, help="Audit DB labels instead of recomputed ones")) -> None:
    df = pd.read_sql(_SQL, engine())
    if df.empty:
        print("No listings found.")
        return

    labels = (
        df["stored"].tolist()
        if stored
        else [pick_best_occasion(weak_label(t)) for t in df["title"]]
    )

    dist: Counter = Counter(l or "(none)" for l in labels)
    flagged: dict[str, list[str]] = defaultdict(list)
    for title, occ in zip(df["title"], labels):
        for v in _violations(occ, _signals(title)):
            flagged[v].append(title)

    source = "stored in DB" if stored else "recomputed with current rules"
    print(f"\n{len(df)} listings — labels {source}\n")
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
        for t in titles[:8]:
            print(f"      {t[:74]}")


if __name__ == "__main__":
    typer.run(main)
