"""Occasion classifier: keyword rules over listing titles.

Assigns each listing one occasion from ACTIVE_OCCASIONS, writing
listing_features.occasion plus a multilabel vector.

Rules, not a model. A DistilBERT variant was trained on weak keyword labels
in an earlier design, but it never outperformed the rules it was distilled
from and inference always used the rules; it has been removed.

`weak_label` is also the scraper's birthday gate (data/scrapers/base.py), so
a listing cannot be stored that this module would then refuse to label.

Usage:
    python -m data.features.occasion_classifier infer
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from common.db import connection
from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS
from common.occasions import MILESTONE_AGES

log = get_logger(__name__)

app = typer.Typer()


@app.callback()
def _cli() -> None:
    """Occasion classification via keyword rules."""

OCCASION_TO_IDX = {o: i for i, o in enumerate(OCCASIONS)}
IDX_TO_OCCASION = {i: o for i, o in enumerate(OCCASIONS)}


# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------
# Ages are parsed as numbers, never as substrings. Matching "9th birthday"
# inside "Happy 29th Birthday", or "3rd birthday" inside "33rd Birthday", put
# adult cards in birthday/kids. See _ages_in().
_RULES: dict[str, list[str]] = {
    "birthday/general": ["birthday", "bday", "happy birthday"],
    "birthday/milestone": ["milestone"],
    "birthday/kids": [
        "kids birthday", "birthday kids", "children's birthday", "birthday children",
        "birthday child", "child's birthday", "birthday boy", "birthday girl",
        "birthday son", "son birthday", "birthday daughter", "daughter birthday",
        "birthday nephew", "birthday niece", "birthday grandson", "birthday granddaughter",
        "first birthday", "second birthday", "third birthday",
        "little one", "toddler", "baby birthday",
    ],
    "birthday/relationship": [
        "boyfriend birthday", "birthday boyfriend", "boyfriend card", "boyfriend happy",
        "girlfriend birthday", "birthday girlfriend", "girlfriend card", "girlfriend happy",
        "husband birthday", "birthday husband", "husband card", "hubby birthday", "hubby card",
        "wife birthday", "birthday wife", "wife card", "wifey birthday", "wifey card",
        "partner birthday", "birthday partner", "partner card",
        "fiance birthday", "birthday fiance", "fiancee birthday", "birthday fiancee",
        "for him birthday", "birthday for him", "for her birthday", "birthday for her",
        "other half", "soulmate", "love of my life",
    ],
    "christmas/general": ["christmas", "xmas", "festive", "merry christmas"],
    "christmas/humorous": ["christmas funny", "funny christmas", "humorous christmas"],
    "mothers_day": ["mother's day", "mothers day", "mum birthday", "mom birthday"],
    "fathers_day": ["father's day", "fathers day", "dad birthday"],
    "valentines_day": ["valentines", "valentine", "love you", "be mine"],
    "easter": ["easter", "happy easter", "easter bunny"],
    "anniversary/general": ["anniversary", "years together", "years married"],
    "wedding/congratulations": ["wedding", "newly wed", "congratulations on your wedding"],
    "wedding/engagement": ["engagement", "engaged", "congratulations engaged"],
    "new_baby": ["new baby", "baby shower", "congratulations baby", "newborn"],
    "sympathy/bereavement": ["sympathy", "condolences", "sorry for your loss", "bereavement", "with deepest sympathy"],
    "sympathy/get_well": ["get well", "get better", "speedy recovery", "feel better"],
    "thank_you": ["thank you", "thanks", "grateful"],
    "congratulations/general": ["congratulations", "congrats", "well done"],
    "congratulations/exam": ["exam", "results", "a-level", "gcse", "degree"],
    "congratulations/new_job": ["new job", "promotion", "new role"],
    "congratulations/new_home": ["new home", "moving", "housewarming"],
    "leaving/retirement": ["retirement", "retiring", "happy retirement"],
    "leaving/job": ["leaving", "farewell", "goodbye", "new adventure"],
    "graduation": ["graduation", "graduate", "well done graduate"],
    "encouragement": ["thinking of you", "you've got this", "keep going"],
    "just_because": ["just because", "no occasion", "thinking of you"],
}

_COOCCURRENCE_RULES: dict[str, list[tuple[str, ...]]] = {
    "birthday/kids": [
        ("birthday", "kid"), ("birthday", "child"), ("birthday", "children"),
        ("birthday", "son"), ("birthday", "daughter"),
        ("birthday", "nephew"), ("birthday", "niece"),
        ("birthday", "grandson"), ("birthday", "granddaughter"),
        ("birthday", "toddler"), ("birthday", "baby"),
        ("birthday", "young"), ("birthday", "little"),
        ("bday", "kid"), ("bday", "child"), ("bday", "son"), ("bday", "daughter"),
    ],
    "birthday/relationship": [
        ("birthday", "husband"), ("birthday", "wife"),
        ("birthday", "boyfriend"), ("birthday", "girlfriend"),
        ("birthday", "partner"), ("birthday", "fiance"), ("birthday", "fiancee"),
        ("birthday", "hubby"), ("birthday", "wifey"),
        ("birthday", "for him"), ("birthday", "for her"),
        ("birthday", "soulmate"), ("birthday", "other half"),
        ("bday", "husband"), ("bday", "wife"),
        ("bday", "boyfriend"), ("bday", "girlfriend"),
    ],
}


KID_AGE_MAX = 12

# "18th", "3rd", "29th" — the whole number, so "29th" cannot match as "9th".
_ORDINAL_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\b")
_AGE_RE = re.compile(r"\bage\s+(\d{1,3})\b")
_YEARS_OLD_RE = re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b")

# In-laws are adults. Without this "Birthday to Future Son in Law" matches the
# (birthday, son) rule and lands in kids.
_IN_LAW_RE = re.compile(r"\b(son|daughter|mother|father|brother|sister)[\s-]+in[\s-]+law\b")


def _ages_in(text: str) -> set[int]:
    """Every age mentioned, parsed as a number rather than matched as text."""
    ages: set[int] = set()
    for pattern in (_ORDINAL_RE, _AGE_RE, _YEARS_OLD_RE):
        ages.update(int(m) for m in pattern.findall(text))
    return ages


def _has_phrase(text: str, phrase: str) -> bool:
    """Whole-word match. Substring matching put 'Colin Robinson' in kids."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def weak_label(text: str) -> list[str]:
    text_l = text.lower()
    # Neutralise in-law phrases before any family-term matching.
    text_l = _IN_LAW_RE.sub("inlaw", text_l)

    labels = []
    for occasion, keywords in _RULES.items():
        if any(_has_phrase(text_l, kw) for kw in keywords):
            labels.append(occasion)
    for occasion, word_groups in _COOCCURRENCE_RULES.items():
        if occasion in labels:
            continue
        for words in word_groups:
            if all(_has_phrase(text_l, w) for w in words):
                labels.append(occasion)
                break

    # Ages decide kids vs milestone, so a wrong ordinal cannot mislabel a card.
    ages = _ages_in(text_l)
    if ages:
        if any(a in MILESTONE_AGES for a in ages) and "birthday/milestone" not in labels:
            labels.append("birthday/milestone")
        if any(a <= KID_AGE_MAX for a in ages) and "birthday/kids" not in labels:
            labels.append("birthday/kids")
        # An adult age present with no kid age must not stay in kids.
        if all(a > KID_AGE_MAX for a in ages) and "birthday/kids" in labels:
            labels.remove("birthday/kids")

    return labels


# ---------------------------------------------------------------------------
# Inference over missing listings
# ---------------------------------------------------------------------------
_SELECT_MISSING = """
SELECT l.listing_id,
       COALESCE(l.title,'')       AS title,
       COALESCE(l.description,'') AS description
FROM listings l
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.occasion IS NULL
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

# LEFT JOIN: a freshly scraped listing has no listing_features row yet, and an
# inner join silently classified nothing after a full wipe. The upsert below
# creates the row, so it does not need to pre-exist.
_SELECT_ALL = """
SELECT l.listing_id,
       COALESCE(l.title,'')       AS title,
       COALESCE(l.description,'') AS description
FROM listings l
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

# Clear a stale label when nothing matches. Without this, re-running infer
# leaves whatever a previous pass wrote — which is how 312 listings with no
# occasion keyword at all ("Surgeon Greeting Card") stayed on birthday/general.
_CLEAR = """
UPDATE listing_features
SET occasion = NULL,
    occasion_confidence = NULL,
    occasion_multilabel = NULL,
    computed_at = NOW()
WHERE listing_id = %(listing_id)s;
"""

_UPSERT = """
INSERT INTO listing_features (listing_id, occasion, occasion_confidence, occasion_multilabel, feature_version)
VALUES (%(listing_id)s, %(occasion)s, %(confidence)s, %(multilabel)s, 'keyword-v2')
ON CONFLICT (listing_id) DO UPDATE
SET occasion = EXCLUDED.occasion,
    occasion_confidence = EXCLUDED.occasion_confidence,
    occasion_multilabel = EXCLUDED.occasion_multilabel,
    computed_at = NOW();
"""


_SPECIFICITY_ORDER = [o for o in OCCASIONS if not o.endswith("/general")] + \
                     [o for o in OCCASIONS if o.endswith("/general")]


def pick_best_occasion(labels: list[str]) -> str | None:
    """Pick most specific occasion from a list of keyword matches.

    Sub-occasions (kids, relationship, milestone) win over /general
    when both match.
    """
    if not labels:
        return None
    for occ in _SPECIFICITY_ORDER:
        if occ in labels:
            return occ
    return labels[0]


@app.command()
def infer(
    limit: int = 50000,
    reclassify_all: bool = typer.Option(True, help="Re-classify all listings"),
    use_description: bool = typer.Option(
        False,
        help="Also match against the description. Off by default: marketplace "
             "descriptions are vendor/template boilerplate and outvote the "
             "actual card content (it put all 500 Greetings Island cards in "
             "birthday/kids, none of which say 'kid' in the title).",
    ),
) -> None:
    """Classify listings using keyword rules (no model needed)."""
    query = _SELECT_ALL if reclassify_all else _SELECT_MISSING
    with connection() as conn, conn.cursor() as cur:
        cur.execute(query, {"limit": limit})
        rows = cur.fetchall()

    src = "title + description" if use_description else "title only"
    log.info(f"Classifying {len(rows)} listings with keyword rules ({src})")
    processed = 0
    cleared = 0
    stats: dict[str, int] = {}

    with connection() as conn, conn.cursor() as cur:
        for r in rows:
            text = r["title"] or ""
            if use_description:
                text = f"{text} {r['description'] or ''}"
            labels = weak_label(text)
            occasion = pick_best_occasion(labels)
            if not occasion:
                # Clear rather than skip, so a stale label from an earlier pass
                # does not survive a re-classification.
                cur.execute(_CLEAR, {"listing_id": r["listing_id"]})
                cleared += 1
                continue
            confidence = 1.0 if len(labels) == 1 else 0.8
            multilabel = {o: (1.0 if o in labels else 0.0) for o in OCCASIONS}
            cur.execute(
                _UPSERT,
                {
                    "listing_id": r["listing_id"],
                    "occasion": occasion,
                    "confidence": confidence,
                    "multilabel": json.dumps(multilabel),
                },
            )
            stats[occasion] = stats.get(occasion, 0) + 1
            processed += 1

    log.info(f"Keyword classification complete: {processed} labelled, {cleared} cleared")
    for occ, n in sorted(stats.items(), key=lambda x: -x[1]):
        log.info(f"  {occ}: {n}")


if __name__ == "__main__":
    app()
