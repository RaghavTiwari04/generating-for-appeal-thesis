"""Canonical occasion taxonomy

Multi-label classification target. Tones (humorous, sincere, religious, etc.)
are orthogonal and represented separately on a card.

ACTIVE_OCCASIONS controls what the system actually supports at runtime.
To add more card types, extend ACTIVE_OCCASIONS with entries from OCCASIONS.
"""

from __future__ import annotations

import re

# Full taxonomy — reference only; do not use directly in pipeline/model code.
OCCASIONS: tuple[str, ...] = (
    "birthday/general",
    "birthday/milestone",
    "birthday/kids",
    "birthday/relationship",
    "christmas/general",
    "christmas/religious",
    "christmas/humorous",
    "christmas/family-specific",
    "mothers_day",
    "fathers_day",
    "valentines_day",
    "easter",
    "anniversary/general",
    "anniversary/wedding-milestone",
    "wedding/congratulations",
    "wedding/engagement",
    "new_baby",
    "sympathy/bereavement",
    "sympathy/get_well",
    "thank_you",
    "congratulations/general",
    "congratulations/exam",
    "congratulations/new_job",
    "congratulations/new_home",
    "leaving/retirement",
    "leaving/job",
    "graduation",
    "encouragement",
    "just_because",
)

BIRTHDAY_OCCASIONS: tuple[str, ...] = (
    "birthday/general",
    "birthday/milestone",
    "birthday/kids",
    "birthday/relationship",
)

# Single control point: add occasion groups here to expand scope.
ACTIVE_OCCASIONS: tuple[str, ...] = BIRTHDAY_OCCASIONS

MILESTONE_AGES: tuple[int, ...] = (18, 21, 30, 40, 50, 60, 70, 80, 90, 100)

RELATIONSHIPS: tuple[str, ...] = (
    "mum",
    "dad",
    "sister",
    "brother",
    "partner",
    "friend",
    "colleague",
    "grandparent",
    "child",
    "auntuncle",
)

TONES: tuple[str, ...] = (
    "warm-sincere",
    "warm-humorous",
    "funny-irreverent",
    "formal-sincere",
    "minimalist",
    "religious",
    "sentimental",
)


def is_valid_occasion(occ: str) -> bool:
    return occ in ACTIVE_OCCASIONS


# ---------------------------------------------------------------------------
# Age parsing
# ---------------------------------------------------------------------------
# Ages must be read as whole numbers. Substring matching once put adult cards
# in birthday/kids ("29th" matching "9th", "33rd" matching "3rd"), and a
# zero-shot model scored "14 Year Old Birthday Gift" as kids at 0.99 — a
# parsed number is simply more reliable than either.
KID_AGE_MAX: int = 12

_ORDINAL_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\b")
_AGE_RE = re.compile(r"\bage\s+(\d{1,3})\b")
_YEARS_OLD_RE = re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b")
_BARE_AGE_RE = re.compile(r"\b(?:at|turning)\s+(\d{1,3})\b")


def parse_ages(text: str) -> set[int]:
    """Every age mentioned in the text, as numbers."""
    lowered = text.lower()
    ages: set[int] = set()
    for pattern in (_ORDINAL_RE, _AGE_RE, _YEARS_OLD_RE, _BARE_AGE_RE):
        ages.update(int(m) for m in pattern.findall(lowered))
    return ages


def occasion_from_age(text: str) -> str | None:
    """Subtype implied by an explicit age, or None when the age is not decisive.

    Only ages that decide the question are returned: 14 is neither a child age
    nor a milestone, so it falls through to whatever the classifier says.
    """
    ages = parse_ages(text)
    if not ages:
        return None
    if any(a <= KID_AGE_MAX for a in ages):
        return "birthday/kids"
    if any(a in MILESTONE_AGES for a in ages):
        return "birthday/milestone"
    return None


def ages_rule_out_kids(text: str) -> bool:
    """True when every age present is above the child range.

    occasion_from_age stays silent on 13-16 because they are neither a child
    age nor a milestone, which let a zero-shot model label "16th Birthday
    Girl" as kids. The ages still rule kids out even when they decide nothing
    else.
    """
    ages = parse_ages(text)
    return bool(ages) and all(a > KID_AGE_MAX for a in ages)
