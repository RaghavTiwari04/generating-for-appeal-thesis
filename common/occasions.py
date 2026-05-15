"""Canonical occasion taxonomy

Multi-label classification target. Tones (humorous, sincere, religious, etc.)
are orthogonal and represented separately on a card.
"""

from __future__ import annotations

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
    return occ in OCCASIONS
