"""Tests for pipeline.conditions.

The naive baseline is the thing everything else is measured against, so a defect
in it inflates every reported effect size rather than showing up as a failure.
One did: the headline was built by formatting the whole occasion path, so cards
read "Happy Birthday General" and "Happy Birthday Relationship". Correcting it
moved condition A from 0.407 to 0.624 and roughly halved every effect size in
the results chapter. TestNaiveHeadline is the regression net for that.
"""

from __future__ import annotations

import re

import pytest

from common.occasions import ACTIVE_OCCASIONS, OCCASIONS
from pipeline.conditions import CONDITION_TAGS, NAIVE_TONE, _naive_headline

# The subtype segments that leaked into headlines when the whole path was
# formatted rather than just the group.
SUBTYPE_WORDS = {"general", "kids", "milestone", "relationship", "bereavement"}


class TestNaiveHeadline:
    @pytest.mark.parametrize("occasion", ACTIVE_OCCASIONS)
    def test_every_birthday_subtype_gets_the_same_plain_greeting(self, occasion):
        assert _naive_headline(occasion) == "Happy Birthday"

    @pytest.mark.parametrize("occasion", OCCASIONS)
    def test_no_taxonomy_subtype_ever_reaches_the_card(self, occasion):
        words = set(re.findall(r"[a-z']+", _naive_headline(occasion).lower()))
        assert not (words & SUBTYPE_WORDS)

    @pytest.mark.parametrize("occasion", OCCASIONS)
    def test_headline_is_human_readable(self, occasion):
        headline = _naive_headline(occasion)
        assert headline.strip() == headline
        assert headline
        assert "_" not in headline
        assert "/" not in headline

    def test_the_group_decides_the_greeting_not_the_subtype(self):
        greetings = {_naive_headline(o) for o in OCCASIONS if o.startswith("birthday/")}
        assert greetings == {"Happy Birthday"}

    @pytest.mark.parametrize(
        ("occasion", "expected"),
        [
            ("birthday/general", "Happy Birthday"),
            ("christmas/general", "Merry Christmas"),
            ("mothers_day", "Happy Mother's Day"),
            ("sympathy/bereavement", "With Sympathy"),
            ("new_baby", "Congratulations"),
            ("thank_you", "Thank You"),
        ],
    )
    def test_known_occasions_map_to_their_greeting(self, occasion, expected):
        assert _naive_headline(occasion) == expected

    def test_unknown_occasion_falls_back_to_a_titlecased_group(self):
        assert _naive_headline("graduation/university") == "Graduation"
        assert _naive_headline("new_job") == "New Job"

    def test_bare_group_and_pathed_group_agree(self):
        assert _naive_headline("birthday") == _naive_headline("birthday/kids")


class TestConditionTags:
    def test_all_four_conditions_are_tagged(self):
        assert set(CONDITION_TAGS) == {"A", "B", "C", "D"}

    def test_tags_are_distinct(self):
        assert len(set(CONDITION_TAGS.values())) == 4

    def test_each_tag_is_prefixed_with_its_condition_letter(self):
        for letter, tag in CONDITION_TAGS.items():
            assert tag.startswith(f"{letter}_")

    def test_condition_d_is_not_called_a_bestseller(self):
        # No engagement data was ever captured, so D is a random sample of
        # human-designed cards. Calling it "bestseller" asserts a commercial
        # fact the corpus cannot support.
        assert CONDITION_TAGS["D"] == "D_human_reference"
        assert "bestseller" not in " ".join(CONDITION_TAGS.values()).lower()


class TestNaiveTone:
    def test_naive_tone_is_a_fixed_default(self):
        # Conditions B and C let the brief choose a tone. A does not: picking one
        # is part of what the pipeline contributes, so handing it to the baseline
        # would remove the thing under test.
        assert NAIVE_TONE == "warm-sincere"

    def test_naive_tone_is_a_recognised_tone(self):
        from common.occasions import TONES

        assert NAIVE_TONE in TONES
