"""Tests for the occasion-label audit invariants (no DB, no model)."""

from __future__ import annotations

from common.occasions import parse_ages as _ages
from scripts.audit_labels import _has, _signals, _violations


class TestAgeParsing:
    def test_ordinal_parsed_as_whole_number(self) -> None:
        # "29th" must not read as "9th" — this mislabelled adult cards as kids.
        assert _ages("happy 29th birthday") == {29}

    def test_double_digit_ordinal_not_split(self) -> None:
        assert _ages("33rd birthday celebration") == {33}

    def test_kid_ordinal(self) -> None:
        assert _ages("3rd birthday dinosaur") == {3}

    def test_age_phrase(self) -> None:
        assert _ages("galaxy girl birthday age 4") == {4}

    def test_years_old_phrase(self) -> None:
        assert _ages("50 years old today") == {50}

    def test_no_age(self) -> None:
        assert _ages("super cute tortoise birthday") == set()

    def test_bare_age_forms(self) -> None:
        # "Sporty at 60" was read as having no age, which made the audit flag
        # correctly-labelled milestone cards.
        assert _ages("sporty at 60 - birthday card") == {60}
        assert _ages("turning 30 birthday card") == {30}


class TestWordBoundaries:
    def test_substring_does_not_match(self) -> None:
        # "Robinson" contains "son" but is not a kid signal.
        assert not _has("colin robinson birthday party", "son")

    def test_whole_word_matches(self) -> None:
        assert _has("birthday card for my son", "son")


class TestSignals:
    def test_in_law_is_not_a_kid_signal(self) -> None:
        assert not _signals("Happy Birthday to Future Son in Law")["kid_word"]

    def test_adult_age_detected(self) -> None:
        sig = _signals("Happy 29th Birthday")
        assert sig["adult_age"] and not sig["kid_age"]

    def test_kid_age_detected(self) -> None:
        sig = _signals("3rd Birthday Dinosaur")
        assert sig["kid_age"] and not sig["adult_age"]

    def test_milestone_age_detected(self) -> None:
        assert _signals("30th Birthday Card")["milestone_age"]

    def test_relationship_word_detected(self) -> None:
        assert _signals("Happy Birthday Husband")["rel_word"]


class TestViolations:
    def test_kids_with_adult_age_flagged(self) -> None:
        v = _violations("birthday/kids", _signals("Happy 29th Birthday"))
        assert "kids_but_adult_age" in v

    def test_kids_with_no_signal_flagged(self) -> None:
        v = _violations("birthday/kids", _signals("Colin Robinson Birthday party"))
        assert "kids_but_no_kid_signal" in v

    def test_genuine_kids_not_flagged(self) -> None:
        assert _violations("birthday/kids", _signals("3rd Birthday Dinosaur")) == []

    def test_milestone_without_age_flagged(self) -> None:
        v = _violations("birthday/milestone", _signals("A lovely birthday card"))
        assert "milestone_but_no_milestone_age" in v

    def test_genuine_milestone_not_flagged(self) -> None:
        assert _violations("birthday/milestone", _signals("30th Birthday Card")) == []

    def test_general_hiding_a_subtype_flagged(self) -> None:
        v = _violations("birthday/general", _signals("Happy 1st Birthday"))
        assert "general_but_kid_signal" in v

    def test_genuine_general_not_flagged(self) -> None:
        assert _violations("birthday/general", _signals("Super cute tortoise birthday")) == []

    def test_unlabelled_never_flagged(self) -> None:
        assert _violations(None, _signals("Surgeon Greeting Card")) == []
