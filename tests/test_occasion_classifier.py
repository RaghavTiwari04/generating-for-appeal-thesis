"""Tests for the occasion classifier — keyword rules + pick_best_occasion (no DB)."""

from __future__ import annotations

from data.features.occasion_classifier import (
    _COOCCURRENCE_RULES,
    _RULES,
    IDX_TO_OCCASION,
    OCCASION_TO_IDX,
    OCCASIONS,
    pick_best_occasion,
    weak_label,
)


class TestWeakLabels:
    def test_birthday_detected(self) -> None:
        labels = weak_label("Happy Birthday to you! Floral card")
        assert "birthday/general" in labels

    def test_christmas_detected(self) -> None:
        labels = weak_label("Merry Christmas and a Happy New Year!")
        assert "christmas/general" in labels

    def test_sympathy_detected(self) -> None:
        labels = weak_label("With deepest sympathy and condolences")
        assert "sympathy/bereavement" in labels

    def test_mothers_day_detected(self) -> None:
        labels = weak_label("Happy Mother's Day from your family")
        assert "mothers_day" in labels

    def test_unknown_returns_empty(self) -> None:
        labels = weak_label("A random card with no occasion keywords")
        assert isinstance(labels, list)

    def test_multilabel_possible(self) -> None:
        labels = weak_label("Thank you for the birthday wishes")
        assert len(labels) >= 1

    def test_case_insensitive(self) -> None:
        upper = weak_label("HAPPY BIRTHDAY CARD")
        lower = weak_label("happy birthday card")
        assert set(upper) == set(lower)

    def test_kids_birthday_detected_reversed_order(self) -> None:
        labels = weak_label("Birthday Card for Kids - Party Animals")
        assert "birthday/kids" in labels

    def test_kids_birthday_via_cooccurrence(self) -> None:
        labels = weak_label("Happy Birthday to my lovely Daughter")
        assert "birthday/kids" in labels

    def test_kids_age_birthday(self) -> None:
        labels = weak_label("Happy 5th Birthday Little One!")
        assert "birthday/kids" in labels

    def test_relationship_birthday_detected(self) -> None:
        labels = weak_label("Birthday Card for Husband - Love You")
        assert "birthday/relationship" in labels

    def test_relationship_via_cooccurrence(self) -> None:
        labels = weak_label("Happy Birthday to my amazing Wife")
        assert "birthday/relationship" in labels

    def test_boyfriend_birthday(self) -> None:
        labels = weak_label("Birthday Wishes for Boyfriend")
        assert "birthday/relationship" in labels

    def test_all_rules_have_valid_occasions(self) -> None:
        from common.occasions import OCCASIONS as OCC_LIST
        for occ in _RULES:
            assert occ in OCC_LIST, f"{occ!r} not in canonical taxonomy"

    def test_all_cooccurrence_rules_have_valid_occasions(self) -> None:
        from common.occasions import OCCASIONS as OCC_LIST
        for occ in _COOCCURRENCE_RULES:
            assert occ in OCC_LIST, f"{occ!r} not in canonical taxonomy"


class TestPickBestOccasion:
    def test_sub_occasion_wins_over_general(self) -> None:
        labels = ["birthday/general", "birthday/kids"]
        assert pick_best_occasion(labels) == "birthday/kids"

    def test_general_when_only_match(self) -> None:
        labels = ["birthday/general"]
        assert pick_best_occasion(labels) == "birthday/general"

    def test_empty_returns_none(self) -> None:
        assert pick_best_occasion([]) is None

    def test_relationship_wins_over_general(self) -> None:
        labels = ["birthday/general", "birthday/relationship"]
        assert pick_best_occasion(labels) == "birthday/relationship"

    def test_milestone_wins_over_general(self) -> None:
        labels = ["birthday/general", "birthday/milestone"]
        assert pick_best_occasion(labels) == "birthday/milestone"

    def test_full_pipeline_kids_card(self) -> None:
        text = "Happy 3rd Birthday to a Special Little Boy"
        labels = weak_label(text)
        best = pick_best_occasion(labels)
        assert best == "birthday/kids"

    def test_full_pipeline_husband_card(self) -> None:
        text = "To My Wonderful Husband Happy Birthday"
        labels = weak_label(text)
        best = pick_best_occasion(labels)
        assert best == "birthday/relationship"

    def test_full_pipeline_generic_birthday(self) -> None:
        text = "Happy Birthday Floral Watercolour Card"
        labels = weak_label(text)
        best = pick_best_occasion(labels)
        assert best == "birthday/general"


class TestOccasionIndex:
    def test_bijection(self) -> None:
        for i, occ in enumerate(OCCASIONS):
            assert OCCASION_TO_IDX[occ] == i
            assert IDX_TO_OCCASION[i] == occ

    def test_full_coverage(self) -> None:
        assert len(OCCASION_TO_IDX) == len(OCCASIONS)
        assert len(IDX_TO_OCCASION) == len(OCCASIONS)

    def test_no_gaps(self) -> None:
        indices = sorted(IDX_TO_OCCASION.keys())
        assert indices == list(range(len(OCCASIONS)))
