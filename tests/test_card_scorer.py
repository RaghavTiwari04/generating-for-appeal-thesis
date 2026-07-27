"""Tests for the scoring instrument's pure logic.

The SSR mapping and the rating extraction decide every label in the corpus, and
both fail quietly when wrong — a mis-parsed rating becomes a confident score,
and a mis-normalised PMF shifts every purchase-intent estimate.
"""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from scoring.card_scorer import (
    SSR_REFERENCE_SETS,
    CardScorer,
    extract_rating,
    openrouter_route,
    scale_pmf,
    similarities_to_pmf,
    ssr_score,
)


class TestExtractRating:
    def test_double_bracket_preferred(self):
        assert extract_rating("Good card. Rating: [[7]]") == 7.0

    def test_single_bracket_fallback(self):
        assert extract_rating("Rating: [8]") == 8.0

    def test_decimal(self):
        assert extract_rating("[[7.5]]") == 7.5

    def test_missing(self):
        assert extract_rating("no rating at all") is None
        assert extract_rating("") is None

    @pytest.mark.parametrize("text", ["published in [2024]", "[[0]]", "[[11]]"])
    def test_out_of_range_rejected(self, text):
        """Clamping would turn a bracketed year into a confident 10."""
        assert extract_rating(text) is None


class TestSSRMapping:
    def test_pmf_normalised(self):
        pmf = similarities_to_pmf(np.array([0.61, 0.64, 0.70, 0.66, 0.62]))
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()

    def test_min_subtraction_zeroes_the_lowest(self):
        pmf = similarities_to_pmf(np.array([0.61, 0.64, 0.70, 0.66, 0.62]))
        assert pmf[0] == pytest.approx(0.0)

    def test_uniform_similarities_give_uniform_pmf(self):
        """All-equal similarities subtract to all-zero; guard the divide."""
        pmf = similarities_to_pmf(np.full(5, 0.5))
        assert pmf == pytest.approx(np.full(5, 0.2))

    def test_epsilon_lifts_the_minimum(self):
        pmf = similarities_to_pmf(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), epsilon=0.1)
        assert pmf[0] > 0
        assert pmf.sum() == pytest.approx(1.0)

    def test_scale_pmf_identity_at_one(self):
        pmf = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        assert scale_pmf(pmf, 1.0) is pmf

    def test_scale_pmf_one_hot_at_zero(self):
        pmf = np.array([0.1, 0.2, 0.4, 0.25, 0.05])
        out = scale_pmf(pmf, 0.0)
        assert out.tolist() == [0, 0, 1, 0, 0]

    def test_score_spans_zero_to_one(self):
        """Likert 1..5 maps onto 0..1, so an extreme PMF hits the endpoints."""
        anchors = np.eye(5)[:, :5]
        with patch("scoring.card_scorer._embed", return_value=np.eye(5)[:1]), patch(
            "scoring.card_scorer.anchor_embeddings",
            return_value=anchors.reshape(1, 5, 5),
        ):
            out = ssr_score("irrelevant")
        assert 0.0 <= out["score"] <= 1.0
        assert out["likert"] == pytest.approx(1 + 4 * out["score"])
        assert len(out["pmf"]) == 5

    def test_reference_sets_are_five_point(self):
        """The expectation is taken over 1..5, so every set must have 5 rungs."""
        assert all(len(s) == 5 for s in SSR_REFERENCE_SETS)


class TestRoute:
    def test_none_and_blank(self):
        assert openrouter_route(None) is None
        assert openrouter_route("  ") is None

    def test_bare_name_pins_without_fallback(self):
        assert openrouter_route("DeepInfra") == {
            "order": ["DeepInfra"],
            "allow_fallbacks": False,
        }

    def test_json_passed_through(self):
        assert openrouter_route('{"quantizations": ["bf16"]}') == {
            "quantizations": ["bf16"]
        }


class TestScorer:
    def _scorer_over_two_cards(self, second_card_fails: str):
        """Score two cards with one scorer; the named dim fails on the second."""
        seen = {"cards": 0}

        def fake_call(b64, system, user, **kw):
            if "Evaluate" not in user:
                return "I would probably buy this one"
            dim_failed = second_card_fails.replace("_", " ").upper() in user
            if dim_failed and seen["cards"] > 0:
                return ""
            return "reasonable [[7]]"

        with patch("scoring.card_scorer.call_vlm", side_effect=fake_call), patch(
            "scoring.card_scorer.ssr_score",
            return_value={"pmf": [0.2] * 5, "likert": 3.0, "score": 0.5},
        ):
            scorer = CardScorer()
            image = Image.new("RGB", (80, 110))
            first = scorer.score(image, occasion="birthday/general")
            seen["cards"] = 1
            second = scorer.score(image, occasion="birthday/general")
        return first, second

    def test_failed_dimension_is_omitted_not_defaulted(self):
        first, second = self._scorer_over_two_cards("distinctiveness")
        assert "distinctiveness" in first
        assert "distinctiveness" not in second

    def test_explanations_do_not_leak_between_cards(self):
        """A scorer is reused across a whole corpus; state must not carry over."""
        _, second = self._scorer_over_two_cards("distinctiveness")
        assert "distinctiveness" not in second["explanations"]

    def test_purchase_intent_omitted_when_every_reply_is_empty(self):
        def only_rubric(b64, system, user, **kw):
            return "reasonable [[7]]" if "Evaluate" in user else ""

        with patch("scoring.card_scorer.call_vlm", side_effect=only_rubric):
            out = CardScorer().score(Image.new("RGB", (80, 110)))
        assert "purchase_intent" not in out
        assert out["ssr_responses"] == []
        assert out["aesthetic"] == pytest.approx((7 - 1) / 9)
