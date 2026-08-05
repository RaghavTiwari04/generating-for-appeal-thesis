"""Tests for the statistics and image handling in eval.llm_system_eval.

Every headline number in the results chapter comes out of these functions, so
they are pinned against hand-checkable cases rather than only against each
other. Two of the tests exist because the corresponding mistakes were made:
`test_normalises_oversized_image_to_judge_long_edge` covers the resolution
asymmetry between generated and scraped cards, and
`test_mean_diff_may_sit_outside_margin_while_equivalent` covers reading the
difference in means as if it were the quantity the rank test bounds.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval.llm_system_eval import (
    CONDITIONS,
    JUDGE_LONG_EDGE,
    _bootstrap_ci,
    _hodges_lehmann,
    _load_image,
    _rank_biserial,
    _tost_equivalence,
    pairwise_holm,
    per_occasion_pairwise,
)


def _png_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (128, 100, 90)).save(buf, format="PNG")
    return buf.getvalue()


def _frame(**by_condition: list[float]) -> pd.DataFrame:
    rows = [
        {"condition": cond, "purchase_intent": v, "occasion": "birthday/general"}
        for cond, values in by_condition.items()
        for v in values
    ]
    return pd.DataFrame(rows)


class TestHodgesLehmann:
    def test_constant_offset_is_recovered_exactly(self):
        a = pd.Series([0.1, 0.2, 0.3, 0.4])
        assert _hodges_lehmann(a + 0.05, a) == pytest.approx(0.05)

    def test_is_zero_for_identical_samples(self):
        a = pd.Series([0.3, 0.5, 0.7])
        assert _hodges_lehmann(a, a) == pytest.approx(0.0)

    def test_sign_follows_argument_order(self):
        hi, lo = pd.Series([0.8, 0.9]), pd.Series([0.1, 0.2])
        assert _hodges_lehmann(hi, lo) > 0
        assert _hodges_lehmann(lo, hi) == pytest.approx(-_hodges_lehmann(hi, lo))

    def test_median_of_pairwise_differences_not_difference_of_medians(self):
        # These disagree whenever the samples are skewed, and the rank test is
        # about the former.
        a = pd.Series([0.0, 0.0, 10.0])
        b = pd.Series([1.0, 2.0, 3.0])
        expected = float(np.median(np.subtract.outer(a.to_numpy(), b.to_numpy())))
        assert _hodges_lehmann(a, b) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("a", "b"), [([], [1.0]), ([1.0], []), ([], [])]
    )
    def test_empty_input_is_nan_not_zero(self, a, b):
        assert np.isnan(_hodges_lehmann(pd.Series(a, dtype=float), pd.Series(b, dtype=float)))


class TestRankBiserial:
    def test_identical_samples_give_zero_effect(self):
        a = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _rank_biserial(a, a) == pytest.approx(0.0)

    def test_complete_separation_saturates(self):
        hi = pd.Series([0.90, 0.95, 0.99])
        lo = pd.Series([0.10, 0.15, 0.20])
        assert _rank_biserial(hi, lo) == pytest.approx(-1.0)
        assert _rank_biserial(lo, hi) == pytest.approx(1.0)

    def test_sign_is_positive_when_the_first_sample_is_lower(self):
        # The opposite of the _hodges_lehmann convention, and of the usual
        # definition of rank-biserial. Pinned because the reported effect sizes
        # depend on it: A vs B is +0.482 with A the lower condition, and C vs D
        # is -0.259 with C the higher one. Flipping this silently reverses the
        # sign of every effect size in the results chapter.
        lo, hi = pd.Series([0.1, 0.2, 0.3]), pd.Series([0.7, 0.8, 0.9])
        assert _rank_biserial(lo, hi) > 0
        assert _hodges_lehmann(lo, hi) < 0

    def test_stays_within_bounds_on_random_samples(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = pd.Series(rng.normal(0.6, 0.1, 30))
            b = pd.Series(rng.normal(0.65, 0.1, 30))
            assert -1.0 <= _rank_biserial(a, b) <= 1.0


class TestTOST:
    def test_identical_samples_are_equivalent(self):
        a = pd.Series(np.linspace(0.60, 0.75, 40))
        out = _tost_equivalence(a, a.copy(), delta=0.02)
        assert out["equivalent"] is True
        assert out["hodges_lehmann"] == pytest.approx(0.0)

    def test_clearly_separated_samples_are_not_equivalent(self):
        a = pd.Series(np.linspace(0.10, 0.20, 40))
        b = pd.Series(np.linspace(0.80, 0.90, 40))
        assert _tost_equivalence(a, b, delta=0.02)["equivalent"] is False

    def test_reports_the_margin_it_was_given(self):
        a = pd.Series(np.linspace(0.6, 0.7, 20))
        assert _tost_equivalence(a, a.copy(), delta=0.05)["delta"] == 0.05

    def test_widening_the_margin_cannot_make_equivalence_harder(self):
        rng = np.random.default_rng(7)
        a = pd.Series(rng.normal(0.68, 0.05, 40))
        b = pd.Series(rng.normal(0.70, 0.05, 40))
        p_narrow = _tost_equivalence(a, b, delta=0.01)["p_tost"]
        p_wide = _tost_equivalence(a, b, delta=0.10)["p_tost"]
        assert p_wide <= p_narrow

    def test_mean_diff_may_sit_outside_margin_while_equivalent(self):
        # One extreme value drags the mean past delta without moving the ranks,
        # which is the documented reason Hodges-Lehmann is what gets quoted.
        rng = np.random.default_rng(3)
        base = rng.normal(0.68, 0.01, 60)
        a = pd.Series(np.append(base, 5.0))
        b = pd.Series(np.append(base.copy(), 0.68))
        out = _tost_equivalence(a, b, delta=0.02)
        assert out["equivalent"] is True
        assert abs(out["mean_diff"]) > out["delta"]
        assert abs(out["hodges_lehmann"]) < out["delta"]


class TestBootstrapCI:
    def test_is_deterministic_for_a_fixed_seed(self):
        v = np.random.default_rng(1).normal(0.7, 0.05, 40)
        assert _bootstrap_ci(v, n_boot=500, seed=42) == _bootstrap_ci(v, n_boot=500, seed=42)

    def test_brackets_the_sample_mean(self):
        v = np.random.default_rng(2).normal(0.7, 0.05, 60)
        lo, hi = _bootstrap_ci(v, n_boot=1000, seed=42)
        assert lo < v.mean() < hi

    def test_a_wider_interval_is_returned_for_a_wider_sample(self):
        rng = np.random.default_rng(4)
        tight = _bootstrap_ci(rng.normal(0.7, 0.01, 50), n_boot=800, seed=42)
        loose = _bootstrap_ci(rng.normal(0.7, 0.20, 50), n_boot=800, seed=42)
        assert (loose[1] - loose[0]) > (tight[1] - tight[0])


class TestPairwiseHolm:
    def test_covers_every_pair_of_present_conditions(self):
        df = _frame(**{c: list(np.linspace(0.5, 0.8, 10)) for c in CONDITIONS})
        p, effects = pairwise_holm(df)
        assert len(p) == 6  # 4 conditions choose 2
        assert set(p) == set(effects)

    def test_pair_keys_follow_the_canonical_condition_order(self):
        df = _frame(**{c: list(np.linspace(0.5, 0.8, 10)) for c in CONDITIONS})
        p, _ = pairwise_holm(df)
        for key in p:
            a, b = key.split("_vs_")
            assert CONDITIONS.index(a) < CONDITIONS.index(b)

    def test_correction_never_lowers_a_p_value(self):
        rng = np.random.default_rng(5)
        df = _frame(**{
            c: list(rng.normal(m, 0.05, 25))
            for c, m in zip(CONDITIONS, [0.50, 0.68, 0.71, 0.69], strict=True)
        })
        p, _ = pairwise_holm(df)
        from scipy.stats import mannwhitneyu
        for key, corrected in p.items():
            a, b = key.split("_vs_")
            _, raw = mannwhitneyu(
                df[df.condition == a].purchase_intent,
                df[df.condition == b].purchase_intent,
                alternative="two-sided",
            )
            assert corrected >= raw - 1e-12

    def test_conditions_with_too_few_cards_are_skipped_not_crashed_on(self):
        df = _frame(A_naive_ai=[0.3, 0.4], B_pipeline_no_rerank=list(np.linspace(0.6, 0.8, 10)))
        assert pairwise_holm(df) == ({}, {})

    def test_empty_frame_returns_empty_rather_than_raising(self):
        empty = pd.DataFrame({"condition": [], "purchase_intent": []})
        assert pairwise_holm(empty) == ({}, {})


class TestPerOccasion:
    def test_splits_results_by_occasion(self):
        rows = []
        for occ in ("birthday/general", "birthday/kids"):
            for cond in ("A_naive_ai", "B_pipeline_no_rerank"):
                for v in np.linspace(0.4, 0.8, 6):
                    rows.append({"condition": cond, "purchase_intent": v, "occasion": occ})
        out = per_occasion_pairwise(pd.DataFrame(rows))
        assert set(out) == {"birthday/general", "birthday/kids"}
        assert all("A_naive_ai_vs_B_pipeline_no_rerank" in v for v in out.values())


class TestLoadImage:
    def test_normalises_oversized_image_to_judge_long_edge(self):
        # Generated cards were 2.17MP against the human reference's 0.75MP, and
        # the judge scores what it is shown.
        with patch("eval.llm_system_eval.get_object", return_value=_png_bytes(2480, 3496)):
            img = _load_image("s3://bucket/big.png")
        assert max(img.size) == JUDGE_LONG_EDGE

    def test_preserves_aspect_ratio_when_downscaling(self):
        with patch("eval.llm_system_eval.get_object", return_value=_png_bytes(2000, 1000)):
            img = _load_image("s3://bucket/wide.png")
        assert img.width == JUDGE_LONG_EDGE
        assert img.height == pytest.approx(JUDGE_LONG_EDGE // 2, abs=1)

    def test_leaves_a_smaller_image_untouched(self):
        with patch("eval.llm_system_eval.get_object", return_value=_png_bytes(600, 800)):
            img = _load_image("s3://bucket/small.png")
        assert img.size == (600, 800)

    def test_returns_none_rather_than_raising_on_a_bad_blob(self):
        with patch("eval.llm_system_eval.get_object", side_effect=OSError("gone")):
            assert _load_image("s3://bucket/missing.png") is None
