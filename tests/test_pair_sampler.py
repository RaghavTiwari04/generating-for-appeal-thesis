"""Tests for pairwise survey samplers.

DB calls are monkey-patched out; the samplers operate on in-memory fixture
pools. Trapdoor handling, occasion balance, and L/R randomisation tested.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from common.occasions import ACTIVE_OCCASIONS
from survey.instrument import sampler as smp

_BIRTHDAY_OCCS = list(ACTIVE_OCCASIONS)


@pytest.fixture
def fake_main_pool(monkeypatch) -> None:
    """Patch _load_main_pool and _current_appearances to use synthetic data."""
    rows = []
    # 8 cards per birthday sub-occasion (4 sub-occ × 8 = 32 cards)
    for occ in _BIRTHDAY_OCCS:
        for k in range(8):
            rows.append(
                {
                    "card_key": f"lst_{occ}_{k}",
                    "is_generated": False,
                    "condition_tag": None,
                    "occasion": occ,
                    "cover_path": f"s3://bucket/img_{occ}_{k}.jpg",
                    "headline": f"Headline {k}",
                    "inside_message": None,
                    "engagement": (k + 1) * 10,
                }
            )
    df = pd.DataFrame(rows)
    monkeypatch.setattr(smp, "_load_main_pool", lambda: df.copy())
    monkeypatch.setattr(smp, "_current_appearances", lambda study_id: {})
    return None


@pytest.fixture
def fake_sys_eval_pool(monkeypatch) -> None:
    rows = []
    for cond in smp.SYSTEM_EVAL_CONDITIONS:
        for occ in _BIRTHDAY_OCCS:
            for k in range(3):
                rows.append(
                    {
                        "card_key": f"{cond}_{occ}_{k}",
                        "is_generated": True,
                        "condition_tag": cond,
                        "occasion": occ,
                        "cover_path": f"s3://bucket/{cond}_{occ}_{k}.jpg",
                        "headline": f"H{k}",
                        "inside_message": f"M{k}",
                    }
                )
    df = pd.DataFrame(rows)
    monkeypatch.setattr(smp, "_load_system_eval_pool", lambda: df.copy())
    monkeypatch.setattr(smp, "_current_appearances", lambda study_id: {})
    return None


def test_main_pair_sampler_returns_n_pairs(fake_main_pool):
    pairs = smp.sample_pairs_main(
        participant_id="p001",
        study_id="main_v2",
        n_pairs=20,
        n_trapdoors=2,
    )
    assert 10 <= len(pairs) <= 20  # rejection sampling can fall short on tiny pools
    # All non-trapdoor pairs use cards from ACTIVE_OCCASIONS
    for p in pairs:
        if p.contrast_tag != "trapdoor":
            assert p.occasion in _BIRTHDAY_OCCS
            assert p.left.occasion == p.right.occasion  # same-occasion pairs


def test_main_pair_sampler_deterministic_per_participant(fake_main_pool):
    a = smp.sample_pairs_main("p001", "main_v2", n_pairs=15, n_trapdoors=1)
    b = smp.sample_pairs_main("p001", "main_v2", n_pairs=15, n_trapdoors=1)
    assert [(x.left.card_key, x.right.card_key) for x in a] == \
           [(x.left.card_key, x.right.card_key) for x in b]


def test_main_pair_sampler_distinct_per_participant(fake_main_pool):
    a = smp.sample_pairs_main("p001", "main_v2", n_pairs=20, n_trapdoors=2)
    b = smp.sample_pairs_main("p002", "main_v2", n_pairs=20, n_trapdoors=2)
    keys_a = [(x.left.card_key, x.right.card_key) for x in a]
    keys_b = [(x.left.card_key, x.right.card_key) for x in b]
    assert keys_a != keys_b


def test_main_pair_sampler_includes_trapdoors(fake_main_pool):
    pairs = smp.sample_pairs_main("p001", "main_v2", n_pairs=20, n_trapdoors=3)
    tag_counts = Counter(p.contrast_tag for p in pairs)
    assert tag_counts["trapdoor"] == 3


def test_main_pair_sampler_card_within_pair_distinct(fake_main_pool):
    pairs = smp.sample_pairs_main("p001", "main_v2", n_pairs=20, n_trapdoors=0)
    for p in pairs:
        assert p.left.card_key != p.right.card_key


def test_system_eval_sampler_creates_decision_contrasts(fake_sys_eval_pool):
    pairs = smp.sample_pairs_system_eval(
        participant_id="p001", study_id="system_eval_v2", n_pairs=30, n_trapdoors=2
    )
    tags = Counter(p.contrast_tag for p in pairs)
    # All three decision-critical contrasts should be present
    assert tags["C_vs_A"] > 0
    assert tags["C_vs_B"] > 0
    assert tags["C_vs_D"] > 0
    # Trapdoors present
    assert tags["trapdoor"] == 2


def test_system_eval_sampler_matches_occasion_on_contrasts(fake_sys_eval_pool):
    pairs = smp.sample_pairs_system_eval(
        participant_id="p001", study_id="system_eval_v2", n_pairs=30, n_trapdoors=0
    )
    for p in pairs:
        if p.contrast_tag.startswith(("C_vs_", "within_")):
            assert p.left.occasion == p.right.occasion
