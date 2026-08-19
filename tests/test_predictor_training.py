"""Tests for predictor training mechanics.

These cover the parts that fail quietly: a checkpoint that cannot be reloaded,
a masked loss that credits the wrong samples, and a seller split that leaks.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.predictor.architecture import (
    HEAD_NAMES,
    PredictorConfig,
    SaleabilityPredictor,
    head_loss_weights,
)
from models.predictor.dataset import (
    PredictorDataset,
    SplitConfig,
    _build_targets,
    group_keys,
    split_by_seller,
)
from models.predictor.infer import PredictorRunner
from models.predictor.train import masked_mse


def _row(**over):
    base = {
        "clip_embedding": list(np.zeros(768)),
        "extracted_text": "happy birthday",
        "occasion_idx": 0,
        "vlm_raw": {
            "occasion_fit": 0.6,
            "aesthetic": 0.5,
            "emotional_resonance": 0.4,
            "distinctiveness": 0.3,
            "purchase_intent": 0.7,
        },
    }
    base.update(over)
    return base


class TestCheckpointRoundTrip:
    def test_swept_architecture_reloads(self):
        """The sweep varies trunk/head width; defaults would mismatch shapes."""
        arch = PredictorConfig(trunk_hidden=256, head_hidden=64, occasion_emb_dim=16)
        model = SaleabilityPredictor(arch)
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.ckpt"
            torch.save(
                {"state_dict": model.state_dict(), "config": {}, "arch": asdict(arch)},
                ckpt,
            )
            runner = PredictorRunner(ckpt)
        assert runner.model.cfg.trunk_hidden == 256
        assert runner.model.cfg.head_hidden == 64

    def test_standardiser_statistics_survive_the_round_trip(self):
        """Buffers must travel with the checkpoint.

        They are fitted on the training split. If they reset to identity on
        load, inference feeds the trunk unscaled features while the weights
        expect scaled ones, and nothing about the failure is visible.
        """
        arch = PredictorConfig(standardise=True)
        model = SaleabilityPredictor(arch)
        mean = torch.full_like(model.feat_mean, 0.25)
        std = torch.full_like(model.feat_std, 4.0)
        model.set_feature_stats(mean, std)

        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.ckpt"
            torch.save(
                {"state_dict": model.state_dict(), "config": {}, "arch": asdict(arch)},
                ckpt,
            )
            runner = PredictorRunner(ckpt)

        assert torch.allclose(runner.model.feat_mean.cpu(), mean)
        assert torch.allclose(runner.model.feat_std.cpu(), std)

    def test_zero_variance_dimension_does_not_produce_infinities(self):
        """A dimension constant across the corpus has std 0; clamp it."""
        model = SaleabilityPredictor(PredictorConfig(standardise=True))
        model.set_feature_stats(
            torch.zeros_like(model.feat_mean), torch.zeros_like(model.feat_std)
        )
        out = model(
            torch.randn(2, 768), torch.randn(2, 768), torch.zeros(2, dtype=torch.long)
        )
        assert all(torch.isfinite(v).all() for v in out.values())

    def test_legacy_checkpoint_without_arch_still_loads(self):
        model = SaleabilityPredictor(PredictorConfig())
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.ckpt"
            torch.save({"state_dict": model.state_dict(), "config": {}}, ckpt)
            runner = PredictorRunner(ckpt)
        assert runner.model.cfg.trunk_hidden == PredictorConfig().trunk_hidden


class TestMaskedLoss:
    def _preds(self, value: float, n: int = 4):
        return {name: torch.full((n,), value) for name in HEAD_NAMES}

    def test_masked_samples_contribute_nothing(self):
        targets = torch.zeros(4, len(HEAD_NAMES))
        full = torch.ones(4, len(HEAD_NAMES))
        none = torch.zeros(4, len(HEAD_NAMES))
        none[:, 0] = 1.0  # only the first head is labelled

        w = {n: 1.0 for n in HEAD_NAMES}
        loss_one_head = masked_mse(self._preds(0.5), targets, none, w)
        # A head with no labels must not dilute or inflate the mean.
        assert loss_one_head == pytest.approx(0.25)
        assert masked_mse(self._preds(0.5), targets, full, w) == pytest.approx(0.25)

    def test_all_masked_returns_zero_and_stays_differentiable(self):
        preds = {n: torch.full((3,), 0.5, requires_grad=True) for n in HEAD_NAMES}
        loss = masked_mse(
            preds, torch.zeros(3, len(HEAD_NAMES)), torch.zeros(3, len(HEAD_NAMES)),
            {n: 1.0 for n in HEAD_NAMES},
        )
        assert float(loss) == 0.0
        loss.backward()  # must not raise

    def test_head_weight_increases_influence(self):
        """purchase_intent is upweighted to offset its smaller label set."""
        targets = torch.zeros(2, len(HEAD_NAMES))
        mask = torch.ones(2, len(HEAD_NAMES))
        preds = {n: torch.zeros(2) for n in HEAD_NAMES}
        preds["purchase_intent"] = torch.ones(2)  # only this head is wrong

        flat = masked_mse(preds, targets, mask, {n: 1.0 for n in HEAD_NAMES})
        upweighted = masked_mse(preds, targets, mask, head_loss_weights(2.0))
        assert float(upweighted) > float(flat)


class TestTargets:
    def test_all_five_heads_labelled_from_raw(self):
        targets, mask = _build_targets(pd.Series(_row()))
        assert mask == [1.0] * 5
        assert targets[HEAD_NAMES.index("purchase_intent")] == pytest.approx(0.7)

    def test_missing_dimension_is_masked_not_defaulted(self):
        raw = {"occasion_fit": 0.6}
        _, mask = _build_targets(pd.Series(_row(vlm_raw=raw)))
        assert mask[HEAD_NAMES.index("aesthetic")] == 0.0
        assert mask[HEAD_NAMES.index("occasion_fit")] == 1.0

    def test_targets_clamped_to_unit_range(self):
        raw = {"aesthetic": 1.4, "occasion_fit": -0.3}
        targets, _ = _build_targets(pd.Series(_row(vlm_raw=raw)))
        assert targets[HEAD_NAMES.index("aesthetic")] == 1.0
        assert targets[HEAD_NAMES.index("occasion_fit")] == 0.0


class TestSellerSplit:
    def test_no_seller_appears_in_two_splits(self):
        df = pd.DataFrame(
            {"seller_id": [f"s{i // 5}" for i in range(100)], "x": range(100)}
        )
        splits = split_by_seller(df, SplitConfig(seed=1))
        seen = [set(s["seller_id"]) for s in splits.values()]
        assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])

    def test_split_is_deterministic_for_a_seed(self):
        df = pd.DataFrame({"seller_id": [f"s{i // 5}" for i in range(100)]})
        a = split_by_seller(df, SplitConfig(seed=7))["train"]["seller_id"].tolist()
        b = split_by_seller(df, SplitConfig(seed=7))["train"]["seller_id"].tolist()
        assert a == b

    def test_null_sellers_become_distinct(self):
        """NULL seller_id must not collapse every such listing into one group."""
        df = pd.DataFrame(
            {"seller_id": [None] * 20, "listing_id": [f"l{i}" for i in range(20)]}
        )
        splits = split_by_seller(df, SplitConfig(seed=3))
        assert sum(len(s) for s in splits.values()) == 20
        assert all(len(s) > 0 for s in splits.values())

    def test_split_survives_a_reordered_query(self):
        """The same cards in a different row order must split identically.

        The query has no ORDER BY, so Postgres row order is free to vary
        between runs. When the split depended on it, two runs twenty minutes
        apart produced train/val sizes of 1715/376 and 1724/373 from identical
        data, and every cross-run comparison was against a different test set.
        """
        df = pd.DataFrame(
            {
                "listing_id": [f"l{i}" for i in range(120)],
                "seller_id": [f"s{i // 4}" if i % 3 else None for i in range(120)],
            }
        )
        shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)

        a = split_by_seller(df, SplitConfig(seed=42))
        b = split_by_seller(shuffled, SplitConfig(seed=42))

        for name in ("train", "val", "test"):
            assert set(a[name]["listing_id"]) == set(b[name]["listing_id"]), name

    def test_duplicate_cluster_never_straddles_the_split(self):
        """Near-duplicates must land on one side, even with no seller.

        Deduplication exists partly to stop near-identical images appearing on
        both sides of the split. Seller grouping cannot enforce that for the
        44% of listings with no seller_id: each was its own group, so two
        colourways of one design were free to separate. The group key now
        merges duplicate clusters as well as sellers.
        """
        df = pd.DataFrame(
            {
                "listing_id": [f"l{i}" for i in range(120)],
                # No seller anywhere: the case seller grouping cannot cover.
                "seller_id": [None] * 120,
                # Pairs of colourways, 60 clusters of two.
                "duplicate_cluster_id": [f"c{i // 2}" for i in range(120)],
            }
        )
        splits = split_by_seller(df, SplitConfig(seed=11))

        where = {}
        for name, rows in splits.items():
            for cluster in rows["duplicate_cluster_id"]:
                where.setdefault(cluster, set()).add(name)
        straddling = {c: s for c, s in where.items() if len(s) > 1}
        assert not straddling, f"clusters split across sides: {straddling}"
        assert sum(len(s) for s in splits.values()) == 120

    def test_cluster_grouping_merges_across_sellers(self):
        """A cluster spanning two sellers pulls both into one group."""
        df = pd.DataFrame(
            {
                "listing_id": ["a", "b", "c", "d"],
                "seller_id": ["s1", "s2", "s3", "s4"],
                "duplicate_cluster_id": ["k", "k", None, None],
            }
        )
        keys = group_keys(df)
        assert keys.iloc[0] == keys.iloc[1]
        assert len({keys.iloc[0], keys.iloc[2], keys.iloc[3]}) == 3

    def test_sellerless_listing_keeps_its_split_when_the_pool_grows(self):
        """Adding cards must not reshuffle the ones already assigned.

        Positional synthetic names renamed every sellerless listing whenever
        the pool changed, so labelling more cards silently moved existing ones
        across the train/test boundary.
        """
        base = pd.DataFrame(
            {"listing_id": [f"l{i}" for i in range(80)], "seller_id": [None] * 80}
        )
        # New rows ahead of the old ones: the query has no ORDER BY, so newly
        # labelled cards can appear anywhere, and positional naming then shifts
        # every existing listing's synthetic seller by the number of new rows.
        grown = pd.concat(
            [
                pd.DataFrame(
                    {"listing_id": [f"new{i}" for i in range(20)], "seller_id": [None] * 20}
                ),
                base,
            ],
            ignore_index=True,
        )

        before = split_by_seller(base, SplitConfig(seed=5))["test"]["listing_id"]
        after = split_by_seller(grown, SplitConfig(seed=5))
        moved = set(before) - set(after["test"]["listing_id"])

        # Growing the pool shifts split boundaries, so a few crossings are
        # expected; wholesale reassignment is the failure being guarded.
        assert len(moved) < 0.25 * len(before), f"{len(moved)}/{len(before)} moved"


class TestTextEmbedding:
    def test_embedded_in_one_batch(self):
        df = pd.DataFrame([_row(), _row(extracted_text=None), _row()])
        calls = []

        def embedder(texts):
            calls.append(len(texts))
            return np.ones((len(texts), 768), dtype=np.float32)

        ds = PredictorDataset(df, text_embedder=embedder)
        assert calls == [2], "one call covering only the rows that have text"
        assert ds[1]["text_emb"].abs().sum() == 0
        assert ds[0]["text_emb"].abs().sum() > 0

    def test_zeros_without_an_embedder(self):
        ds = PredictorDataset(pd.DataFrame([_row()]), text_embedder=None)
        assert ds[0]["text_emb"].shape == (768,)
        assert ds[0]["text_emb"].abs().sum() == 0
