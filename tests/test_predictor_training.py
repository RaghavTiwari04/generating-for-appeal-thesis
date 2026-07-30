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
        df = pd.DataFrame({"seller_id": [None] * 20})
        splits = split_by_seller(df, SplitConfig(seed=3))
        assert sum(len(s) for s in splits.values()) == 20
        assert all(len(s) > 0 for s in splits.values())


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
