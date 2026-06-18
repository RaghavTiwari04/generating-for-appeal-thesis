"""Unit tests for predictor architecture (no GPU / DB needed)."""

from __future__ import annotations

import torch

from models.predictor.architecture import (
    HEAD_NAMES,
    PredictorConfig,
    SaleabilityPredictor,
    head_loss_weights,
)


def _dummy_batch(bs: int = 4, cfg: PredictorConfig | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or PredictorConfig()
    return {
        "image_emb": torch.randn(bs, cfg.image_dim),
        "text_emb": torch.randn(bs, cfg.text_dim),
        "occasion_idx": torch.randint(0, cfg.occasion_vocab, (bs,)),
        "price_rel": torch.randn(bs, 1),
    }


def test_forward_output_keys() -> None:
    model = SaleabilityPredictor()
    batch = _dummy_batch()
    out = model(**batch)
    assert set(out.keys()) == set(HEAD_NAMES)


def test_forward_output_range() -> None:
    model = SaleabilityPredictor()
    batch = _dummy_batch()
    out = model(**batch)
    for name, tensor in out.items():
        assert tensor.shape == (4,), f"{name} shape mismatch"
        assert (tensor >= 0.0).all() and (tensor <= 1.0).all(), f"{name} out of [0,1]"


def test_no_nan() -> None:
    model = SaleabilityPredictor()
    batch = _dummy_batch()
    out = model(**batch)
    for name, tensor in out.items():
        assert not torch.isnan(tensor).any(), f"NaN in {name}"


def test_head_loss_weights_purchase_intent_doubled() -> None:
    weights = head_loss_weights(purchase_intent_factor=2.0)
    assert weights["purchase_intent"] == 2.0
    for k, v in weights.items():
        if k != "purchase_intent":
            assert v == 1.0


def test_different_seeds_different_output() -> None:
    torch.manual_seed(0)
    model = SaleabilityPredictor()
    b1 = _dummy_batch(bs=2)
    b2 = _dummy_batch(bs=2)
    o1 = model(**b1)["purchase_intent"]
    o2 = model(**b2)["purchase_intent"]
    # Different random inputs should produce different outputs (almost certainly)
    assert not torch.allclose(o1, o2)
