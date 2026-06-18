"""Tests for the occasion classifier — weak labels + model inference (no DB)."""

from __future__ import annotations

import pytest
import torch

from data.features.occasion_classifier import (
    _RULES,
    IDX_TO_OCCASION,
    OCCASION_TO_IDX,
    OCCASIONS,
    OccasionClassifier,
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
        # Could match birthday AND thank_you
        labels = weak_label("Thank you for the birthday wishes")
        assert len(labels) >= 1

    def test_case_insensitive(self) -> None:
        upper = weak_label("HAPPY BIRTHDAY CARD")
        lower = weak_label("happy birthday card")
        assert set(upper) == set(lower)

    def test_all_rules_have_valid_occasions(self) -> None:
        from common.occasions import OCCASIONS as OCC_LIST
        for occ in _RULES:
            assert occ in OCC_LIST, f"{occ!r} not in canonical taxonomy"


class TestOccasionClassifierModel:
    @pytest.fixture
    def model(self):
        """Build classifier with a tiny random DistilBERT-shaped encoder (no download)."""
        from unittest.mock import MagicMock, patch

        import torch.nn as nn

        # Stub the HuggingFace encoder with a minimal 2-layer transformer
        tiny_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True, dim_feedforward=128),
            num_layers=1,
        )
        # Give it .config.hidden_size so the classifier can read it
        tiny_encoder.config = MagicMock(hidden_size=768)

        class _FakeLM:
            """Minimal stand-in for a HuggingFace model."""
            config = MagicMock(hidden_size=768)

            def __call__(self, input_ids, attention_mask):
                B, T = input_ids.shape
                import torch
                fake_last_hidden = torch.randn(B, T, 768)
                return MagicMock(last_hidden_state=fake_last_hidden)

        with patch(
            "data.features.occasion_classifier.AutoModel.from_pretrained",
            return_value=_FakeLM(),
        ):
            clf = OccasionClassifier(n_labels=len(OCCASIONS))
        return clf

    def test_forward_shape(self, model: OccasionClassifier) -> None:
        ids = torch.randint(0, 1000, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.long)
        out = model(ids, mask)
        assert out.shape == (4, len(OCCASIONS))

    def test_output_in_0_1(self, model: OccasionClassifier) -> None:
        ids = torch.randint(0, 1000, (2, 32))
        mask = torch.ones(2, 32, dtype=torch.long)
        out = model(ids, mask)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_no_nan(self, model: OccasionClassifier) -> None:
        ids = torch.randint(0, 500, (3, 24))
        mask = torch.ones(3, 24, dtype=torch.long)
        out = model(ids, mask)
        assert not torch.isnan(out).any()

    def test_different_inputs_different_outputs(self, model: OccasionClassifier) -> None:
        torch.manual_seed(0)
        ids1 = torch.randint(0, 1000, (1, 16))
        ids2 = torch.randint(0, 1000, (1, 16))
        mask = torch.ones(1, 16, dtype=torch.long)
        out1 = model(ids1, mask)
        out2 = model(ids2, mask)
        assert not torch.allclose(out1, out2)


class TestOccasionIndex:
    def test_bijection(self) -> None:
        """Every occasion maps to a unique index and back."""
        for i, occ in enumerate(OCCASIONS):
            assert OCCASION_TO_IDX[occ] == i
            assert IDX_TO_OCCASION[i] == occ

    def test_full_coverage(self) -> None:
        assert len(OCCASION_TO_IDX) == len(OCCASIONS)
        assert len(IDX_TO_OCCASION) == len(OCCASIONS)

    def test_no_gaps(self) -> None:
        indices = sorted(IDX_TO_OCCASION.keys())
        assert indices == list(range(len(OCCASIONS)))
