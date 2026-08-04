"""Mocked end-to-end pipeline integration tests.

No GPU, no DB, no LLM API calls required. All external calls are patched.
Tests that the orchestration logic (brief → image → layout → message → rerank)
connects correctly and data flows through without errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from generation.brief.schema import Brief
from generation.layout.compose import ComposedCard
from generation.message.generate import InsideMessage
from models.predictor.architecture import HEAD_NAMES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_brief() -> Brief:
    return Brief(
        concept="Wildflowers and tea",
        headline="Thanks For Everything, Mum",
        inside_message="Happy Birthday, Mum.",
        # v2 briefs no longer forbid text: the image model letters the headline
        # itself and the overlay is the fallback.
        visual_prompt="Watercolour wildflowers, uncluttered band across the top",
        negative_prompt="photorealistic, deformed hands",
        style_tags=["watercolour", "illustrated"],
        target_price_band="premium",
    )


@pytest.fixture
def dummy_cover() -> Image.Image:
    return Image.new("RGB", (1024, 1024), color=(200, 180, 160))


@pytest.fixture
def dummy_scores() -> dict[str, float]:
    return {name: 0.5 for name in HEAD_NAMES} | {"purchase_intent_calibrated": 0.72, "saleability_calibrated": 0.72}


# ---------------------------------------------------------------------------
# Brief generator
# ---------------------------------------------------------------------------

class TestBriefGenerator:
    def test_generate_brief_calls_llm_and_parses(self, dummy_brief: Brief) -> None:
        with patch("common.llm.call_llm", return_value=json.dumps(dummy_brief.model_dump())):
            with patch("generation.brief.market_signals.gather", return_value=MagicMock(
                top_tropes=[], coverage_gaps=[], longevity_caution="avoid dated refs"
            )):
                from generation.brief.generate import generate_brief
                result = generate_brief({"occasion": "birthday/general", "tone": "warm-humorous"})
        assert isinstance(result, Brief)
        assert result.headline == dummy_brief.headline

    def test_brief_validate_price_band(self, dummy_brief: Brief) -> None:
        assert dummy_brief.target_price_band in ("budget", "standard", "premium", "luxury")

    def test_brief_style_tags_capped(self, dummy_brief: Brief) -> None:
        assert len(dummy_brief.style_tags) <= 4


# ---------------------------------------------------------------------------
# Layout composer (no font files needed — expect FileNotFoundError, not crash)
# ---------------------------------------------------------------------------

class TestLayoutCompose:
    def test_mask_bbox_within_image(self, dummy_cover: Image.Image) -> None:
        from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
        spec = LayoutMaskSpec(width=1024, height=1024)
        _mask, bbox = build_headline_mask(spec)
        x0, y0, x1, y1 = bbox
        assert 0 <= x0 < x1 <= 1024
        assert 0 <= y0 < y1 <= 1024

    def test_compose_raises_without_fonts(self, dummy_cover: Image.Image) -> None:
        from generation.layout.compose import compose
        with pytest.raises((FileNotFoundError, RuntimeError)):
            compose(dummy_cover, headline="Happy Birthday!", tone="warm-humorous", style_tags=["watercolour"])


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    def _make_candidates(self, n: int) -> list:
        from pipeline.rerank import Candidate
        return [
            Candidate(
                image=Image.new("RGB", (64, 64), (i * 30 % 255, 0, 0)),
                headline=f"Headline {i}",
                inside_message=f"Message {i}",
                brief={},
                occasion="birthday/general",
                seed=i,
            )
            for i in range(n)
        ]

    def test_rerank_orders_by_saleability(self, dummy_scores: dict) -> None:
        from pipeline.rerank import rerank

        candidates = self._make_candidates(4)
        # Assign varying saleability to each candidate manually
        scores_list = [
            {**dummy_scores, "saleability_calibrated": 0.9},
            {**dummy_scores, "saleability_calibrated": 0.3},
            {**dummy_scores, "saleability_calibrated": 0.7},
            {**dummy_scores, "saleability_calibrated": 0.5},
        ]
        mock_predictor = MagicMock()
        mock_predictor.score.return_value = scores_list
        mock_embedder = MagicMock()
        mock_embedder.embed_images.return_value = np.zeros((4, 768), dtype=np.float32)
        mock_embedder.embed_texts.return_value = np.zeros((4, 768), dtype=np.float32)

        ranked = rerank(candidates, predictor=mock_predictor, embedder=mock_embedder)
        saleabilities = [c.scores["saleability_calibrated"] for c in ranked]
        assert saleabilities == sorted(saleabilities, reverse=True)

    def test_rerank_top_k(self, dummy_scores: dict) -> None:
        from pipeline.rerank import rerank
        candidates = self._make_candidates(8)
        scores_list = [{**dummy_scores, "saleability_calibrated": float(i) / 8} for i in range(8)]
        mock_predictor = MagicMock()
        mock_predictor.score.return_value = scores_list
        mock_embedder = MagicMock()
        mock_embedder.embed_images.return_value = np.zeros((8, 768), dtype=np.float32)
        mock_embedder.embed_texts.return_value = np.zeros((8, 768), dtype=np.float32)

        ranked = rerank(candidates, predictor=mock_predictor, embedder=mock_embedder, top_k=3)
        assert len(ranked) == 3

    def test_rerank_empty_input(self) -> None:
        from pipeline.rerank import rerank
        result = rerank([], predictor=MagicMock(), embedder=MagicMock())
        assert result == []

    def test_rerank_llm_calls_scorer_and_sorts(self) -> None:
        from pipeline.rerank import rerank_llm

        candidates = self._make_candidates(3)
        mock_scores = [
            {"occasion_fit": 0.7, "aesthetic": 0.8, "emotional_resonance": 0.6,
             "distinctiveness": 0.5, "purchase_intent": 0.6, "purchase_intent_calibrated": 0.6},
            {"occasion_fit": 0.9, "aesthetic": 0.9, "emotional_resonance": 0.8,
             "distinctiveness": 0.8, "purchase_intent": 0.9, "purchase_intent_calibrated": 0.9},
            {"occasion_fit": 0.3, "aesthetic": 0.4, "emotional_resonance": 0.3,
             "distinctiveness": 0.2, "purchase_intent": 0.3, "purchase_intent_calibrated": 0.3},
        ]
        mock_scorer = MagicMock()
        mock_scorer.score.side_effect = mock_scores
        mock_scorer.provider = "anthropic"
        mock_scorer.model = "test-model"

        with patch("scoring.CardScorer", return_value=mock_scorer):
            ranked = rerank_llm(candidates, top_k=2)

        assert len(ranked) == 2
        assert mock_scorer.score.call_count == 3
        saleabilities = [c.scores["saleability_calibrated"] for c in ranked]
        assert saleabilities == sorted(saleabilities, reverse=True)

    def test_rerank_llm_empty_input(self) -> None:
        from pipeline.rerank import rerank_llm
        result = rerank_llm([])
        assert result == []


# ---------------------------------------------------------------------------
# Orchestrator (fully mocked)
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def _mock_all(self, dummy_brief: Brief, dummy_cover: Image.Image, dummy_scores: dict):
        composed = ComposedCard(
            image=dummy_cover,
            headline=dummy_brief.headline,
            bbox=(10, 10, 300, 100),
            font_family="Rubik",
            font_size=28,
            colour_rgb=(20, 20, 20),
        )
        inside = InsideMessage(primary="Happy Birthday, Mum.", alternatives=[])
        return {
            "generation.brief.generate.generate_brief": dummy_brief,
            "generation.image.diffusion.get_runner": MagicMock(return_value=MagicMock(
                generate=MagicMock(return_value=[dummy_cover] * 2)
            )),
            "generation.image.headline_text.render_card": composed,
            "generation.message.generate.generate_message": inside,
            "pipeline.orchestrator.PredictorRunner": MagicMock(return_value=MagicMock(
                score=MagicMock(return_value=[dummy_scores] * 2)
            )),
            "pipeline.orchestrator.CLIPEmbedder": MagicMock(return_value=MagicMock(
                embed_images=MagicMock(return_value=np.zeros((2, 768), dtype=np.float32)),
                embed_texts=MagicMock(return_value=np.zeros((2, 768), dtype=np.float32)),
            )),
            "pipeline.orchestrator.put_image": MagicMock(return_value=("abc", "s3://bucket/key")),
        }

    @pytest.mark.parametrize("scorer", ["ridge", "mlp"])
    def test_orchestrator_returns_candidates(
        self, dummy_brief: Brief, dummy_cover: Image.Image, dummy_scores: dict,
        scorer: str, tmp_path: Path,
    ) -> None:
        # render_card returns a RenderedCard; the orchestrator reads .image
        # and .text_in_image from it.
        rendered_mock = MagicMock(image=dummy_cover, text_in_image=True, match_score=1.0)
        cursor_mock = MagicMock(
            fetchone=MagicMock(return_value={"card_id": "uuid-1"})
        )
        cursor_ctx = MagicMock(__enter__=MagicMock(return_value=cursor_mock),
                               __exit__=MagicMock(return_value=False))
        conn_inner = MagicMock(cursor=MagicMock(return_value=cursor_ctx))
        conn_ctx = MagicMock(__enter__=MagicMock(return_value=conn_inner),
                             __exit__=MagicMock(return_value=False))

        with patch("pipeline.orchestrator.generate_brief", return_value=dummy_brief) as brief_call, \
             patch("generation.brief.market_signals.subject_pool_size", return_value=12), \
             patch("pipeline.orchestrator.get_diffusion_runner", return_value=MagicMock(
                 return_value=MagicMock(generate=MagicMock(return_value=[dummy_cover] * 2))
             )), \
             patch("pipeline.orchestrator.render_card", return_value=rendered_mock) as render, \
             patch("pipeline.orchestrator.generate_message", return_value=InsideMessage(
                 primary="Happy Birthday", alternatives=[]
             )) as message_call, \
             patch("models.predictor.infer.PredictorRunner", return_value=MagicMock(
                 score=MagicMock(return_value=[dummy_scores] * 2)
             )), \
             patch("models.predictor.ridge.RidgePredictor.load", return_value=MagicMock(
                 score=MagicMock(return_value=[dummy_scores] * 2)
             )), \
             patch("data.features.clip_embed.CLIPEmbedder", return_value=MagicMock(
                 embed_images=MagicMock(return_value=np.zeros((2, 768))),
                 embed_texts=MagicMock(return_value=np.zeros((2, 768))),
             )), \
             patch("pipeline.orchestrator.put_image", return_value=("x", "s3://b/k")), \
             patch("pipeline.orchestrator.connection", return_value=conn_ctx):

            from pipeline.orchestrator import OrchestratorConfig, generate
            # The ridge path refuses to run without its artifact, so the file
            # has to exist even though loading it is patched out.
            ridge_path = tmp_path / "ridge.npz"
            ridge_path.touch()
            cfg = OrchestratorConfig(n_candidates=2, top_k=1,
                                     scorer=scorer,
                                     predictor_ckpt=Path("/dev/null"),
                                     predictor_ridge=ridge_path,
                                     predictor_calib=None)
            result = generate({"occasion": "birthday/general", "tone": "warm-humorous"}, cfg)

        # One brief per candidate, each anchored on a different bestseller.
        # Sharing a brief across candidates made reranking choose between
        # renders of one idea rather than between designs.
        assert brief_call.call_count == 2
        subjects = [
            c.args[0]["constraints"]["suggested_subject"] for c in brief_call.call_args_list
        ]
        assert len(set(subjects)) == 2, subjects

        # Messages are written after reranking, so the discarded candidate
        # never costs one.
        assert message_call.call_count == 1

        assert len(result) <= 2
        for c in result:
            assert c.occasion == "birthday/general"

        # The LoRA's trigger token has to reach the prompt. has_lora used to be
        # decided by rebuilding the directory path here, which looked for
        # loras/birthday_general and missed the group LoRA at loras/birthday —
        # so the token was silently dropped while the runner loaded the weights
        # anyway, and nothing in the logs said so.
        assert render.call_args.kwargs["visual_prompt"].startswith("TOK ")
