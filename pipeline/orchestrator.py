"""End-to-end card generation orchestrator.

Wires:
    request -> brief -> N parallel image gen -> layout -> message
            -> predictor reranking -> top-k candidates -> persist

Public entry point: `generate(request, n=8) -> list[Candidate]` ranked by
calibrated saleability score.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from common.config import settings
from common.db import connection
from common.logging import get_logger
from common.storage import put_image
from data.features.clip_embed import CLIPEmbedder
from generation.brief.generate import PROMPT_VERSION as BRIEF_VERSION, generate_brief
from generation.brief.schema import Brief
from generation.image.controlnet import LayoutMaskSpec, build_headline_mask
from generation.image.diffusion import get_runner as get_diffusion_runner
from generation.layout.compose import compose
from generation.message.generate import generate_message
from models.predictor.infer import PredictorRunner
from pipeline.rerank import Candidate, rerank

log = get_logger(__name__)

PIPELINE_VERSION = "pipeline_v1"


@dataclass
class OrchestratorConfig:
    n_candidates: int = 8
    top_k: int = 3
    predictor_ckpt: Path = Path("./artifacts/predictor/best.ckpt")
    predictor_calib: Path | None = Path("./artifacts/predictor/isotonic.joblib")
    image_seed_base: int | None = None
    condition_tag: str = "C_pipeline_rerank"


def generate(request: dict, cfg: OrchestratorConfig | None = None) -> list[Candidate]:
    cfg = cfg or OrchestratorConfig()

    brief: Brief = generate_brief(request)
    log.info(
        f"Brief generated occasion={request['occasion']} tone={request['tone']}: "
        f"{brief.headline!r}"
    )

    diffusion = get_diffusion_runner()
    mask_spec = LayoutMaskSpec()
    mask_image, _ = build_headline_mask(mask_spec)

    images = diffusion.generate(
        prompt=brief.visual_prompt,
        negative_prompt=brief.negative_prompt,
        occasion=request["occasion"],
        seed=cfg.image_seed_base,
        n=cfg.n_candidates,
        controlnet_image=mask_image,
    )

    inside = generate_message(
        occasion=request["occasion"],
        tone=request["tone"],
        concept=brief.concept,
        headline=brief.headline,
    )

    candidates: list[Candidate] = []
    for i, cover in enumerate(images):
        composed = compose(
            cover=cover,
            headline=brief.headline,
            tone=request["tone"],
            style_tags=list(brief.style_tags),
            mask_spec=mask_spec,
        )
        candidates.append(
            Candidate(
                image=composed.image,
                headline=brief.headline,
                inside_message=inside.primary,
                brief=brief.model_dump(),
                occasion=request["occasion"],
                seed=(cfg.image_seed_base + i) if cfg.image_seed_base is not None else None,
            )
        )

    predictor = PredictorRunner(cfg.predictor_ckpt, cfg.predictor_calib)
    embedder = CLIPEmbedder()
    ranked = rerank(candidates, predictor=predictor, embedder=embedder, top_k=cfg.top_k)

    _persist(ranked, request=request, cfg=cfg, brief=brief, inside_alternatives=inside.alternatives)
    return ranked


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
_INSERT = """
INSERT INTO generated_cards (
    pipeline_version, condition_tag, brief, cover_path,
    inside_message, headline_text, predicted_scores, seed, generated_at
) VALUES (
    %(pipeline_version)s, %(condition_tag)s, %(brief)s, %(cover_path)s,
    %(inside_message)s, %(headline_text)s, %(predicted_scores)s, %(seed)s, NOW()
)
RETURNING card_id;
"""


def _persist(
    ranked: list[Candidate],
    *,
    request: dict,
    cfg: OrchestratorConfig,
    brief: Brief,
    inside_alternatives: list[str],
) -> None:
    with connection() as conn, conn.cursor() as cur:
        for cand in ranked:
            buf = io.BytesIO()
            cand.image.save(buf, format="PNG")
            _, storage_path = put_image(buf.getvalue(), content_type="image/png")

            cur.execute(
                _INSERT,
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "condition_tag": cfg.condition_tag,
                    "brief": Jsonb(
                        {
                            "request": request,
                            "brief": brief.model_dump(),
                            "inside_alternatives": inside_alternatives,
                            "brief_prompt_version": BRIEF_VERSION,
                        }
                    ),
                    "cover_path": storage_path,
                    "inside_message": cand.inside_message,
                    "headline_text": cand.headline,
                    "predicted_scores": Jsonb(cand.scores or {}),
                    "seed": cand.seed,
                },
            )


if __name__ == "__main__":
    import typer

    def cli(
        occasion: str,
        tone: str = "warm-sincere",
        relationship: str | None = None,
        n: int = 8,
        top_k: int = 3,
    ) -> None:
        ranked = generate(
            {"occasion": occasion, "tone": tone, "relationship": relationship},
            OrchestratorConfig(n_candidates=n, top_k=top_k),
        )
        for i, c in enumerate(ranked):
            sale = (c.scores or {}).get("saleability_calibrated", float("nan"))
            print(f"#{i+1} sale={sale:.3f} headline={c.headline!r}")

    typer.run(cli)
