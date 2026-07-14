"""End-to-end card generation orchestrator.

Wires:
    request -> brief -> N parallel image gen -> layout -> message
            -> predictor reranking -> top-k candidates -> persist

Public entry point: `generate(request, n=8) -> list[Candidate]` ranked by
calibrated saleability score.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

from psycopg.types.json import Jsonb

from common.db import connection
from common.logging import get_logger
from common.storage import put_image
from generation.brief.generate import PROMPT_VERSION as BRIEF_VERSION
from generation.brief.generate import generate_brief
from generation.brief.schema import Brief
from generation.image.diffusion import get_runner as get_diffusion_runner
from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
from generation.layout.compose import compose
from generation.message.generate import generate_message
from pipeline.rerank import Candidate, rerank, rerank_llm

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
    scorer: str = "predictor"  # "predictor" | "llm"


def generate(request: dict, cfg: OrchestratorConfig | None = None) -> list[Candidate]:
    cfg = cfg or OrchestratorConfig()

    brief: Brief = generate_brief(request)
    log.info(
        f"Brief generated occasion={request['occasion']} tone={request['tone']}: "
        f"{brief.headline!r}"
    )

    diffusion = get_diffusion_runner()
    # Reserve the headline region at *generation* resolution so the Fill pass
    # clears it to whitespace. compose() derives its own bbox from the upscaled
    # cover, so both stay proportionally aligned to the same fractional region.
    gen_mask_spec = LayoutMaskSpec(width=diffusion.cfg.width, height=diffusion.cfg.height)
    headline_mask, _ = build_headline_mask(gen_mask_spec)

    visual_prompt = brief.visual_prompt
    lora_dir = Path("generation/image/loras") / request["occasion"].replace("/", "_")
    has_lora = lora_dir.exists()

    STYLE_PREFIXES = [
        "",
        "watercolour illustration of ",
        "3D rendered cartoon of ",
        "paper cut collage of ",
        "flat vector illustration of ",
        "oil painting of ",
        "whimsical pencil sketch of ",
        "retro vintage poster of ",
    ]

    images = []
    rng = random.Random(cfg.image_seed_base or 0)
    for i in range(cfg.n_candidates):
        style = rng.choice(STYLE_PREFIXES)
        prompt_i = f"TOK {style}{visual_prompt}" if has_lora else f"{style}{visual_prompt}"
        img = diffusion.generate(
            prompt=prompt_i,
            negative_prompt=brief.negative_prompt,
            occasion=request["occasion"],
            seed=(cfg.image_seed_base + i) if cfg.image_seed_base is not None else None,
            n=1,
            mask_image=headline_mask,
        )
        images.extend(img)

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

    if cfg.scorer == "llm":
        ranked = rerank_llm(candidates, top_k=cfg.top_k)
    else:
        from data.features.clip_embed import CLIPEmbedder
        from models.predictor.infer import PredictorRunner

        calib = cfg.predictor_calib if (cfg.predictor_calib and cfg.predictor_calib.exists()) else None
        predictor = PredictorRunner(cfg.predictor_ckpt, calib)
        embedder = CLIPEmbedder()
        ranked = rerank(
            candidates, predictor=predictor, embedder=embedder, top_k=cfg.top_k
        )

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
    out_dir = Path("./artifacts/generated_cards")
    out_dir.mkdir(parents=True, exist_ok=True)

    with connection() as conn, conn.cursor() as cur:
        for rank, cand in enumerate(ranked):
            buf = io.BytesIO()
            cand.image.save(buf, format="PNG")
            data = buf.getvalue()

            try:
                _, storage_path = put_image(data, content_type="image/png")
            except Exception as e:
                import hashlib
                occasion_slug = request.get("occasion", "unknown").replace("/", "_")
                digest = hashlib.sha256(data).hexdigest()[:16]
                local_path = out_dir / f"{occasion_slug}_{rank:02d}_{digest}.png"
                local_path.write_bytes(data)
                storage_path = str(local_path)
                log.warning(f"MinIO upload failed ({e}), saved locally: {local_path}")

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
            row = cur.fetchone()
            if row:
                cand.card_id = str(row["card_id"])


if __name__ == "__main__":
    import sys
    import typer
    from common.storage import ensure_buckets

    def cli(
        occasion: str,
        tone: str = "warm-sincere",
        relationship: str | None = None,
        n: int = 8,
        top_k: int = 3,
        scorer: str = "predictor",
    ) -> None:
        ensure_buckets()
        try:
            ranked = generate(
                {"occasion": occasion, "tone": tone, "relationship": relationship},
                OrchestratorConfig(n_candidates=n, top_k=top_k, scorer=scorer),
            )
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            raise SystemExit(1)
        for i, c in enumerate(ranked):
            sale = (c.scores or {}).get("saleability_calibrated", float("nan"))
            print(f"#{i+1} sale={sale:.3f} headline={c.headline!r}")

    typer.run(cli)
