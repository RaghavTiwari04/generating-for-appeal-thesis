"""End-to-end card generation orchestrator.

Wires:
    request -> brief -> N parallel image gen (headline lettered in) -> message
            -> predictor reranking -> top-k candidates -> persist

Public entry point: `generate(request, n=8) -> list[Candidate]` ranked by
calibrated saleability score.
"""

from __future__ import annotations

import io
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
from generation.image.headline_text import render_card
from generation.message.generate import generate_message
from pipeline.rerank import Candidate, rerank, rerank_llm

log = get_logger(__name__)

PIPELINE_VERSION = "pipeline_v1"


@dataclass
class OrchestratorConfig:
    n_candidates: int = 8
    top_k: int = 3
    predictor_ckpt: Path = Path("./artifacts/predictor/best.ckpt")
    predictor_ridge: Path = Path("./artifacts/predictor/ridge.npz")
    predictor_calib: Path | None = Path("./artifacts/predictor/isotonic.joblib")
    image_seed_base: int | None = None
    condition_tag: str = "C_pipeline_rerank"
    # "ridge" is the default because it is the model that ranks best: it leads
    # the MLP on all five heads on the held-out split and recovers 71.4% of the
    # best-of-8 gain against the MLP's 66.4%. "mlp" and "llm" stay selectable
    # so the comparison can be re-run.
    scorer: str = "ridge"  # "ridge" | "mlp" | "llm"


def _load_predictor(cfg: OrchestratorConfig, calib: Path | None):
    """The scoring model rerank ranks with. Both expose the same `.score()`."""
    if cfg.scorer == "mlp":
        from models.predictor.infer import PredictorRunner

        return PredictorRunner(cfg.predictor_ckpt, calib)

    from models.predictor.ridge import RidgePredictor

    if not cfg.predictor_ridge.exists():
        raise SystemExit(
            f"No ridge model at {cfg.predictor_ridge}. Fit one with "
            "`python -m models.predictor.ridge`, or set scorer='mlp'."
        )
    return RidgePredictor.load(cfg.predictor_ridge, calib)


def generate(request: dict, cfg: OrchestratorConfig | None = None) -> list[Candidate]:
    cfg = cfg or OrchestratorConfig()

    brief: Brief = generate_brief(request)
    log.info(
        f"Brief generated occasion={request['occasion']} tone={request['tone']}: "
        f"{brief.headline!r}"
    )

    diffusion = get_diffusion_runner()
    visual_prompt = brief.visual_prompt
    lora_dir = Path("generation/image/loras") / request["occasion"].replace("/", "_")
    has_lora = lora_dir.exists()

    # The brief already specifies an art medium (see the "Vary the art medium"
    # rule in brief_v1.txt). Prepending a second, randomly chosen medium here
    # produced contradictory prompts like "oil painting of a papercut collage
    # of ...". Candidate variation now comes from the seed and from the
    # per-card bestseller-index rotation in pipeline/conditions.py.
    prompt = f"TOK {visual_prompt}" if has_lora else visual_prompt

    # Each candidate is rendered with its headline lettered into the artwork
    # where the model manages it, and composed onto a reserved region where it
    # does not. See generation/image/headline_text.py.
    rendered = []
    for i in range(cfg.n_candidates):
        rendered.append(
            render_card(
                diffusion,
                visual_prompt=prompt,
                headline=brief.headline,
                tone=request["tone"],
                style_tags=list(brief.style_tags),
                occasion=request["occasion"],
                seed=(cfg.image_seed_base + i) if cfg.image_seed_base is not None else None,
                negative_prompt=brief.negative_prompt,
            )
        )
    in_image = sum(r.text_in_image for r in rendered)
    log.info(f"Headline rendered into artwork for {in_image}/{len(rendered)} candidates")

    inside = generate_message(
        occasion=request["occasion"],
        tone=request["tone"],
        concept=brief.concept,
        headline=brief.headline,
    )

    candidates: list[Candidate] = []
    for i, card in enumerate(rendered):
        candidates.append(
            Candidate(
                image=card.image,
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

        calib = cfg.predictor_calib if (cfg.predictor_calib and cfg.predictor_calib.exists()) else None
        predictor = _load_predictor(cfg, calib)
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
    retries: int = 3,
) -> None:
    import time

    out_dir = Path("./artifacts/generated_cards")
    out_dir.mkdir(parents=True, exist_ok=True)

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

        for attempt in range(retries):
            try:
                with connection() as conn, conn.cursor() as cur:
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
                break
            except Exception as e:
                log.warning(f"DB persist attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(10 * (attempt + 1))
                else:
                    log.error(f"DB persist failed after {retries} attempts for rank={rank}")


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
