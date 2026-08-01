"""Generate cards for all four evaluation conditions.

Conditions:
  A  Naive AI        — Flux + naive occasion prompt, no LoRA, no brief LLM
  B  Pipeline no-rerank — full pipeline (LoRA + brief LLM + LLM message), N=1
  C  Pipeline + rerank  — full pipeline, predictor best-of-N (N=8)
  D  Human bestsellers  — pulled from `listings` table (no generation)

`generate_eval_set(occasions, n_per_condition_per_occasion)` produces a list of
`EvalCard`s that are persisted to `generated_cards` with the correct
`condition_tag`. The human bestsellers (D) are sampled directly from the DB.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from psycopg.types.json import Jsonb

from common.db import connection
from common.logging import get_logger
from common.storage import put_image
from generation.brief.market_signals import bestseller_subjects_for_occasion

log = get_logger(__name__)

_subject_cache: dict[str, int] = {}

def _get_subject_pool_size(occasion: str) -> int:
    """Return number of bestseller titles available for this occasion."""
    if occasion in _subject_cache:
        return _subject_cache[occasion]
    try:
        raw = bestseller_subjects_for_occasion(occasion, limit=30)
        n = len(raw)
        log.info(f"Loaded {n} bestseller titles for {occasion}")
    except Exception as e:
        log.warning(f"Failed to load bestseller titles for {occasion} ({e}), using default pool size")
        n = 15
    _subject_cache[occasion] = max(n, 5)
    return _subject_cache[occasion]

# Rotated across the k cards of every condition. Fixing tone at warm-sincere
# left every generated card sincere while the scraped corpus — and therefore
# condition D — is humour-heavy, so any measured gap partly reflected tone
# rather than quality. A/B/C use the same tone for the same k, keeping the
# conditions matched.
EVAL_TONES: tuple[str, ...] = (
    "warm-sincere",
    "warm-humorous",
    "funny-irreverent",
    "sentimental",
    "minimalist",
)

CONDITION_TAGS = {
    "A": "A_naive_ai",
    "B": "B_pipeline_no_rerank",
    "C": "C_pipeline_rerank",
    "D": "D_human_bestseller",
}


@dataclass
class EvalCard:
    condition: str          # A / B / C / D
    condition_tag: str
    occasion: str
    cover_path: str | None
    headline: str | None
    inside_message: str | None
    predicted_scores: dict
    card_id: str | None = None
    listing_id: str | None = None  # D only


# ---------------------------------------------------------------------------
# Condition A — naive AI
# ---------------------------------------------------------------------------
def _generate_naive(occasion: str, seed: int, tone: str = "warm-sincere") -> EvalCard:
    """Naive prompt, no LoRA, no brief LLM.

    Uses the same headline rendering as B/C so the only difference is prompt
    quality and the absence of a LoRA — not card format.
    """
    from generation.image.diffusion import get_runner
    from generation.image.headline_text import render_card

    naive_prompt = f"a greeting card for {occasion.replace('/', ' ').replace('_', ' ')}, digital art"
    headline = f"Happy {occasion.replace('_', ' ').replace('/', ' ').title()}"

    # Share the process-wide runner. A private DiffusionRunner would hold a
    # second copy of Flux + Flux-Fill (~48GB) alongside the shared one and OOM
    # an 80GB card. Passing occasion=None makes _ensure_occasion drop any
    # LoRA-fused pipeline and reload clean, preserving A's no-LoRA baseline.
    runner = get_runner()

    # Same card format as B and C — headline lettered into the artwork where
    # the model manages it, overlaid where it does not. A must differ only in
    # prompt quality and the absence of a LoRA, not in how text is applied.
    card = render_card(
        runner,
        visual_prompt=naive_prompt,
        headline=headline,
        tone=tone,
        style_tags=[],
        occasion=None,     # no LoRA
        seed=seed,
    )
    composed = card.image

    import hashlib
    import io
    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    data = buf.getvalue()
    try:
        _, storage_path = put_image(data, content_type="image/png")
    except Exception as e:
        out_dir = Path("./artifacts/generated_cards")
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = occasion.replace("/", "_")
        digest = hashlib.sha256(data).hexdigest()[:16]
        local_path = out_dir / f"naive_{slug}_{seed}_{digest}.png"
        local_path.write_bytes(data)
        storage_path = str(local_path)
        log.warning(f"MinIO upload failed ({e}), saved locally: {local_path}")

    from generation.message.generate import generate_message
    msg = generate_message(
        occasion=occasion,
        tone=tone,
        concept=f"A card for {occasion}",
        headline=headline,
    )

    return EvalCard(
        condition="A",
        condition_tag=CONDITION_TAGS["A"],
        occasion=occasion,
        cover_path=storage_path,
        headline=headline,
        inside_message=msg.primary,
        predicted_scores={},
    )


# ---------------------------------------------------------------------------
# Condition B — pipeline, no rerank (N=1)
# ---------------------------------------------------------------------------
def _generate_pipeline_no_rerank(
    occasion: str, seed: int, scorer: str = "predictor", subject: str | None = None,
    tone: str = "warm-sincere",
) -> EvalCard:
    from pipeline.orchestrator import OrchestratorConfig, generate

    cfg = OrchestratorConfig(
        n_candidates=1,
        top_k=1,
        image_seed_base=seed,
        condition_tag=CONDITION_TAGS["B"],
        scorer=scorer,
    )
    request: dict = {"occasion": occasion, "tone": tone}
    if subject:
        request["constraints"] = {"suggested_subject": subject}
    ranked = generate(request, cfg)
    c = ranked[0]
    return EvalCard(
        condition="B",
        condition_tag=CONDITION_TAGS["B"],
        occasion=occasion,
        cover_path=None,
        headline=c.headline,
        inside_message=c.inside_message,
        predicted_scores=c.scores or {},
        card_id=c.card_id,
    )


# ---------------------------------------------------------------------------
# Condition C — pipeline + rerank (N=8)
# ---------------------------------------------------------------------------
def _generate_pipeline_rerank(
    occasion: str, seed: int, scorer: str = "predictor", subject: str | None = None,
    tone: str = "warm-sincere",
) -> EvalCard:
    from pipeline.orchestrator import OrchestratorConfig, generate

    cfg = OrchestratorConfig(
        n_candidates=8,
        top_k=1,
        image_seed_base=seed,
        condition_tag=CONDITION_TAGS["C"],
        scorer=scorer,
    )
    request: dict = {"occasion": occasion, "tone": tone}
    if subject:
        request["constraints"] = {"suggested_subject": subject}
    ranked = generate(request, cfg)
    c = ranked[0]
    return EvalCard(
        condition="C",
        condition_tag=CONDITION_TAGS["C"],
        occasion=occasion,
        cover_path=None,
        headline=c.headline,
        inside_message=c.inside_message,
        predicted_scores=c.scores or {},
        card_id=c.card_id,
    )


# ---------------------------------------------------------------------------
# Condition D — human bestsellers (DB query, no generation)
# ---------------------------------------------------------------------------
# One listing per duplicate cluster, best-scoring member representing it.
#
# Without the collapse the pool is colourways of a few designs: print-on-demand
# catalogues carry the same artwork many times, copies score near-identically
# because they are the same artwork, so they sort adjacently and fill the top
# 50. Sampling from that shows the same design repeatedly across the
# evaluation, and condition D is meant to stand for the range of what human
# designers sell — a handful of designs in different colours would understate
# it and bias the equivalence test the whole study turns on.
_TOP_LISTINGS_SQL = """
SELECT listing_id, storage_path, title, occasion
FROM (
    SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text))
           l.listing_id, li.storage_path, l.title, lf.occasion,
           COALESCE(sl.score, 0) AS score
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
    LEFT JOIN saleability_labels sl
      ON sl.listing_id = l.listing_id AND sl.label_source = 'llm_ssr_rubric_v2'
    WHERE lf.occasion = %(occasion)s
    ORDER BY COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text),
             COALESCE(sl.score, 0) DESC,
             l.listing_id
) representatives
ORDER BY score DESC
LIMIT %(limit)s;
"""


def _sample_human_bestseller(occasion: str, seed: int, pool_size: int = 50) -> EvalCard | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_TOP_LISTINGS_SQL, {"occasion": occasion, "limit": pool_size})
        rows = cur.fetchall()
    if not rows:
        return None
    rng = random.Random(seed)
    row = rng.choice(rows)
    return EvalCard(
        condition="D",
        condition_tag=CONDITION_TAGS["D"],
        occasion=occasion,
        cover_path=row["storage_path"],
        headline=row.get("title"),
        inside_message=None,
        predicted_scores={},
        listing_id=str(row["listing_id"]),
    )


# ---------------------------------------------------------------------------
# Persist to generated_cards
# ---------------------------------------------------------------------------
_INSERT = """
INSERT INTO generated_cards (
    pipeline_version, condition_tag, brief, cover_path,
    inside_message, headline_text, predicted_scores, seed
) VALUES (
    %(pv)s, %(ct)s, %(brief)s, %(cover_path)s,
    %(inside_message)s, %(headline_text)s, %(predicted_scores)s, %(seed)s
) RETURNING card_id;
"""


def _persist_eval_card(card: EvalCard, seed: int, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            with connection() as conn, conn.cursor() as cur:
                cur.execute(
                    _INSERT,
                    {
                        "pv": f"eval_{card.condition}",
                        "ct": card.condition_tag,
                        "brief": Jsonb({"request": {"occasion": card.occasion}, "condition": card.condition}),
                        "cover_path": card.cover_path,
                        "inside_message": card.inside_message,
                        "headline_text": card.headline,
                        "predicted_scores": Jsonb(card.predicted_scores),
                        "seed": seed,
                    },
                )
                card_id = str(cur.fetchone()["card_id"])
            return card_id
        except Exception as e:
            log.warning(f"DB persist attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"DB persist failed after {retries} attempts")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def generate_eval_set(
    occasions: list[str],
    *,
    n_per_condition_per_occasion: int = 2,
    seed_base: int = 0,
    conditions: tuple[str, ...] = ("A", "B", "C", "D"),
    scorer: str = "predictor",
) -> list[EvalCard]:
    cards: list[EvalCard] = []
    log.info(
        f"Starting eval set: occasions={occasions} conditions={conditions} "
        f"n_per={n_per_condition_per_occasion} seed_base={seed_base}"
    )
    for occ_i, occasion in enumerate(occasions):
        pool_size = _get_subject_pool_size(occasion)
        for cond_j, cond in enumerate(conditions):
            for k in range(n_per_condition_per_occasion):
                seed = seed_base + occ_i * 1000 + cond_j * 100 + k
                bestseller_idx = (k % pool_size) + 1
                tone = EVAL_TONES[k % len(EVAL_TONES)]
                try:
                    if cond == "A":
                        card = _generate_naive(occasion, seed, tone=tone)
                    elif cond == "B":
                        card = _generate_pipeline_no_rerank(occasion, seed, scorer=scorer, subject=str(bestseller_idx), tone=tone)
                    elif cond == "C":
                        card = _generate_pipeline_rerank(occasion, seed, scorer=scorer, subject=str(bestseller_idx), tone=tone)
                    elif cond == "D":
                        card = _sample_human_bestseller(occasion, seed)
                        if card is None:
                            log.warning(f"No human bestsellers for {occasion}, skipping D")
                            continue
                    else:
                        raise ValueError(cond)

                    if cond in ("B", "C") and card.card_id:
                        cards.append(card)
                        log.info(f"Generated {cond} {occasion} tone={tone} seed={seed} card_id={card.card_id}")
                    else:
                        card_id = _persist_eval_card(card, seed)
                        card.card_id = card_id
                        cards.append(card)
                        log.info(f"Generated {cond} {occasion} seed={seed} card_id={card_id}")
                except Exception as e:
                    log.error(f"Failed condition={cond} occasion={occasion} seed={seed}: {e}")
    return cards


if __name__ == "__main__":
    import typer


    def cli(
        occasions: str = "birthday/general,birthday/milestone,birthday/kids,birthday/relationship",
        n: int = 2,
        conditions: str = "A,B,C",
        seed: int = 0,
        scorer: str = "predictor",
    ) -> None:
        occ_list = [o.strip() for o in occasions.split(",")]
        cond_tuple = tuple(c.strip() for c in conditions.split(","))
        cards = generate_eval_set(
            occ_list,
            n_per_condition_per_occasion=n,
            seed_base=seed,
            conditions=cond_tuple,
            scorer=scorer,
        )
        print(f"Generated {len(cards)} eval cards")

    typer.run(cli)
