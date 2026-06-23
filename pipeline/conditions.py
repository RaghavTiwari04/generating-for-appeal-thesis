"""Generate cards for all four evaluation conditions.

Conditions:
  A  Naive AI        — Flux + naive occasion prompt + headline overlay, no LoRA, no brief LLM
  B  Pipeline no-rerank — full pipeline (LoRA + inpainting mask + layout + LLM message), N=1
  C  Pipeline + rerank  — full pipeline, predictor best-of-N (N=8)
  D  Human bestsellers  — pulled from `listings` table (no generation)

`generate_eval_set(occasions, n_per_condition_per_occasion)` produces a list of
`EvalCard`s that are persisted to `generated_cards` with the correct
`condition_tag`. The human bestsellers (D) are sampled directly from the DB.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from psycopg.types.json import Jsonb

from common.db import connection
from common.logging import get_logger
from common.storage import put_image

log = get_logger(__name__)

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
def _generate_naive(occasion: str, seed: int) -> EvalCard:
    """Naive prompt + headline mask + typography overlay, no LoRA, no brief LLM.

    Uses the same inpainting mask and layout composer as B/C so the only
    difference is prompt quality and LoRA — not card format.
    """
    from generation.image.diffusion import DiffusionConfig, DiffusionRunner
    from generation.image.headline_mask import LayoutMaskSpec, build_headline_mask
    from generation.layout.compose import compose

    naive_prompt = f"a greeting card for {occasion.replace('/', ' ').replace('_', ' ')}, digital art"
    headline = f"Happy {occasion.replace('_', ' ').replace('/', ' ').title()}"

    cfg = DiffusionConfig()
    runner = DiffusionRunner(cfg)
    mask_spec = LayoutMaskSpec()
    mask_image, _ = build_headline_mask(mask_spec)

    images = runner.generate(
        prompt=naive_prompt,
        occasion=None,     # no LoRA
        seed=seed,
        n=1,
        mask_image=mask_image,
    )

    result = compose(
        cover=images[0], headline=headline,
        tone="warm-sincere", style_tags=[], mask_spec=mask_spec,
    )
    composed = result.image

    import io
    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    _, storage_path = put_image(buf.getvalue(), content_type="image/png")

    from generation.message.generate import generate_message
    msg = generate_message(
        occasion=occasion,
        tone="warm-sincere",
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
def _generate_pipeline_no_rerank(occasion: str, seed: int, scorer: str = "predictor") -> EvalCard:
    from pipeline.orchestrator import OrchestratorConfig, generate

    cfg = OrchestratorConfig(
        n_candidates=1,
        top_k=1,
        image_seed_base=seed,
        condition_tag=CONDITION_TAGS["B"],
        scorer=scorer,
    )
    ranked = generate({"occasion": occasion, "tone": "warm-sincere"}, cfg)
    c = ranked[0]
    return EvalCard(
        condition="B",
        condition_tag=CONDITION_TAGS["B"],
        occasion=occasion,
        cover_path=None,  # already persisted by orchestrator
        headline=c.headline,
        inside_message=c.inside_message,
        predicted_scores=c.scores or {},
    )


# ---------------------------------------------------------------------------
# Condition C — pipeline + rerank (N=8)
# ---------------------------------------------------------------------------
def _generate_pipeline_rerank(occasion: str, seed: int, scorer: str = "predictor") -> EvalCard:
    from pipeline.orchestrator import OrchestratorConfig, generate

    cfg = OrchestratorConfig(
        n_candidates=8,
        top_k=1,
        image_seed_base=seed,
        condition_tag=CONDITION_TAGS["C"],
        scorer=scorer,
    )
    ranked = generate({"occasion": occasion, "tone": "warm-sincere"}, cfg)
    c = ranked[0]
    return EvalCard(
        condition="C",
        condition_tag=CONDITION_TAGS["C"],
        occasion=occasion,
        cover_path=None,
        headline=c.headline,
        inside_message=c.inside_message,
        predicted_scores=c.scores or {},
    )


# ---------------------------------------------------------------------------
# Condition D — human bestsellers (DB query, no generation)
# ---------------------------------------------------------------------------
_TOP_LISTINGS_SQL = """
SELECT l.listing_id, li.storage_path, l.title, lf.occasion
FROM listings l
JOIN listing_features lf USING (listing_id)
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
WHERE lf.occasion = %(occasion)s
  AND l.is_bestseller = TRUE
ORDER BY COALESCE(l.review_count, 0) + COALESCE(l.favourite_count, 0) DESC
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


def _persist_eval_card(card: EvalCard, seed: int) -> str:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            _INSERT,
            {
                "pv": f"eval_{card.condition}",
                "ct": card.condition_tag,
                "brief": Jsonb({"occasion": card.occasion, "condition": card.condition}),
                "cover_path": card.cover_path,
                "inside_message": card.inside_message,
                "headline_text": card.headline,
                "predicted_scores": Jsonb(card.predicted_scores),
                "seed": seed,
            },
        )
        card_id = str(cur.fetchone()["card_id"])
    return card_id


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
    for occ_i, occasion in enumerate(occasions):
        for cond_j, cond in enumerate(conditions):
            for k in range(n_per_condition_per_occasion):
                seed = seed_base + occ_i * 1000 + cond_j * 100 + k
                try:
                    if cond == "A":
                        card = _generate_naive(occasion, seed)
                    elif cond == "B":
                        card = _generate_pipeline_no_rerank(occasion, seed, scorer=scorer)
                    elif cond == "C":
                        card = _generate_pipeline_rerank(occasion, seed, scorer=scorer)
                    elif cond == "D":
                        card = _sample_human_bestseller(occasion, seed)
                        if card is None:
                            log.warning(f"No human bestsellers for {occasion}, skipping D")
                            continue
                    else:
                        raise ValueError(cond)

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
