"""Score scraped listings with the LLM card scorer, for predictor training.

Uses scoring.CardScorer — the same instrument that reranks generated candidates
and runs the system evaluation. SSR (Maier et al. 2025) for purchase intent,
rubric-guided judge (Zheng et al. 2023) for the four quality dimensions.

This module previously had its own direct-rating prompt, so the predictor was
trained on targets measured differently from how its output is finally judged.

The pool is one listing per duplicate cluster, birthday cards only: colourways
of one design would otherwise each cost a full scoring pass, and listings with
no occasion are unusable downstream.

Labels land in `saleability_labels` with `score` = purchase_intent — the
construct the table is named for, and the one the human survey validates
against — and the full per-dimension detail in `raw`.

    python -m data.labels.vlm_labels label
    python -m data.labels.vlm_labels label --limit 20      # smoke test
    python -m data.labels.vlm_labels label --provider gemini
    python -m data.labels.vlm_labels stats
    python -m data.labels.vlm_labels rescore --dimension aesthetic
"""

from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass

import typer
from PIL import Image
from psycopg.types.json import Jsonb

from common.db import connection, engine
from common.logging import get_logger
from common.storage import get_object
from scoring import (
    DIMS,
    RUBRIC_DIMS,
    USAGE,
    CardScorer,
    openrouter_route,
)

log = get_logger(__name__)

LABEL_SOURCE = "llm_ssr_rubric_v2"

# Cards scored concurrently. Each card is 10 VLM calls, so this is the real
# multiplier against the provider's rate limit — and the main lever on how long
# a full run takes, since the work is entirely API latency.
MAX_CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
COMMIT_EVERY = 25

# One listing per duplicate cluster, and only cards with an occasion.
#
# Print-on-demand catalogues carry the same design in several colourways as
# separate listings; scoring each costs a full pass for a card already judged.
# Unclustered listings group by their own id, so they are all kept.
#
# Listings with a NULL occasion are skipped because nothing downstream can use
# them: the predictor requires occasion IS NOT NULL, and the LoRA and
# condition-D selections filter by occasion.
#
# Scoring reads the stored blob, not the listing's remote image URL. Dedup,
# the CLIP embeddings, LoRA training and the galleries all read the blob, so
# fetching the URL here scored whatever the site served at that moment — which
# need not be the image the rest of the pipeline associates with the listing,
# and need not be the same across two runs.
_POOL_SQL = """
SELECT listing_id, title, occasion, storage_path
FROM (
    SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text))
           l.listing_id::text AS listing_id,
           l.title,
           lf.occasion,
           li.storage_path,
           l.last_seen_at
    FROM listings l
    JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
    LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
    WHERE li.storage_path IS NOT NULL
      AND lf.occasion IS NOT NULL
    ORDER BY COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text), l.listing_id
) one_per_design
ORDER BY last_seen_at DESC
"""

_ALREADY_LABELLED_SQL = """
SELECT listing_id::text FROM saleability_labels WHERE label_source = %(label_source)s
"""

_UPSERT_LABEL = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
VALUES (%(listing_id)s, %(label_source)s, %(score)s, %(raw)s)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


@dataclass
class Card:
    listing_id: str
    title: str | None
    occasion: str | None
    storage_path: str


def _load_pool() -> list[Card]:
    """Every scorable card. Callers apply --limit, so the full size stays
    available for projecting a smoke run out to the whole corpus."""
    import pandas as pd

    df = pd.read_sql(_POOL_SQL, engine())
    return [
        Card(
            listing_id=r["listing_id"],
            title=r.get("title"),
            occasion=r.get("occasion"),
            storage_path=r["storage_path"],
        )
        for _, r in df.iterrows()
        if r.get("storage_path")
    ]


def _already_labelled(label_source: str) -> set[str]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_ALREADY_LABELLED_SQL, {"label_source": label_source})
        return {r["listing_id"] for r in cur.fetchall()}


def _persist(rows: list[dict], label_source: str) -> int:
    if not rows:
        return 0
    with connection() as conn, conn.cursor() as cur:
        for row in rows:
            cur.execute(
                _UPSERT_LABEL,
                {
                    "listing_id": row["listing_id"],
                    "label_source": label_source,
                    "score": row["score"],
                    "raw": Jsonb(row["raw"]),
                },
            )
    return len(rows)


def _load_image(card: Card) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(get_object(card.storage_path))).convert("RGB")
    except Exception as e:
        log.warning(f"Could not load {card.storage_path}: {e}")
        return None


async def _score_card(
    card: Card,
    sem: asyncio.Semaphore,
    scorer: CardScorer,
) -> dict | None:
    image = await asyncio.to_thread(_load_image, card)
    if image is None:
        return None
    async with sem:
        # CardScorer is synchronous; running it in a thread keeps the event
        # loop free so other cards' image fetches proceed during the API waits.
        scores = await asyncio.to_thread(
            scorer.score,
            image,
            occasion=card.occasion or "",
            headline=card.title or "",
        )
    # Persist only complete cards. A provider that fails mid-run — exhausted
    # credits, rate limits — returns empty text, and each affected dimension is
    # omitted. Writing the survivors would leave a card that a resume then skips
    # as done, so the gap becomes permanent and invisible. Treated as a failure
    # instead, it is simply re-scored on the next run.
    missing = [d for d in DIMS if d not in scores]
    if missing:
        log.warning(f"{card.listing_id[:8]}: incomplete ({', '.join(missing)}), not stored")
        return None
    return {
        "listing_id": card.listing_id,
        # The sortable summary is purchase intent, the construct the table is
        # named for: LoRA exemplars, condition D and the market signals all rank
        # on it, and it is the dimension the human survey validates against. The
        # other four stay in `raw` as separate analysable dimensions.
        "score": scores["purchase_intent"],
        "raw": scores,
    }


async def _run(cards: list[Card], scorer: CardScorer, label_source: str) -> int:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    written = failed = 0
    for start in range(0, len(cards), COMMIT_EVERY):
        chunk = cards[start : start + COMMIT_EVERY]
        results = await asyncio.gather(*(_score_card(c, sem, scorer) for c in chunk))
        good = [r for r in results if r is not None]
        failed += len(results) - len(good)
        written += _persist(good, label_source)
        log.info(
            f"  {min(start + COMMIT_EVERY, len(cards))}/{len(cards)} — "
            f"{written} scored, {failed} failed"
        )
    return written


app = typer.Typer(help="LLM saleability labelling (SSR + rubric judge)")


@app.command()
def label(
    provider: str = typer.Option("anthropic", help="anthropic | openai"),
    model: str | None = typer.Option(None, help="Model override"),
    label_source: str = typer.Option(LABEL_SOURCE, help="Label source tag"),
    limit: int | None = typer.Option(None, help="Score only the first N cards"),
    force: bool = typer.Option(False, help="Re-score cards that already have labels"),
    route: str | None = typer.Option(
        None,
        help=(
            "Gateway routing: an upstream name (pins it, no fallback) or a JSON "
            "object passed through. Pin before treating scores as reproducible."
        ),
    ),
) -> None:
    """Score listings on all five dimensions and persist the results."""
    pool = _load_pool()
    pool_total = len(pool)
    cards = pool[:limit] if limit else pool
    if force:
        log.info(f"Pool: {len(cards)} cards (--force, re-scoring all)")
    else:
        done = _already_labelled(label_source)
        before = len(cards)
        cards = [c for c in cards if c.listing_id not in done]
        log.info(
            f"Pool: {before} total, {before - len(cards)} already scored, "
            f"{len(cards)} to do"
        )

    if not cards:
        log.info("Nothing to do.")
        return

    scorer = CardScorer(
        provider=provider, model=model, route=openrouter_route(route)
    )
    per_card = len(scorer.profiles) * scorer.samples_per_persona + len(RUBRIC_DIMS)
    log.info(
        f"{len(cards)} cards x {per_card} calls = {len(cards) * per_card} "
        f"VLM calls via {provider}"
    )
    written = asyncio.run(_run(cards, scorer, label_source))
    print(f"Scored {written} cards")
    # Measured rather than estimated: what a full run over this pool will
    # actually cost, and whether the images leave any headroom to reclaim.
    print()
    print(USAGE.report(cards=written, project_to=pool_total))


@app.command()
def rescore(
    label_source: str = typer.Option(LABEL_SOURCE, help="Label source to rewrite"),
    dimension: str = typer.Option("purchase_intent", help=f"One of {', '.join(DIMS)}"),
) -> None:
    """Recompute `score` from the stored per-dimension detail.

    `raw` holds every dimension, so changing which one ranks the corpus costs a
    single UPDATE rather than re-scoring 2,468 cards through the API.
    """
    if dimension not in DIMS:
        raise typer.BadParameter(f"{dimension} is not one of {DIMS}")
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE saleability_labels "
            "SET score = (raw->>%(dim)s)::float "
            "WHERE label_source = %(src)s AND raw ? %(dim)s",
            {"dim": dimension, "src": label_source},
        )
        n = cur.rowcount
    print(f"Rewrote score = {dimension} for {n} rows of {label_source}")


@app.command()
def stats(label_source: str = typer.Option(LABEL_SOURCE)) -> None:
    """Report label coverage and per-dimension means."""
    import pandas as pd

    df = pd.read_sql(
        "SELECT score, raw FROM saleability_labels WHERE label_source = %(s)s",
        engine(),
        params={"s": label_source},
    )
    if df.empty:
        print(f"No labels for {label_source}.")
        return
    import numpy as np

    print(
        f"{len(df)} cards scored, composite mean {df['score'].mean():.3f} "
        f"sd {df['score'].std():.3f}"
    )
    # Spread matters more than the mean: a dimension every card scores the same
    # on carries no signal for the predictor to learn, however sensible its
    # average looks.
    print(f"  {'dimension':22s} {'mean':>6s} {'sd':>6s} {'min':>6s} {'max':>6s}    n")
    for dim in DIMS:
        vals = [
            r.get(dim)
            for r in df["raw"]
            if isinstance(r, dict) and r.get(dim) is not None
        ]
        if vals:
            a = np.asarray(vals, dtype=float)
            print(
                f"  {dim:22s} {a.mean():6.3f} {a.std(ddof=1) if len(a) > 1 else 0.0:6.3f} "
                f"{a.min():6.3f} {a.max():6.3f} {len(a):4d}"
            )

    # `score` is the mean of the rubric dimensions and excludes purchase_intent,
    # so ranking by it means "best looking" rather than "most likely to sell".
    # How much that matters is exactly this correlation: near 1 and the choice
    # is cosmetic, low and it changes which cards LoRA trains on and which ones
    # stand as condition D.
    wide = pd.DataFrame(
        [r for r in df["raw"] if isinstance(r, dict)]
    ).reindex(columns=list(DIMS))
    wide["composite"] = df["score"].to_numpy()
    usable = wide.dropna()
    if len(usable) > 2:
        print(f"\nSpearman between dimensions (n={len(usable)})\n")
        corr = usable.corr(method="spearman")
        cols = list(corr.columns)
        print("  " + " " * 22 + " ".join(f"{c[:9]:>9s}" for c in cols))
        for row in cols:
            cells = " ".join(f"{corr.loc[row, c]:9.3f}" for c in cols)
            print(f"  {row:22s} {cells}")


if __name__ == "__main__":
    app()
