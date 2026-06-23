"""LLM-as-judge system evaluation (synthetic consumer panel).

Replaces Prolific human study with VLM ratings. Methodological justification:
    Maier et al. (2024) "LLMs Reproduce Human Purchase Intent via Semantic
    Similarity Elicitation of Likert Ratings" — SSR achieves 90% of human
    test-retest reliability with KS similarity > 0.85.

Four conditions (same as system_eval.py):
    A_naive_ai          — raw diffusion, no pipeline
    B_pipeline_no_rerank — full pipeline, random top-k (no predictor)
    C_pipeline_rerank   — full pipeline + predictor reranking
    D_human_bestseller  — top-selling scraped marketplace cards

Each card scored by LLM on 5 dimensions:
    occasion_fit, aesthetic, emotional_resonance, distinctiveness, purchase_intent

Statistical analysis mirrors system_eval.py:
    - Per-condition means + SEM
    - Holm-corrected pairwise Mann-Whitney U
    - Per-occasion breakdown
    - Per-dimension breakdown

Usage:
    python -m eval.llm_system_eval --occasions birthday/general,birthday/milestone
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common.db import connection, engine
from common.logging import get_logger
from common.storage import get_object
from eval.ssr_scorer import DIMS, SSRScorer

log = get_logger(__name__)

CONDITIONS = ("A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller")

N_JUDGES = 3


@dataclass
class LLMSystemEvalReport:
    condition_means: dict[str, float]
    condition_stderr: dict[str, float]
    pairwise_p_holm: dict[str, float]
    per_occasion_means: pd.DataFrame
    per_head_means: dict[str, dict[str, float]]
    n_cards: int
    n_ratings: int


def _load_generated_cards(conditions: list[str]) -> pd.DataFrame:
    sql = """
    SELECT gc.card_id::text AS card_key,
           gc.condition_tag,
           gc.cover_path,
           gc.headline_text,
           gc.inside_message,
           (gc.brief->'request'->>'occasion') AS occasion
    FROM generated_cards gc
    WHERE gc.condition_tag = ANY(%(conditions)s)
    """
    return pd.read_sql(sql, engine(), params={"conditions": conditions})


def _load_human_bestsellers(occasions: list[str], per_occasion: int = 3) -> pd.DataFrame:
    sql = """
    SELECT li.listing_id::text AS card_key,
           'D_human_bestseller' AS condition_tag,
           li.storage_path AS cover_path,
           l.title AS headline_text,
           '' AS inside_message,
           lf.occasion
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
    LEFT JOIN saleability_labels sl
      ON sl.listing_id = l.listing_id AND sl.label_source = 'vlm_4head_v1'
    WHERE lf.occasion = ANY(%(occasions)s)
    ORDER BY COALESCE(sl.score, 0) DESC
    LIMIT %(limit)s
    """
    return pd.read_sql(
        sql, engine(),
        params={"occasions": occasions, "limit": per_occasion * len(occasions)},
    )


def _load_image(cover_path: str) -> Image.Image | None:
    import io

    try:
        if cover_path.startswith("s3://") or cover_path.startswith("greeting-cards"):
            data = get_object(cover_path)
        else:
            data = Path(cover_path).read_bytes()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        log.warning(f"Failed to load image {cover_path}: {e}")
        return None


def _score_cards(cards_df: pd.DataFrame, n_judges: int = N_JUDGES, out_dir: Path | None = None) -> pd.DataFrame:
    """Score cards using SSR methodology (Maier et al. 2025).

    Each card is evaluated by n_judges synthetic consumer profiles.
    SSR elicits free-text responses from the LLM, then maps them to
    Likert distributions via embedding similarity to reference statements.
    This produces realistic response distributions (KS>0.85) unlike
    direct numerical ratings (KS=0.26).
    """
    scorer = SSRScorer()
    log.info(
        f"SSR scoring {len(cards_df)} cards × {len(scorer.profiles)} consumer profiles"
    )

    ratings = []
    response_log = []
    for _, row in cards_df.iterrows():
        img = _load_image(row["cover_path"])
        if img is None:
            continue

        scores = scorer.score_one(
            image=img,
            headline=row.get("headline_text", "") or "",
            inside_message=row.get("inside_message", "") or "",
            occasion=row.get("occasion", "birthday/general") or "birthday/general",
        )

        responses = scores.pop("_responses", [])
        response_log.extend([
            {"card_key": row["card_key"], **r} for r in responses
        ])

        for profile in scorer.profiles:
            ratings.append({
                "card_key": row["card_key"],
                "condition": row["condition_tag"],
                "occasion": row.get("occasion", ""),
                "judge_id": f"consumer_age{profile['age']}",
                **{d: scores.get(d, 0.5) for d in DIMS},
            })

        log.info(
            f"  {row['condition_tag']} card={row['card_key'][:8]}... "
            f"pi={scores.get('purchase_intent', 0):.2f} "
            f"aes={scores.get('aesthetic', 0):.2f}"
        )

    if response_log:
        resp_df = pd.DataFrame(response_log)
        save_dir = out_dir or Path("./artifacts/llm_system_eval")
        save_dir.mkdir(parents=True, exist_ok=True)
        resp_df.to_csv(save_dir / "ssr_responses.csv", index=False)

    return pd.DataFrame(ratings)


def pairwise_holm(df: pd.DataFrame, metric: str = "purchase_intent") -> dict[str, float]:
    import statsmodels.stats.multitest as smm
    from scipy.stats import mannwhitneyu

    df = df.dropna(subset=[metric, "condition"])
    pairs = []
    raw_p = []
    seen = [c for c in CONDITIONS if c in df["condition"].values]
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            sa = df[df["condition"] == a][metric]
            sb = df[df["condition"] == b][metric]
            if len(sa) < 3 or len(sb) < 3:
                continue
            _, p = mannwhitneyu(sa, sb, alternative="two-sided")
            pairs.append(f"{a}_vs_{b}")
            raw_p.append(p)
    if not raw_p:
        return {}
    _, p_corrected, _, _ = smm.multipletests(raw_p, alpha=0.05, method="holm")
    return dict(zip(pairs, [float(p) for p in p_corrected], strict=False))


def run(
    occasions: str = "birthday/general",
    n_judges: int = N_JUDGES,
    human_per_occasion: int = 3,
    out_dir: str | Path = "./artifacts/llm_system_eval",
) -> LLMSystemEvalReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    occ_list = [o.strip() for o in occasions.split(",") if o.strip()]
    gen_conditions = [c for c in CONDITIONS if c != "D_human_bestseller"]

    cards_gen = _load_generated_cards(gen_conditions)
    if occ_list:
        cards_gen = cards_gen[cards_gen["occasion"].isin(occ_list)]

    cards_human = _load_human_bestsellers(occ_list, per_occasion=human_per_occasion)

    all_cards = pd.concat([cards_gen, cards_human], ignore_index=True)
    log.info(
        f"Cards to evaluate: {len(all_cards)} "
        f"({len(cards_gen)} generated + {len(cards_human)} human bestsellers)"
    )

    if all_cards.empty:
        raise SystemExit("No cards found. Generate cards under conditions A/B/C first.")

    ratings_df = _score_cards(all_cards, n_judges=n_judges, out_dir=out)
    ratings_df.to_csv(out / "raw_ratings.csv", index=False)

    cond_means = ratings_df.groupby("condition")["purchase_intent"].mean().to_dict()
    cond_stderr = (
        ratings_df.groupby("condition")["purchase_intent"].sem(ddof=1).fillna(0.0).to_dict()
    )

    pairwise = pairwise_holm(ratings_df)

    per_occ = (
        ratings_df.groupby(["occasion", "condition"])["purchase_intent"]
        .mean()
        .unstack("condition")
    )

    per_head = {}
    for dim in DIMS:
        per_head[dim] = ratings_df.groupby("condition")[dim].mean().dropna().to_dict()

    report = LLMSystemEvalReport(
        condition_means={str(k): float(v) for k, v in cond_means.items()},
        condition_stderr={str(k): float(v) for k, v in cond_stderr.items()},
        pairwise_p_holm=pairwise,
        per_occasion_means=per_occ,
        per_head_means=per_head,
        n_cards=len(all_cards),
        n_ratings=len(ratings_df),
    )

    report_dict = {
        "method": "LLM-as-judge (synthetic consumer panel)",
        "reference": "Maier et al. (2024) SSR — 90% human test-retest reliability",
        "n_judges": n_judges,
        "n_cards": report.n_cards,
        "n_ratings": report.n_ratings,
        "conditions": list(CONDITIONS),
        "occasions_evaluated": occ_list,
        "condition_means": report.condition_means,
        "condition_stderr": report.condition_stderr,
        "pairwise_p_holm": report.pairwise_p_holm,
        "per_head_means": report.per_head_means,
    }
    (out / "report.json").write_text(json.dumps(report_dict, indent=2))
    per_occ.to_csv(out / "per_occasion.csv")

    log.info(f"\n{'='*60}")
    log.info("LLM System Evaluation Results")
    log.info(f"{'='*60}")
    log.info(f"Cards: {report.n_cards}  Ratings: {report.n_ratings}  Judges: {n_judges}")
    log.info(f"\nPurchase Intent by condition:")
    for cond in CONDITIONS:
        m = report.condition_means.get(cond, float("nan"))
        s = report.condition_stderr.get(cond, float("nan"))
        log.info(f"  {cond:30s}  {m:.3f} ± {s:.3f}")
    log.info(f"\nPairwise (Holm-corrected p-values):")
    for pair, p in report.pairwise_p_holm.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log.info(f"  {pair:50s}  p={p:.4f} {sig}")
    log.info(f"\nPer-dimension means:")
    for dim, cond_vals in report.per_head_means.items():
        vals = "  ".join(f"{c[:5]}={v:.2f}" for c, v in cond_vals.items())
        log.info(f"  {dim:25s}  {vals}")

    return report


if __name__ == "__main__":
    import typer

    typer.run(run)
