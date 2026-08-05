"""LLM-as-judge system evaluation (synthetic consumer panel).

Replaces Prolific human study with hybrid LLM evaluation:
    - Purchase intent: SSR (Maier et al. 2025, arXiv:2510.08338).
      The paper's KS>0.85 result concerns reproducing response
      *distributions*; here three personas are averaged to a point
      estimate, so that validation does not transfer.
    - Other dims: Rubric-guided LLM judge (Zheng et al. 2023, NeurIPS)
      — >80% agreement with human annotators

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common.db import engine
from common.logging import get_logger
from common.storage import get_object
from scoring import DIMS, CardScorer
from scoring.card_scorer import CONSUMER_PROFILES

log = get_logger(__name__)

CONDITIONS = ("A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller")



@dataclass
class LLMSystemEvalReport:
    condition_means: dict[str, float]
    condition_stderr: dict[str, float]
    bootstrap_ci: dict[str, tuple[float, float]]
    pairwise_p_holm: dict[str, float]
    pairwise_effect_size: dict[str, float]
    tost_equivalence: dict[str, dict]
    per_occasion_means: pd.DataFrame
    per_occasion_pairwise: dict[str, dict[str, float]]
    per_head_means: dict[str, dict[str, float]]
    n_cards: int
    n_ratings: int


def _latest_run_tag() -> str | None:
    """The most recently generated run's tag, or None if nothing is tagged."""
    row = pd.read_sql(
        """
        SELECT gc.brief->'request'->>'eval_run' AS run_tag
        FROM generated_cards gc
        WHERE gc.brief->'request'->>'eval_run' IS NOT NULL
        ORDER BY gc.generated_at DESC
        LIMIT 1
        """,
        engine(),
    )
    return None if row.empty else row.iloc[0]["run_tag"]


def _load_generated_cards(conditions: list[str], run_tag: str | None) -> pd.DataFrame:
    """Cards from one generation run.

    Selecting on condition_tag alone pooled every card ever generated, smoke
    tests included — cards made under an earlier brief prompt, carrying the
    typographic overlay, ranked on a different objective. The first full run
    showed condition C below B for exactly that reason: C's rows were all old
    while B's were fresh. The table looked plausible.

    run_tag None reads everything, which is only right for inspecting history.
    """
    sql = """
    SELECT gc.card_id::text AS card_key,
           gc.condition_tag,
           gc.cover_path,
           gc.headline_text,
           gc.inside_message,
           COALESCE(gc.brief->'request'->>'occasion', gc.brief->>'occasion') AS occasion
    FROM generated_cards gc
    WHERE gc.condition_tag = ANY(%(conditions)s)
      -- Cast required: Postgres cannot infer a bare parameter's type from
      -- `$2 IS NULL` alone and refuses the statement.
      AND (%(run_tag)s::text IS NULL OR gc.brief->'request'->>'eval_run' = %(run_tag)s::text)
    """
    return pd.read_sql(
        sql, engine(), params={"conditions": conditions, "run_tag": run_tag}
    )


def _load_human_bestsellers(
    occasions: list[str],
    per_occasion: int = 5,
    *,
    pool_size: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Condition D: cards drawn at random from each occasion's bestseller band.

    Random within the band, not the top `per_occasion` by score. Those scores
    come from the same judge that rates every condition, so taking the maximum
    of them selects on the dependent variable: D would be handed the cards the
    judge happens to score highest, including whatever part of that is noise,
    and the pipeline would be measured against an inflated baseline.

    Drawing from the top `pool_size` keeps "bestseller" meaningful — these are
    still the strongest-selling designs for the occasion — while making D's
    expected score the band's mean rather than its maximum. Seeded, so the
    comparison is reproducible.
    """
    # Rank WITHIN each occasion. A global ORDER BY score would fill the whole
    # D sample from the highest-scoring occasions and drop others entirely,
    # leaving D with a different occasion mix from the balanced A/B/C sets —
    # so a C-vs-D difference would partly reflect occasion mix, not quality.
    # Duplicate clusters collapse before ranking. Print-on-demand catalogues
    # carry one artwork in many colourways, and copies score near-identically
    # because they are the same artwork, so the top 5 for an occasion could be
    # one or two designs repeated. Condition D would then stand for the range of
    # what human designers sell while showing almost none of it, and it is the
    # baseline every other condition is compared against.
    sql = """
    SELECT card_key, condition_tag, cover_path, headline_text, inside_message, occasion
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY occasion ORDER BY score DESC, card_key
               ) AS rn
        FROM (
            SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text))
                   li.listing_id::text AS card_key,
                   'D_human_bestseller' AS condition_tag,
                   li.storage_path AS cover_path,
                   l.title AS headline_text,
                   '' AS inside_message,
                   lf.occasion,
                   COALESCE(sl.score, 0) AS score
            FROM listings l
            JOIN listing_features lf USING (listing_id)
            JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
            LEFT JOIN saleability_labels sl
              ON sl.listing_id = l.listing_id AND sl.label_source = 'llm_ssr_rubric_v2'
            WHERE lf.occasion = ANY(%(occasions)s)
            ORDER BY COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text),
                     COALESCE(sl.score, 0) DESC,
                     l.listing_id
        ) representatives
    ) ranked
    WHERE rn <= %(pool_size)s
    """
    pool = pd.read_sql(
        sql, engine(),
        params={"occasions": occasions, "pool_size": pool_size},
    )
    if pool.empty:
        return pool
    # Sampled per occasion so every occasion contributes the same number of
    # cards, matching the balanced A/B/C sets.
    rng = np.random.default_rng(seed)
    picked = [
        rows.sample(n=min(per_occasion, len(rows)), random_state=rng.integers(2**32))
        for _, rows in pool.groupby("occasion", sort=True)
    ]
    return pd.concat(picked, ignore_index=True)


# Every card reaches the judge at the same scale.
#
# Generated cards are upscaled to print resolution, 1240x1748, while the
# scraped marketplace images are 736x1030 or 750x1000 — 2.9x the pixels, and
# after the provider's own downscale to a 1568 long edge still about 2.3x. The
# first full run had condition D at 0.57 on aesthetic against C's 0.73, which a
# resolution gap that size could produce on its own.
#
# Set to the human corpus's long edge so nothing is upscaled: enlarging D would
# invent detail and trade one artefact for another.
JUDGE_LONG_EDGE = 1030


def _load_image(cover_path: str) -> Image.Image | None:
    import io

    try:
        # get_object handles s3://, file:// and plain filesystem paths, so
        # locally-stored blobs from the MinIO fallback resolve too.
        data = get_object(cover_path)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if max(img.size) > JUDGE_LONG_EDGE:
            scale = JUDGE_LONG_EDGE / max(img.size)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
            )
        return img
    except Exception as e:
        short_path = cover_path.rsplit("/", 1)[-1][:20] if "/" in cover_path else cover_path[:20]
        log.warning(f"Image load failed [{type(e).__name__}]: ...{short_path} — {e}")
        return None


def _score_cards(
    cards_df: pd.DataFrame,
    out_dir: Path | None = None,
    provider: str = "openai",
    model: str | None = None,
) -> pd.DataFrame:
    """Score cards using SSR methodology (Maier et al. 2025).

    Each card is evaluated by 3 synthetic consumer profiles (SSR)
    SSR elicits free-text responses from the LLM, then maps them to
    Likert distributions via embedding similarity to reference statements.
    Free-text elicitation avoids the known failure of asking an LLM for a
    number directly; the distributional validation in the paper applies to
    population-level use, not to this per-card point estimate.
    """
    scorer = CardScorer(provider=provider, model=model)
    log.info(
        f"SSR scoring {len(cards_df)} cards × {len(scorer.profiles)} consumer profiles"
    )

    ratings = []
    response_log = []
    # Recorded per condition so a resolution gap between conditions is visible
    # in the log rather than inferred from a suspicious aesthetic score.
    seen_sizes: dict[str, set[tuple[int, int]]] = {}
    for _, row in cards_df.iterrows():
        img = _load_image(row["cover_path"])
        if img is None:
            continue
        seen_sizes.setdefault(row["condition_tag"], set()).add(img.size)

        # Occasion only. Headline and inside message are withheld from every
        # condition: D has no scraped inside message and only a marketplace
        # title, so supplying them would penalise it for a gap in data
        # collection rather than for the card.
        scores = scorer.score(
            img,
            occasion=row.get("occasion", "birthday/general") or "birthday/general",
        )

        responses = [
            {"dimension": d, "explanation": e}
            for d, e in scores.pop("explanations", {}).items()
        ] + [{"dimension": "purchase_intent", "response": r}
             for r in scores.pop("ssr_responses", [])]
        response_log.extend([
            {"card_key": row["card_key"], **r} for r in responses
        ])

        ratings.append({
            "card_key": row["card_key"],
            "condition": row["condition_tag"],
            "occasion": row.get("occasion", ""),
            **{d: scores.get(d, np.nan) for d in DIMS},
        })

        log.info(
            f"  {row['condition_tag']} card={row['card_key'][:8]}... "
            f"pi={scores.get('purchase_intent', 0):.2f} "
            f"aes={scores.get('aesthetic', 0):.2f}"
        )

    for tag, sizes in sorted(seen_sizes.items()):
        px = [w * h / 1e6 for w, h in sizes]
        log.info(
            f"  {tag:24s} judged at {sorted(sizes)[:2]} "
            f"({min(px):.2f}-{max(px):.2f} MP)"
        )

    if response_log:
        resp_df = pd.DataFrame(response_log)
        save_dir = out_dir or Path("./artifacts/llm_system_eval")
        save_dir.mkdir(parents=True, exist_ok=True)
        resp_df.to_csv(save_dir / "ssr_responses.csv", index=False)

    return pd.DataFrame(ratings)


def _rank_biserial(sa: pd.Series, sb: pd.Series) -> float:
    """Rank-biserial correlation (effect size for Mann-Whitney U)."""
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(sa, sb, alternative="two-sided")
    n1, n2 = len(sa), len(sb)
    return 1 - (2 * u) / (n1 * n2)


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def _tost_equivalence(
    sa: pd.Series, sb: pd.Series, delta: float = 0.02,
) -> dict:
    """Two One-Sided Tests for equivalence within ±delta."""
    from scipy.stats import mannwhitneyu
    diff = sa.mean() - sb.mean()
    _, p_upper = mannwhitneyu(sa, sb + delta, alternative="less")
    _, p_lower = mannwhitneyu(sa, sb - delta, alternative="greater")
    p_tost = max(p_lower, p_upper)
    return {
        "mean_diff": float(diff),
        "delta": delta,
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < 0.05),
    }


def pairwise_holm(df: pd.DataFrame, metric: str = "purchase_intent") -> tuple[dict[str, float], dict[str, float]]:
    """Returns (holm_p_values, effect_sizes) dicts."""
    import statsmodels.stats.multitest as smm
    from scipy.stats import mannwhitneyu

    df = df.dropna(subset=[metric, "condition"])
    pairs = []
    raw_p = []
    effects = {}
    seen = [c for c in CONDITIONS if c in df["condition"].values]
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            sa = df[df["condition"] == a][metric]
            sb = df[df["condition"] == b][metric]
            if len(sa) < 3 or len(sb) < 3:
                continue
            _, p = mannwhitneyu(sa, sb, alternative="two-sided")
            pair_key = f"{a}_vs_{b}"
            pairs.append(pair_key)
            raw_p.append(p)
            effects[pair_key] = _rank_biserial(sa, sb)
    if not raw_p:
        return {}, {}
    _, p_corrected, _, _ = smm.multipletests(raw_p, alpha=0.05, method="holm")
    p_dict = dict(zip(pairs, [float(p) for p in p_corrected], strict=False))
    return p_dict, effects


def per_occasion_pairwise(df: pd.DataFrame, metric: str = "purchase_intent") -> dict[str, dict[str, float]]:
    """Pairwise Mann-Whitney within each occasion (uncorrected, exploratory)."""
    from scipy.stats import mannwhitneyu
    results = {}
    for occ in df["occasion"].dropna().unique():
        occ_df = df[df["occasion"] == occ].dropna(subset=[metric, "condition"])
        seen = [c for c in CONDITIONS if c in occ_df["condition"].values]
        occ_results = {}
        for i, a in enumerate(seen):
            for b in seen[i + 1:]:
                sa = occ_df[occ_df["condition"] == a][metric]
                sb = occ_df[occ_df["condition"] == b][metric]
                if len(sa) < 2 or len(sb) < 2:
                    continue
                _, p = mannwhitneyu(sa, sb, alternative="two-sided")
                occ_results[f"{a}_vs_{b}"] = float(p)
        results[occ] = occ_results
    return results


def run(
    occasions: str = "birthday/general,birthday/milestone,birthday/kids,birthday/relationship",
    # 0 matches condition D to whatever the generation run produced per
    # occasion. A fixed number pins it regardless.
    human_per_occasion: int = 0,
    out_dir: str = "./artifacts/llm_system_eval",
    provider: str = "gemini",
    model: str = "",
    analyze_only: bool = False,
    run_tag: str = "",
    all_runs: bool = False,
) -> LLMSystemEvalReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    occ_list = [o.strip() for o in occasions.split(",") if o.strip()]

    if analyze_only:
        csv_path = out / "raw_ratings.csv"
        if not csv_path.exists():
            raise SystemExit(f"No ratings file at {csv_path}. Run without --analyze-only first.")
        ratings_df = pd.read_csv(csv_path)
        all_cards = ratings_df
        log.info(f"Loaded {len(ratings_df)} existing ratings from {csv_path}")
    else:
        gen_conditions = [c for c in CONDITIONS if c != "D_human_bestseller"]

        # One run's cards, not every card ever generated. Defaults to the most
        # recent tagged run; --all-runs is for inspecting history and will mix
        # cards made under different prompts and reranking objectives.
        tag = None if all_runs else (run_tag or _latest_run_tag())
        if tag is None and not all_runs:
            log.warning(
                "No tagged generation run found. Scoring every generated card, "
                "which mixes runs — regenerate with a run_tag for a clean comparison."
            )
        log.info(f"Generated cards from run_tag={tag or '(all)'}")

        cards_gen = _load_generated_cards(gen_conditions, tag)
        n_before_filter = len(cards_gen)
        if occ_list:
            cards_gen = cards_gen[cards_gen["occasion"].isin(occ_list)]
        if len(cards_gen) < n_before_filter:
            dropped = n_before_filter - len(cards_gen)
            log.warning(f"Occasion filter dropped {dropped}/{n_before_filter} generated cards (NULL/mismatched occasion)")

        # Matched to what the generation run actually produced, per occasion.
        # Fixed at 5, D stayed at 20 cards while N_PER=10 took A/B/C to 40 —
        # half the sample on the baseline every equivalence test is against,
        # which widens D's interval and makes equivalence harder to show for no
        # reason. An explicit --human-per-occasion still wins.
        per_occ = human_per_occasion
        if not per_occ and not cards_gen.empty and occ_list:
            per_cond_occ = cards_gen.groupby(["condition_tag", "occasion"]).size()
            per_occ = round(per_cond_occ.mean()) if len(per_cond_occ) else 5
            log.info(f"Condition D matched to generated conditions: {per_occ} per occasion")
        cards_human = _load_human_bestsellers(occ_list, per_occasion=per_occ or 5)

        all_cards = pd.concat([cards_gen, cards_human], ignore_index=True)
        per_cond = cards_gen.groupby("condition_tag").size().to_dict()
        log.info(
            f"Cards to evaluate: {len(all_cards)} "
            f"({len(cards_gen)} generated + {len(cards_human)} human bestsellers)"
        )
        log.info(f"Per-condition breakdown: {per_cond}")

        if all_cards.empty:
            raise SystemExit("No cards found. Generate cards under conditions A/B/C first.")

        # Judge only what has not been judged. Scoring is deterministic for the
        # rubric dimensions and costs ten VLM calls a card, so re-rating an
        # unchanged card buys nothing. Before this the cache was all-or-nothing:
        # adding twenty condition D cards meant re-judging the hundred and
        # nineteen generated ones that had not changed.
        csv_path = out / "raw_ratings.csv"
        cached = pd.read_csv(csv_path) if csv_path.exists() else None
        if cached is not None and "card_key" in cached:
            known = set(cached["card_key"].astype(str))
            todo = all_cards[~all_cards["card_key"].astype(str).isin(known)]
            log.info(f"{len(cached)} cards already rated; judging {len(todo)} new")
        else:
            known, todo = set(), all_cards

        fresh = (
            _score_cards(todo, out_dir=out, provider=provider, model=model or None)
            if len(todo)
            else pd.DataFrame()
        )
        ratings_df = pd.concat([cached, fresh], ignore_index=True) if cached is not None else fresh
        # Cards no longer in the comparison stay out of the analysis but remain
        # in the cache, so re-running an older run_tag does not re-judge them.
        ratings_df.to_csv(csv_path, index=False)
        wanted = set(all_cards["card_key"].astype(str))
        ratings_df = ratings_df[ratings_df["card_key"].astype(str).isin(wanted)]

    cond_means = ratings_df.groupby("condition")["purchase_intent"].mean().to_dict()
    cond_stderr = (
        ratings_df.groupby("condition")["purchase_intent"].sem(ddof=1).fillna(0.0).to_dict()
    )

    # Bootstrap 95% CIs
    boot_ci = {}
    for cond in CONDITIONS:
        vals = ratings_df[ratings_df["condition"] == cond]["purchase_intent"].dropna().values
        if len(vals) >= 3:
            boot_ci[cond] = _bootstrap_ci(vals)

    # Pairwise tests with effect sizes
    pairwise, effect_sizes = pairwise_holm(ratings_df)

    # TOST equivalence tests (all pairs)
    tost_results = {}
    seen_conds = [c for c in CONDITIONS if c in ratings_df["condition"].values]
    tost_pairs = [(a, b) for i, a in enumerate(seen_conds) for b in seen_conds[i + 1:]]
    for pair_a, pair_b in tost_pairs:
        sa = ratings_df[ratings_df["condition"] == pair_a]["purchase_intent"].dropna()
        sb = ratings_df[ratings_df["condition"] == pair_b]["purchase_intent"].dropna()
        if len(sa) >= 3 and len(sb) >= 3:
            tost_results[f"{pair_a}_vs_{pair_b}"] = _tost_equivalence(sa, sb)

    # Per-occasion pairwise (exploratory)
    occ_pairwise = per_occasion_pairwise(ratings_df)

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
        bootstrap_ci={str(k): v for k, v in boot_ci.items()},
        pairwise_p_holm=pairwise,
        pairwise_effect_size={k: float(v) for k, v in effect_sizes.items()},
        tost_equivalence=tost_results,
        per_occasion_means=per_occ,
        per_occasion_pairwise=occ_pairwise,
        per_head_means=per_head,
        n_cards=len(all_cards),
        n_ratings=len(ratings_df),
    )

    report_dict = {
        "method": "Hybrid LLM evaluation",
        "purchase_intent_method": "SSR (Maier et al. 2025, arXiv:2510.08338)",
        "other_dims_method": "Rubric-guided LLM judge (Zheng et al. 2023, NeurIPS)",
        "n_consumer_profiles": len(CONSUMER_PROFILES),
        "n_cards": report.n_cards,
        "n_ratings": report.n_ratings,
        "conditions": list(CONDITIONS),
        "occasions_evaluated": occ_list,
        "condition_means": report.condition_means,
        "condition_stderr": report.condition_stderr,
        "bootstrap_ci_95": {k: list(v) for k, v in report.bootstrap_ci.items()},
        "pairwise_p_holm": report.pairwise_p_holm,
        "pairwise_effect_size_rank_biserial": report.pairwise_effect_size,
        "tost_equivalence": report.tost_equivalence,
        "per_occasion_pairwise": report.per_occasion_pairwise,
        "per_head_means": report.per_head_means,
    }
    (out / "report.json").write_text(json.dumps(report_dict, indent=2))
    per_occ.to_csv(out / "per_occasion.csv")

    log.info(f"\n{'='*60}")
    log.info("LLM System Evaluation Results")
    log.info(f"{'='*60}")
    log.info(f"Cards: {report.n_cards}  Ratings: {report.n_ratings}")
    log.info("\nPurchase Intent by condition:")
    for cond in CONDITIONS:
        m = report.condition_means.get(cond, float("nan"))
        s = report.condition_stderr.get(cond, float("nan"))
        ci = report.bootstrap_ci.get(cond)
        ci_str = f"  95%CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        log.info(f"  {cond:30s}  {m:.3f} ± {s:.3f}{ci_str}")
    log.info("\nPairwise (Holm-corrected p + rank-biserial r):")
    for pair, p in report.pairwise_p_holm.items():
        r = report.pairwise_effect_size.get(pair, float("nan"))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log.info(f"  {pair:50s}  p={p:.4f} r={r:+.3f} {sig}")
    log.info("\nTOST equivalence tests (δ=0.02):")
    for pair, res in report.tost_equivalence.items():
        eq = "EQUIVALENT" if res["equivalent"] else "inconclusive"
        log.info(f"  {pair:50s}  Δ={res['mean_diff']:+.4f} p_tost={res['p_tost']:.4f} {eq}")
    log.info("\nPer-occasion pairwise (exploratory, uncorrected):")
    for occ, pairs in report.per_occasion_pairwise.items():
        sig_pairs = [f"{p.split('_vs_')[0][:5]}v{p.split('_vs_')[1][:5]}={v:.3f}" for p, v in pairs.items() if v < 0.05]
        if sig_pairs:
            log.info(f"  {occ:30s}  sig: {', '.join(sig_pairs)}")
        else:
            log.info(f"  {occ:30s}  no significant pairs")
    log.info("\nPer-dimension means:")
    for dim, cond_vals in report.per_head_means.items():
        vals = "  ".join(f"{c[:5]}={v:.2f}" for c, v in cond_vals.items())
        log.info(f"  {dim:25s}  {vals}")

    return report


if __name__ == "__main__":
    import typer

    typer.run(run)
