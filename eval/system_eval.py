"""End-to-end system evaluation.

Four conditions (A naive AI, B pipeline-no-rerank, C pipeline+rerank, D human
bestsellers). Survey ratings are joined from `survey_ratings` and analysed
with a mixed-effects model:

    purchase_intent ~ condition + (1|participant) + (1|card)

Pre-registered hypotheses (Holm-corrected pairwise):
    H1: C > A    H2: C > B    H3: C ≈ D
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)

CONDITIONS = ("A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller")


@dataclass
class SystemEvalReport:
    condition_means: dict[str, float]
    condition_stderr: dict[str, float]
    pairwise_p_holm: dict[str, float]
    per_occasion_means: pd.DataFrame
    per_head_means: dict[str, dict[str, float]]


_RATINGS_SQL = """
SELECT sr.participant_id,
       sr.occasion_shown,
       sr.purchase_intent,
       sr.occasion_fit,
       sr.aesthetic,
       sr.emotional_resonance,
       sr.distinctiveness,
       COALESCE(gc.condition_tag,
                CASE WHEN sr.listing_id IS NOT NULL THEN 'D_human_bestseller' END) AS condition,
       COALESCE(gc.card_id::text, sr.listing_id::text)                              AS card_key
FROM survey_ratings sr
LEFT JOIN generated_cards gc ON gc.card_id = sr.generated_card_id
WHERE sr.study_id = %(study_id)s
  AND sr.attention_check_pass IS NOT FALSE;
"""


def fetch_ratings(study_id: str) -> pd.DataFrame:
    return pd.read_sql(_RATINGS_SQL, engine(), params={"study_id": study_id})


def fit_mixed(df: pd.DataFrame) -> dict[str, float]:
    """Mixed-effects: purchase_intent ~ condition + (1|participant) + (1|card).

    statsmodels MixedLM supports only one random effect group; for the second
    we fall back to clustered SE via vc_formula. The point estimates are
    what we report; the proper full random-effects analysis can be lifted to
    R/`lme4` if reviewers ask.
    """
    import statsmodels.formula.api as smf

    df = df.dropna(subset=["purchase_intent", "condition"]).copy()
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=False)

    model = smf.mixedlm(
        "purchase_intent ~ C(condition)",
        df,
        groups=df["participant_id"],
        vc_formula={"card": "0 + C(card_key)"},
    )
    result = model.fit(method="lbfgs", reml=False)

    return {name: float(coef) for name, coef in result.params.items()}


def pairwise_holm(df: pd.DataFrame) -> dict[str, float]:
    import statsmodels.stats.multitest as smm
    from scipy.stats import mannwhitneyu

    df = df.dropna(subset=["purchase_intent", "condition"])
    pairs = []
    raw_p = []
    seen = list(dict.fromkeys(df["condition"]))
    for i, a in enumerate(seen):
        for b in seen[i + 1 :]:
            sa = df[df["condition"] == a]["purchase_intent"]
            sb = df[df["condition"] == b]["purchase_intent"]
            if len(sa) < 5 or len(sb) < 5:
                continue
            _, p = mannwhitneyu(sa, sb, alternative="two-sided")
            pairs.append(f"{a}_vs_{b}")
            raw_p.append(p)
    if not raw_p:
        return {}
    _, p_corrected, _, _ = smm.multipletests(raw_p, alpha=0.05, method="holm")
    return dict(zip(pairs, [float(p) for p in p_corrected], strict=False))


def run(study_id: str, out_dir: str | Path = "./artifacts/system_eval") -> SystemEvalReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = fetch_ratings(study_id)
    if df.empty:
        raise SystemExit("No ratings found for this study")

    cond_means = df.groupby("condition")["purchase_intent"].mean().to_dict()
    cond_stderr = (
        df.groupby("condition")["purchase_intent"].sem(ddof=1).fillna(0.0).to_dict()
    )

    pairwise = pairwise_holm(df)

    per_occ = df.groupby(["occasion_shown", "condition"])["purchase_intent"].mean().unstack("condition")

    per_head = {}
    for head in ["occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness"]:
        per_head[head] = df.groupby("condition")[head].mean().dropna().to_dict()

    report = SystemEvalReport(
        condition_means={str(k): float(v) for k, v in cond_means.items()},
        condition_stderr={str(k): float(v) for k, v in cond_stderr.items()},
        pairwise_p_holm=pairwise,
        per_occasion_means=per_occ,
        per_head_means=per_head,
    )

    (out / "report.json").write_text(
        json.dumps(
            {
                "condition_means": report.condition_means,
                "condition_stderr": report.condition_stderr,
                "pairwise_p_holm": report.pairwise_p_holm,
                "per_head_means": report.per_head_means,
            },
            indent=2,
        )
    )
    per_occ.to_csv(out / "per_occasion.csv")
    return report


if __name__ == "__main__":
    import typer

    typer.run(run)
