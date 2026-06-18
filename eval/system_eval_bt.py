"""BT-based system evaluation for pairwise 2AFC studies.

Four conditions (A naive AI, B pipeline-no-rerank, C pipeline+rerank, D human
bestsellers). Fits a single Bradley-Terry model across all cards, then
groups BT sale_scores by condition for comparison.

Pre-registered hypotheses (Holm-corrected pairwise Mann-Whitney U):
    H1: C > A    H2: C > B    H3: C ≈ D

Complements system_eval.py (Likert-based, for v1 pilot data).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.db import engine
from common.logging import get_logger
from survey.analysis.bradley_terry import BTResult, fit_bradley_terry, load_pairs

log = get_logger(__name__)

CONDITIONS = ("A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller")


@dataclass
class BTSystemEvalReport:
    condition_means: dict[str, float]
    condition_stderr: dict[str, float]
    pairwise_p_holm: dict[str, float]
    per_occasion_means: pd.DataFrame
    bt_result: BTResult


_CARD_CONDITION_SQL = """
SELECT card_id::text AS card_key, condition_tag,
       (brief->'request'->>'occasion') AS occasion
FROM generated_cards
WHERE condition_tag = ANY(%(conditions)s)
UNION ALL
SELECT listing_id::text AS card_key, 'D_human_bestseller' AS condition_tag,
       lf.occasion
FROM listings l
JOIN listing_features lf USING (listing_id)
WHERE l.is_bestseller = TRUE;
"""


def _card_conditions() -> pd.DataFrame:
    return pd.read_sql(
        _CARD_CONDITION_SQL, engine(), params={"conditions": list(CONDITIONS)}
    )


def pairwise_holm_bt(scores_by_condition: dict[str, np.ndarray]) -> dict[str, float]:
    import statsmodels.stats.multitest as smm
    from scipy.stats import mannwhitneyu

    conds = [c for c in CONDITIONS if c in scores_by_condition]
    pairs = []
    raw_p = []
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            sa = scores_by_condition[a]
            sb = scores_by_condition[b]
            if len(sa) < 5 or len(sb) < 5:
                continue
            _, p = mannwhitneyu(sa, sb, alternative="two-sided")
            pairs.append(f"{a}_vs_{b}")
            raw_p.append(p)
    if not raw_p:
        return {}
    _, p_corrected, _, _ = smm.multipletests(raw_p, alpha=0.05, method="holm")
    return dict(zip(pairs, [float(p) for p in p_corrected], strict=False))


def run(
    study_id: str,
    question_dim: str = "purchase_intent",
    out_dir: str | Path = "./artifacts/system_eval_bt",
) -> BTSystemEvalReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs_df = load_pairs(study_id, question_dim=question_dim)
    if pairs_df.empty:
        raise SystemExit("No pairs found for this study")

    bt = fit_bradley_terry(pairs_df)
    bt_scores = pd.DataFrame({"card_key": bt.card_keys, "sale_score": bt.sale_scores})

    card_conds = _card_conditions()
    merged = bt_scores.merge(card_conds, on="card_key", how="inner")

    if merged.empty:
        raise SystemExit("No card_key matches between BT results and generated_cards")

    cond_means = merged.groupby("condition_tag")["sale_score"].mean().to_dict()
    cond_stderr = merged.groupby("condition_tag")["sale_score"].sem(ddof=1).fillna(0.0).to_dict()

    scores_by_cond = {
        c: merged[merged["condition_tag"] == c]["sale_score"].to_numpy()
        for c in CONDITIONS if c in merged["condition_tag"].values
    }
    pairwise = pairwise_holm_bt(scores_by_cond)

    per_occ = merged.groupby(["occasion", "condition_tag"])["sale_score"].mean().unstack("condition_tag")

    report = BTSystemEvalReport(
        condition_means={str(k): float(v) for k, v in cond_means.items()},
        condition_stderr={str(k): float(v) for k, v in cond_stderr.items()},
        pairwise_p_holm=pairwise,
        per_occasion_means=per_occ,
        bt_result=bt,
    )

    (out / "report.json").write_text(
        json.dumps(
            {
                "study_id": study_id,
                "question_dim": question_dim,
                "n_cards": bt.n_cards,
                "n_comparisons": bt.n_comparisons,
                "converged": bt.converged,
                "condition_means": report.condition_means,
                "condition_stderr": report.condition_stderr,
                "pairwise_p_holm": report.pairwise_p_holm,
            },
            indent=2,
        )
    )
    per_occ.to_csv(out / "per_occasion.csv")
    log.info(f"BT system eval: {len(merged)} cards across {len(cond_means)} conditions")
    return report


if __name__ == "__main__":
    import typer

    typer.run(run)
