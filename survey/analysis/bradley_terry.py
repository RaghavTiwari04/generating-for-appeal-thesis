"""Bradley-Terry scaling from pairwise (2AFC) survey comparisons.

Recovers a scalar saleability score per card from `survey_pairs` rows.
Replaces per-card Likert means used by the original Likert-based pipeline.

Model
-----
P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))

We fit s = (s_1, ..., s_n) by maximum likelihood via the standard MM
(minorisation-maximisation) update of Hunter (2004):

    s_i^(t+1) = W_i / sum_{j != i} (N_ij / (exp(s_i^(t)) + exp(s_j^(t))))

where W_i is total wins for card i and N_ij is the number of i-vs-j
comparisons. Ties counted as 0.5 wins to each side.

Outputs are normalised to mean 0 and rescaled to [0, 1] via logistic for
direct comparison with predictor head outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)


_PAIRS_SQL = """
SELECT
    pair_id,
    participant_id,
    study_id,
    COALESCE(left_listing_id::text,  left_generated_id::text)  AS left_key,
    COALESCE(right_listing_id::text, right_generated_id::text) AS right_key,
    occasion_shown,
    question_dim,
    winner_side,
    response_time_ms,
    attention_check_pass
FROM survey_pairs
WHERE study_id = %(study_id)s
  AND question_dim = %(question_dim)s;
"""


def load_pairs(
    study_id: str,
    *,
    question_dim: str = "purchase_intent",
    exclude_failed_attention: bool = True,
) -> pd.DataFrame:
    df = pd.read_sql(
        _PAIRS_SQL, engine(), params={"study_id": study_id, "question_dim": question_dim}
    )
    if exclude_failed_attention:
        df = df[df["attention_check_pass"].fillna(True)]
    return df.reset_index(drop=True)


@dataclass
class BTResult:
    card_keys: list[str]
    scores: np.ndarray            # log-odds, mean-centred
    sale_scores: np.ndarray       # logistic(scores), in [0,1]
    n_comparisons: int
    n_cards: int
    n_iter: int
    converged: bool


def fit_bradley_terry(
    df: pd.DataFrame,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    prior_strength: float = 1.0,
) -> BTResult:
    """MM-fit Bradley-Terry on long-format pair df.

    `prior_strength` adds a Beta(prior_strength, prior_strength) pseudocount on
    every observed (i,j) pair, which keeps scores finite for cards that win
    or lose every comparison.
    """
    if df.empty:
        raise ValueError("No pairs to fit.")

    cards = sorted(set(df["left_key"]) | set(df["right_key"]))
    idx = {c: k for k, c in enumerate(cards)}
    n = len(cards)

    wins = np.zeros(n)
    pair_counts = np.zeros((n, n))

    for row in df.itertuples(index=False):
        i = idx[row.left_key]
        j = idx[row.right_key]
        if row.winner_side == "L":
            wins[i] += 1.0
        elif row.winner_side == "R":
            wins[j] += 1.0
        else:  # tie
            wins[i] += 0.5
            wins[j] += 0.5
        pair_counts[i, j] += 1
        pair_counts[j, i] += 1

    # Dirichlet-like prior pseudocounts to regularise extremes
    if prior_strength > 0:
        wins += prior_strength
        pair_counts += prior_strength * (1 - np.eye(n))

    s = np.zeros(n)
    converged = False
    for it in range(max_iter):
        exp_s = np.exp(s)
        # sum_j N_ij / (exp(s_i) + exp(s_j)) for each i
        denom_matrix = exp_s[:, None] + exp_s[None, :]
        np.fill_diagonal(denom_matrix, 1.0)  # avoid div-by-zero on diagonal
        ratios = pair_counts / denom_matrix
        np.fill_diagonal(ratios, 0.0)
        denom = ratios.sum(axis=1)
        # New log-scores
        new_s = np.log(np.clip(wins, 1e-12, None)) - np.log(np.clip(denom, 1e-12, None))
        new_s -= new_s.mean()  # identifiability: mean-centre

        delta = float(np.abs(new_s - s).max())
        s = new_s
        if delta < tol:
            converged = True
            log.info(f"BT converged at iter {it + 1}, max|Δs| = {delta:.2e}")
            break

    sale_scores = 1.0 / (1.0 + np.exp(-s))

    return BTResult(
        card_keys=cards,
        scores=s,
        sale_scores=sale_scores,
        n_comparisons=int(df.shape[0]),
        n_cards=n,
        n_iter=it + 1,
        converged=converged,
    )


def to_dataframe(result: BTResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "card_key": result.card_keys,
            "bt_score": result.scores,
            "sale_score": result.sale_scores,
        }
    )


# -- Persistence ------------------------------------------------------------

_UPSERT_LABEL = """
INSERT INTO saleability_labels (listing_id, label_source, score, raw)
SELECT %(listing_id)s::uuid, %(label_source)s, %(score)s, %(raw)s
WHERE EXISTS (SELECT 1 FROM listings WHERE listing_id = %(listing_id)s::uuid)
ON CONFLICT (listing_id, label_source) DO UPDATE
SET score = EXCLUDED.score,
    raw   = EXCLUDED.raw,
    created_at = NOW();
"""


def persist_bt_labels(
    result: BTResult,
    *,
    study_id: str,
    question_dim: str = "purchase_intent",
) -> int:
    """Write BT sale_score for each marketplace listing in the result.

    Generated-card scores are skipped here (they have no `listings` row); use
    a separate writer if you need to persist them on `generated_cards`.
    """
    from psycopg.types.json import Jsonb

    from common.db import connection

    label_source = f"survey_{study_id}_bt_{question_dim}"
    rows = [
        {
            "listing_id": k,
            "label_source": label_source,
            "score": float(result.sale_scores[i]),
            "raw": Jsonb({
                "bt_score": float(result.scores[i]),
                "question_dim": question_dim,
            }),
        }
        for i, k in enumerate(result.card_keys)
    ]
    with connection() as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT_LABEL, rows)
    return len(rows)
