"""Mine market signals to inject into the brief-generator prompt.

Four signals:
- top tropes for the occasion: cluster the top-saleability listings via
  CLIP embeddings, surface the cluster headline themes.
- bestseller subjects: frequency-ranked visual subjects from bestseller
  titles/descriptions — gives the LLM a data-grounded palette to riff on.
- coverage gaps: (occasion × tone) pairs underrepresented in the dataset.
- longevity caution: short prompt-side warning about time-bound references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)

LONGEVITY_CAUTION = (
    "Avoid current-events references, brand mentions, or anything that will "
    "feel dated within twelve months."
)

@dataclass
class MarketSignals:
    top_tropes: list[str]
    bestseller_subjects: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    longevity_caution: str = LONGEVITY_CAUTION


# One listing per duplicate cluster, as everywhere else that ranks by score.
#
# It matters more here than most: these rows are k-means clustered into five
# "tropes" that describe the occasion to the brief. Colourways of one artwork
# sit on top of each other in embedding space, so without the collapse they
# form clusters of their own and a single design becomes several of the five
# themes the market is said to contain.
_TOP_BY_OCCASION_SQL = """
SELECT listing_id, clip_embedding, headline_text
FROM (
    SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, lf.listing_id::text))
           lf.listing_id,
           lf.clip_embedding,
           COALESCE(lf.extracted_text, l.title) AS headline_text,
           COALESCE(sl.score, 0) AS score
    FROM listing_features lf
    JOIN listings l USING (listing_id)
    LEFT JOIN saleability_labels sl
      ON sl.listing_id = lf.listing_id AND sl.label_source = 'llm_ssr_rubric_v2'
    WHERE lf.occasion = %(occasion)s
      AND lf.clip_embedding IS NOT NULL
    ORDER BY COALESCE(lf.duplicate_cluster_id::text, lf.listing_id::text),
             COALESCE(sl.score, 0) DESC,
             lf.listing_id
) representatives
ORDER BY score DESC
LIMIT %(limit)s;
"""


def top_tropes_for_occasion(occasion: str, *, k_clusters: int = 5, top_n: int = 100) -> list[str]:
    df = pd.read_sql(
        _TOP_BY_OCCASION_SQL,
        engine(),
        params={"occasion": occasion, "limit": top_n},
    )
    if df.empty:
        return []
    embeds = np.stack(df["clip_embedding"].apply(np.asarray).to_list())
    k = min(k_clusters, len(df))
    if k < 2:
        return [str(t)[:160] for t in df["headline_text"].head(5).tolist() if t]

    km = KMeans(n_clusters=k, n_init=4, random_state=0)
    labels = km.fit_predict(embeds)
    df["cluster"] = labels
    tropes: list[str] = []
    for c in range(k):
        sub = df[df["cluster"] == c]
        sample = (
            sub["headline_text"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .head(3)
            .tolist()
        )
        if sample:
            tropes.append(f"cluster_{c}: {' | '.join(s[:80] for s in sample)}")
    return tropes


# One title per duplicate cluster, the cluster's best-scoring member standing
# for it.
#
# This list is the brief's inspiration palette, and the prompt tells the model
# to draw from all of it rather than the top few. Print-on-demand catalogues
# carry one artwork in many colourways, which score near-identically because
# they are the same artwork, so without the collapse the top 30 could be a
# dozen designs listed twice or three times each — a narrower palette than the
# count suggests, and one that pushes every candidate toward the same subject.
_TOP_TITLES_SQL = """
SELECT title, score
FROM (
    SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text))
           l.title,
           COALESCE(sl.score, 0) AS score
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    LEFT JOIN saleability_labels sl
      ON sl.listing_id = l.listing_id AND sl.label_source = 'llm_ssr_rubric_v2'
    WHERE lf.occasion = %(occasion)s
      AND l.title IS NOT NULL
    ORDER BY COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text),
             COALESCE(sl.score, 0) DESC,
             l.listing_id
) representatives
ORDER BY score DESC
LIMIT %(limit)s;
"""


@lru_cache(maxsize=32)
def subject_pool_size(occasion: str, default: int = 15) -> int:
    """How many distinct bestseller subjects an occasion offers.

    Callers rotate a subject index across candidates and need to know where to
    wrap. Cached because it is a database round trip that returns the same
    answer for the whole run, and it sits on the path of every generation.
    """
    try:
        return max(len(bestseller_subjects_for_occasion(occasion, limit=30)), 5)
    except Exception as e:
        log.warning(f"Could not size the bestseller pool for {occasion} ({e})")
        return default


def bestseller_subjects_for_occasion(occasion: str, limit: int = 30) -> list[str]:
    """Return top-scoring listing titles for this occasion, as LLM inspiration."""
    df = pd.read_sql(_TOP_TITLES_SQL, engine(), params={"occasion": occasion, "limit": limit})
    if df.empty:
        return []
    results = []
    for _, row in df.iterrows():
        title = (row["title"] or "").strip()
        title = title.replace(" Greeting Card", "").replace(" greeting card", "").strip()
        if title:
            results.append(f"{title} (score: {row['score']:.2f})")
    return results


_GAP_SQL = """
SELECT lf.occasion, COUNT(*) AS n
FROM listing_features lf
GROUP BY lf.occasion
ORDER BY n ASC
LIMIT 5;
"""


def coverage_gaps() -> list[str]:
    df = pd.read_sql(_GAP_SQL, engine())
    return [f"low_volume:{r.occasion} (n={int(r.n)})" for r in df.itertuples()]


def gather(occasion: str) -> MarketSignals:
    try:
        tropes = top_tropes_for_occasion(occasion)
    except Exception:
        tropes = []
    try:
        subjects = bestseller_subjects_for_occasion(occasion)
    except Exception:
        subjects = []
    try:
        gaps = coverage_gaps()
    except Exception:
        gaps = []
    return MarketSignals(top_tropes=tropes, bestseller_subjects=subjects, coverage_gaps=gaps)


def render_for_prompt(signals: MarketSignals) -> dict[str, str]:
    if signals.bestseller_subjects:
        bs = "\n".join(f"{i+1}. {s}" for i, s in enumerate(signals.bestseller_subjects))
    else:
        bs = "  (none yet)"
    return {
        "top_tropes": "\n".join(f"- {t}" for t in signals.top_tropes) or "  (none yet)",
        "bestseller_subjects": bs,
        "coverage_gaps": "\n".join(f"- {g}" for g in signals.coverage_gaps) or "  (none yet)",
        "longevity_caution": signals.longevity_caution,
    }
