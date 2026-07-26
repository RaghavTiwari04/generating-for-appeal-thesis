"""Explain why duplicate clusters were merged.

The gallery shows large clusters holding visibly different cards. Union-find
takes a transitive closure, so a cluster can be held together by a chain of
merely-similar pairs without any two ends resembling each other. This measures
that directly: for each cluster it recomputes pairwise pHash, CLIP and title
similarity, counts how many pairs actually clear each threshold, and reports
the weakest pair.

A genuine duplicate cluster is dense — most pairs clear a threshold. A chained
cluster is sparse, with roughly n-1 qualifying pairs holding n members together
and a very low weakest pair.

Read-only.

    python -m scripts.inspect_clusters --clusters 6
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typer

from common.db import engine
from data.features.dedup import (
    CLIP_COSINE_THRESHOLD,
    PHASH_HAMMING_THRESHOLD,
    TFIDF_THRESHOLD,
    hamming,
)

_SQL = """
SELECT lf.duplicate_cluster_id AS cluster_id,
       lf.duplicate_cluster_size AS cluster_size,
       lf.listing_id,
       l.title,
       l.source,
       lf.clip_embedding,
       li.phash
FROM listing_features lf
JOIN listings l USING (listing_id)
LEFT JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
WHERE lf.duplicate_cluster_size > 1
ORDER BY lf.duplicate_cluster_size DESC, lf.duplicate_cluster_id;
"""


def _as_vec(v) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, str):
        import json

        v = json.loads(v)
    return np.asarray(v, dtype=np.float32)


def _title_sim(a: str, b: str) -> float:
    """Word-level Jaccard — a cheap stand-in for the TF-IDF cosine."""
    wa, wb = set((a or "").lower().split()), set((b or "").lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def inspect(clusters: int, members_shown: int) -> None:
    df = pd.read_sql(_SQL, engine())
    if df.empty:
        print("No clusters found. Has dedup run?")
        return

    for cluster_id, g in list(df.groupby("cluster_id", sort=False))[:clusters]:
        n = len(g)
        if n < 2:
            continue
        embs = [_as_vec(v) for v in g["clip_embedding"]]
        phashes = [
            int(str(p), 2) if isinstance(p, str) else p for p in g["phash"]
        ]
        titles = list(g["title"])

        pairs = phash_links = clip_links = title_links = 0
        weakest_cos = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                if phashes[i] is not None and phashes[j] is not None:
                    if hamming(phashes[i], phashes[j]) <= PHASH_HAMMING_THRESHOLD:
                        phash_links += 1
                if embs[i] is not None and embs[j] is not None:
                    cos = float(np.dot(embs[i], embs[j]))
                    weakest_cos = min(weakest_cos, cos)
                    if cos >= CLIP_COSINE_THRESHOLD:
                        clip_links += 1
                if _title_sim(titles[i], titles[j]) >= TFIDF_THRESHOLD:
                    title_links += 1

        linked = max(phash_links, clip_links, title_links)
        density = linked / pairs if pairs else 0.0
        verdict = (
            "dense — genuine duplicates"
            if density > 0.5
            else "CHAINED — held together by a few links"
        )
        print(f"\n{'=' * 78}")
        print(f"cluster {str(cluster_id)[:8]}  members={n}  {verdict}")
        print(f"  pairs={pairs}  phash>=thr={phash_links}  clip>=thr={clip_links}  title>=thr={title_links}")
        print(f"  qualifying-pair density={density:.2f}  (a chain needs only {n - 1})")
        print(f"  weakest CLIP cosine between any two members={weakest_cos:.3f}")
        for t in titles[:members_shown]:
            print(f"      {(t or '')[:70]}")
        if n > members_shown:
            print(f"      ... {n - members_shown} more")


def main(clusters: int = 6, members_shown: int = 10) -> None:
    inspect(clusters, members_shown)


if __name__ == "__main__":
    typer.run(main)
