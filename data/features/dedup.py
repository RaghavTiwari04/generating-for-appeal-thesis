"""Three-stage deduplication pipeline.

Stage 1: perceptual hash near-duplicate clustering   (cheap, image-only)
Stage 2: CLIP cosine > 0.95 semantic duplicate       (uses pgvector kNN)
Stage 3: TF-IDF title+description cosine > 0.85       (text-only fallback)

We assign each listing to a single canonical cluster_id and record cluster size
on `listing_features`. Canonical = highest combined engagement (review_count +
favourite_count).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass

import imagehash
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.db import connection
from common.logging import get_logger

log = get_logger(__name__)

PHASH_HAMMING_THRESHOLD = 6   # ~10% of 64-bit hash
CLIP_COSINE_THRESHOLD = 0.95
TFIDF_THRESHOLD = 0.85


@dataclass
class DedupStats:
    listings_total: int
    clusters: int
    duplicates: int


# ---------------------------------------------------------------------------
# Stage 1: pHash
# ---------------------------------------------------------------------------
def compute_phash(img: Image.Image | bytes) -> int:
    from io import BytesIO

    if isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))
    img = img.convert("RGB")
    h = imagehash.phash(img)
    return int(str(h), 16)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Stage 2: CLIP semantic dup (pgvector kNN)
# ---------------------------------------------------------------------------
_CLIP_NEIGHBOURS = """
SELECT lf2.listing_id AS neighbour_id,
       1 - (lf1.clip_embedding <=> lf2.clip_embedding) AS cosine
FROM listing_features lf1
JOIN listing_features lf2
  ON lf2.listing_id <> lf1.listing_id
WHERE lf1.listing_id = %(listing_id)s
  AND lf2.clip_embedding IS NOT NULL
ORDER BY lf1.clip_embedding <=> lf2.clip_embedding
LIMIT %(k)s;
"""


def find_clip_neighbours(listing_id: str, *, k: int = 20) -> list[tuple[str, float]]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_CLIP_NEIGHBOURS, {"listing_id": listing_id, "k": k})
        # str(): psycopg returns UUID objects, and the union-find is keyed by
        # string. Mixing the two puts one listing in under two keys.
        return [(str(r["neighbour_id"]), float(r["cosine"])) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Stage 3: TF-IDF text dup
# ---------------------------------------------------------------------------
def tfidf_duplicates(rows: list[tuple[str, str]], threshold: float = TFIDF_THRESHOLD):
    """rows: [(listing_id, text)]. Yields (id_a, id_b, sim) above threshold."""
    if len(rows) < 2:
        return
    ids = [r[0] for r in rows]
    texts = [r[1] or "" for r in rows]
    n = len(texts)
    # Scale min_df / max_df for small corpora to avoid empty vocabulary
    min_df = max(1, min(2, n - 1))
    max_df = 1.0 if n < 10 else 0.9
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, 2)).fit(texts)
    mat = vec.transform(texts)
    sim = cosine_similarity(mat, dense_output=False)
    rows_, cols_ = sim.nonzero()
    for r, c in zip(rows_, cols_, strict=False):
        if r >= c:
            continue
        s = sim[r, c]
        if s >= threshold:
            yield ids[r], ids[c], float(s)


# ---------------------------------------------------------------------------
# Union-Find for clustering
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_dedup(limit: int | None = None) -> DedupStats:
    """Materialise duplicate clusters into listing_features.duplicate_cluster_id."""
    uf = UnionFind()

    log.info("Dedup stage 1/3: pHash")
    with connection() as conn, conn.cursor() as cur:
        # Stage 1: pHash. Already stored at ingest (listing_images.phash).
        cur.execute(
            "SELECT listing_id, phash FROM listing_images WHERE is_primary AND phash IS NOT NULL"
        )
        rows = cur.fetchall()
        by_top = defaultdict(list)
        for r in rows:
            phash = int(str(r["phash"]), 2) if isinstance(r["phash"], str) else r["phash"]
            by_top[phash >> 56].append((str(r["listing_id"]), phash))
        for bucket in by_top.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    if hamming(bucket[i][1], bucket[j][1]) <= PHASH_HAMMING_THRESHOLD:
                        uf.union(bucket[i][0], bucket[j][0])

        log.info(f"  pHash: {len(rows)} primary images compared")

        # Stage 2: CLIP semantic
        log.info("Dedup stage 2/3: CLIP nearest neighbours")
        cur.execute("SELECT listing_id FROM listing_features WHERE clip_embedding IS NOT NULL")
        for r in cur.fetchall():
            for neighbour, cos in find_clip_neighbours(str(r["listing_id"]), k=10):
                if cos >= CLIP_COSINE_THRESHOLD:
                    uf.union(str(r["listing_id"]), neighbour)

        # Stage 3: TF-IDF on title+description (small batches)
        log.info("Dedup stage 3/3: TF-IDF over titles and descriptions")
        # Title only. Marketplace descriptions are vendor boilerplate — every
        # Greetings Island card repeats "...card template you can print or send
        # online as eCard", every Redbubble one "Digitally printed cards on
        # heavyweight stock..." — so TF-IDF over them scores near-identical for
        # unrelated cards and merged whole sources into one cluster.
        cur.execute("SELECT listing_id, COALESCE(title,'') AS t FROM listings")
        text_rows = [(str(r["listing_id"]), r["t"]) for r in cur.fetchall()]
        for a, b, _ in tfidf_duplicates(text_rows):
            uf.union(a, b)

        # Materialise clusters — must use ALL listings ever added to the union-find,
        # not just the pHash rows (which excludes image-less listings processed only
        # in CLIP or TF-IDF stages).
        clusters: dict[str, list[str]] = defaultdict(list)
        for lid in uf.parent:
            clusters[uf.find(lid)].append(lid)

        # Cluster id = stable UUID derived from canonical member
        # Canonical = max engagement (review_count + favourite_count NULLs as 0)
        cur.execute(
            "SELECT listing_id, "
            "COALESCE(review_count,0) + COALESCE(favourite_count,0) AS engagement "
            "FROM listings"
        )
        engagement = {str(r["listing_id"]): r["engagement"] for r in cur.fetchall()}

        updates = []
        for members in clusters.values():
            canonical = max(members, key=lambda m: engagement.get(m, 0))
            cluster_id = str(uuid.uuid5(uuid.NAMESPACE_OID, canonical))
            size = len(members)
            for m in members:
                updates.append((cluster_id, size, m))

        log.info(f"  {len(clusters)} clusters covering {len(updates)} listings")

        # Clear previous assignments first. The upsert below only touches
        # listings in this run's clusters, so a listing clustered by an earlier
        # run kept its old id and size — leaving the table a mix of runs. A
        # stale 671-member cluster survived this way after the merge bug was
        # fixed.
        cur.execute(
            "UPDATE listing_features "
            "SET duplicate_cluster_id = NULL, duplicate_cluster_size = NULL "
            "WHERE duplicate_cluster_id IS NOT NULL;"
        )
        cur.executemany(
            """
            INSERT INTO listing_features (listing_id, duplicate_cluster_id, duplicate_cluster_size, feature_version)
            VALUES (%s, %s, %s, 'dedup-v1')
            ON CONFLICT (listing_id) DO UPDATE
            SET duplicate_cluster_id   = EXCLUDED.duplicate_cluster_id,
                duplicate_cluster_size = EXCLUDED.duplicate_cluster_size,
                computed_at            = NOW();
            """,
            [(lid, cid, sz) for (cid, sz, lid) in updates],
        )

    # Redundant copies, not cluster members: a cluster of 5 contributes 4.
    duplicates = len(updates) - len(clusters)
    log.info(
        f"Dedup complete: {len(updates)} listings in {len(clusters)} clusters, "
        f"{duplicates} redundant copies, {len(clusters)} distinct designs"
    )
    largest = max((len(m) for m in clusters.values()), default=0)
    if largest > 50:
        log.warning(
            f"Largest cluster has {largest} members — union-find takes a transitive "
            "closure, so a permissive threshold chains unrelated cards together"
        )
    return DedupStats(
        listings_total=len(updates),
        clusters=len(clusters),
        duplicates=duplicates,
    )


def main(limit: int | None = None) -> None:
    stats = run_dedup(limit)
    # typer.run does not print a return value, so the job logged nothing at all.
    print(
        f"Dedup: {stats.listings_total} listings, {stats.clusters} clusters, "
        f"{stats.duplicates} duplicates"
    )


if __name__ == "__main__":
    import typer

    typer.run(main)
