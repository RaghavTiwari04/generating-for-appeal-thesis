"""Recompute every corpus number the writeup quotes, and reconcile them.

Three inconsistencies were found in the reported tables and none of them can be
settled without the database:

  1. Table 3.1 gives 2,463 distinct designs, but the dedup paragraph's
     "3,491 listings, 1,111 redundant copies" implies 2,380. An 83-design gap.
  2. Table 3.2's per-subtype listings sum to 3,484 against Table 3.1's 3,491.
  3. Table 3.2 reports more labelled cards than distinct designs for
     `general` (2,020 vs 2,011) and `milestone` (137 vs 133), which one
     representative per cluster makes impossible.

Note the shape of (3): the per-subtype differences are +9, -2, +4, -6, summing
to +5, and the totals differ by exactly 5 (2,468 against 2,463). That is what
cross-subtype migration looks like, and the likely cause is that occasions were
reassigned by the NLI pass, or clusters recomputed, after labelling ran. This
script tests that directly by counting labels against the occasion recorded now
and against cluster representative status.

Run on the cluster with the services up:

    source cluster/jobs/_start_services.sh
    python -m scripts.verify_corpus_numbers
"""

from __future__ import annotations

import pandas as pd

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)

LABEL_SOURCE = "llm_ssr_rubric_v2"

# One design = one duplicate cluster, falling back to the listing itself when it
# was never clustered. This is the expression every consumer of the corpus uses.
DESIGN_KEY = "COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text)"

QUERIES: dict[str, str] = {
    "total_listings": "SELECT COUNT(*) AS n FROM listings",

    "listings_with_occasion": """
        SELECT COUNT(*) AS n
        FROM listings l JOIN listing_features lf USING (listing_id)
        WHERE lf.occasion IS NOT NULL
    """,

    "listings_without_occasion": """
        SELECT COUNT(*) AS n
        FROM listings l LEFT JOIN listing_features lf USING (listing_id)
        WHERE lf.occasion IS NULL
    """,

    "distinct_designs_total": f"""
        SELECT COUNT(DISTINCT {DESIGN_KEY}) AS n
        FROM listings l LEFT JOIN listing_features lf USING (listing_id)
    """,

    "distinct_designs_with_occasion": f"""
        SELECT COUNT(DISTINCT {DESIGN_KEY}) AS n
        FROM listings l JOIN listing_features lf USING (listing_id)
        WHERE lf.occasion IS NOT NULL
    """,

    # The dedup paragraph's three numbers, recomputed from what is stored.
    "clustered_listings": """
        SELECT COUNT(*) AS n FROM listing_features
        WHERE duplicate_cluster_id IS NOT NULL
    """,
    "clusters": """
        SELECT COUNT(DISTINCT duplicate_cluster_id) AS n FROM listing_features
        WHERE duplicate_cluster_id IS NOT NULL
    """,

    "labelled_total": f"""
        SELECT COUNT(*) AS n FROM saleability_labels
        WHERE label_source = '{LABEL_SOURCE}'
    """,

    # Labels whose listing is NOT the representative its consumers would pick.
    # Any row here is a label that the one-per-cluster rule should have excluded,
    # and is the direct explanation for labelled exceeding distinct.
    "labels_on_non_representatives": f"""
        WITH rep AS (
            SELECT DISTINCT ON ({DESIGN_KEY})
                   l.listing_id, {DESIGN_KEY} AS design
            FROM listings l
            JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
            LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
            WHERE li.storage_path IS NOT NULL AND lf.occasion IS NOT NULL
            ORDER BY {DESIGN_KEY}, l.listing_id
        )
        SELECT COUNT(*) AS n
        FROM saleability_labels sl
        WHERE sl.label_source = '{LABEL_SOURCE}'
          AND sl.listing_id NOT IN (SELECT listing_id FROM rep)
    """,

    "distinct_designs_labelled": f"""
        SELECT COUNT(DISTINCT {DESIGN_KEY}) AS n
        FROM saleability_labels sl
        JOIN listings l ON l.listing_id = sl.listing_id
        LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
        WHERE sl.label_source = '{LABEL_SOURCE}'
    """,
}

PER_SUBTYPE_SQL = f"""
SELECT lf.occasion,
       COUNT(*)                                   AS listings,
       COUNT(DISTINCT {DESIGN_KEY})               AS distinct_designs,
       COUNT(sl.listing_id)                       AS labelled,
       COUNT(DISTINCT CASE WHEN sl.listing_id IS NOT NULL
                           THEN {DESIGN_KEY} END) AS labelled_designs
FROM listings l
JOIN listing_features lf USING (listing_id)
LEFT JOIN saleability_labels sl
       ON sl.listing_id = l.listing_id AND sl.label_source = %(src)s
WHERE lf.occasion IS NOT NULL
GROUP BY lf.occasion
ORDER BY listings DESC
"""


def main() -> None:
    eng = engine()
    scalars: dict[str, int] = {}
    for name, sql in QUERIES.items():
        scalars[name] = int(pd.read_sql(sql, eng).iloc[0]["n"])
        print(f"{name:34s} {scalars[name]:>7,}")

    print("\n=== per subtype ===")
    per = pd.read_sql(PER_SUBTYPE_SQL, eng, params={"src": LABEL_SOURCE})
    per["labelled_minus_designs"] = per.labelled - per.distinct_designs
    print(per.to_string(index=False))
    print(f"\ncolumn sums: listings={per.listings.sum():,} "
          f"distinct={per.distinct_designs.sum():,} labelled={per.labelled.sum():,}")

    print("\n=== reconciliation ===")
    clustered = scalars["clustered_listings"]
    clusters = scalars["clusters"]
    total = scalars["total_listings"]
    redundant = clustered - clusters
    unclustered = total - clustered
    print(f"clustered listings                 {clustered:>7,}")
    print(f"clusters (designs among those)     {clusters:>7,}")
    print(f"redundant copies = clustered-clusters {redundant:>4,}")
    print(f"unclustered listings               {unclustered:>7,}")
    print(f"implied distinct = unclustered + clusters = {unclustered + clusters:,}")
    print(f"measured distinct designs total          = {scalars['distinct_designs_total']:,}")
    if unclustered + clusters != scalars["distinct_designs_total"]:
        print("  MISMATCH: the dedup paragraph and the corpus table disagree")

    print(f"\nlabels on non-representatives      {scalars['labels_on_non_representatives']:>7,}")
    print(f"labelled rows                      {scalars['labelled_total']:>7,}")
    print(f"distinct designs labelled          {scalars['distinct_designs_labelled']:>7,}")
    if scalars["labels_on_non_representatives"]:
        print("  This is why labelled can exceed distinct designs: some labels sit on")
        print("  listings that are not the representative selected now, so occasions or")
        print("  clusters changed after labelling ran.")

    print("\n=== LoRA training set, as the code computes it ===")
    n_images = 150
    subtypes = len(per)
    per_subtype = max(1, n_images // subtypes)
    print(f"n_images={n_images}, subtypes={subtypes} -> "
          f"{per_subtype} per subtype, {per_subtype * subtypes} total")


if __name__ == "__main__":
    main()
