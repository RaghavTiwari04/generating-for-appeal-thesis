"""Recompute every corpus number the writeup quotes, and reconcile them.

Written to settle three inconsistencies in an earlier draft of the tables, all
since corrected: a distinct-design count that disagreed with the dedup
paragraph, per-subtype listings that did not sum to the corpus total, and more
labelled cards than distinct designs in two subtypes. The last of these is real
and survives: labelling ran before the final subtype pass and the final dedup,
so some labels now sit on listings that are not the representative the current
queries select. That is measured here rather than assumed.

Four counting bases run through this corpus, and quoting one without saying
which produces the appearance of drift between numbers that are all correct:

  listings              -- rows as scraped.
  designs, all listings -- one per duplicate cluster over the whole scrape.
      Deduplication runs there, not on the classified subset, so this is the
      figure the dedup paragraph needs and it exceeds the classified count.
  designs, classified   -- one per cluster among listings with a subtype. This
      is the population the corpus table and the funnel are about.
  label rows            -- one per labelled listing. Equal to designs when
      labelling runs against a settled corpus, and larger when it does not.

Per-subtype design counts use the representative's subtype, matching every
consumer of the corpus (label pool, LoRA stratification, the human reference
condition, both market-signal queries). Counting a design in every subtype its
listings fall into instead inflates the per-subtype sum above the group total;
both are reported so the difference is visible rather than surprising.

Per-source rows can also double-count a design listed on both marketplaces, so
they are not guaranteed to sum to the corpus total; `designs_on_both_sources`
measures whether that is happening at all.

Read-only. Run on the cluster with the services up:

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

    # Whether a per-source table can sum to the corpus total, or whether the
    # same artwork appears on both marketplaces and is counted in both rows.
    "designs_on_both_sources": f"""
        SELECT COUNT(*) AS n FROM (
            SELECT {DESIGN_KEY} AS design
            FROM listings l LEFT JOIN listing_features lf USING (listing_id)
            GROUP BY 1
            HAVING COUNT(DISTINCT l.source) > 1
        ) t
    """,

    "classified_designs_on_both_sources": f"""
        SELECT COUNT(*) AS n FROM (
            SELECT {DESIGN_KEY} AS design
            FROM listings l JOIN listing_features lf USING (listing_id)
            WHERE lf.occasion IS NOT NULL
            GROUP BY 1
            HAVING COUNT(DISTINCT l.source) > 1
        ) t
    """,
}

PER_SOURCE_SQL = f"""
SELECT l.source,
       COUNT(*)                                                     AS listings,
       COUNT(*) FILTER (WHERE lf.occasion IS NOT NULL)              AS classified,
       COUNT(DISTINCT {DESIGN_KEY})                                 AS distinct_designs,
       COUNT(DISTINCT CASE WHEN lf.occasion IS NOT NULL
                           THEN {DESIGN_KEY} END)                   AS classified_designs
FROM listings l
LEFT JOIN listing_features lf USING (listing_id)
GROUP BY l.source
ORDER BY listings DESC
"""

PER_SUBTYPE_SQL = f"""
WITH rep AS (
    -- One row per design: the representative, and the subtype it carries.
    -- This is the convention every consumer of the corpus uses.
    SELECT DISTINCT ON ({DESIGN_KEY})
           {DESIGN_KEY} AS design, l.listing_id, lf.occasion
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    WHERE lf.occasion IS NOT NULL
    ORDER BY {DESIGN_KEY}, l.listing_id
),
per_listing AS (
    SELECT lf.occasion,
           COUNT(*)                                   AS listings,
           COUNT(DISTINCT {DESIGN_KEY})               AS designs_any_subtype,
           COUNT(sl.listing_id)                       AS labelled,
           COUNT(DISTINCT CASE WHEN sl.listing_id IS NOT NULL
                               THEN {DESIGN_KEY} END) AS labelled_designs
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    LEFT JOIN saleability_labels sl
           ON sl.listing_id = l.listing_id AND sl.label_source = %(src)s
    WHERE lf.occasion IS NOT NULL
    GROUP BY lf.occasion
)
SELECT p.occasion,
       p.listings,
       COUNT(rep.design)      AS distinct_designs,
       p.designs_any_subtype,
       p.labelled,
       p.labelled_designs
FROM per_listing p
LEFT JOIN rep ON rep.occasion = p.occasion
GROUP BY p.occasion, p.listings, p.designs_any_subtype, p.labelled,
         p.labelled_designs
ORDER BY p.listings DESC
"""


def main() -> None:
    eng = engine()
    scalars: dict[str, int] = {}
    for name, sql in QUERIES.items():
        scalars[name] = int(pd.read_sql(sql, eng).iloc[0]["n"])
        print(f"{name:34s} {scalars[name]:>7,}")

    print("\n=== per source ===")
    src = pd.read_sql(PER_SOURCE_SQL, eng)
    print(src.to_string(index=False))
    print(f"sums: listings={src.listings.sum():,} classified={src.classified.sum():,} "
          f"designs={src.distinct_designs.sum():,}")

    print("\n=== per subtype ===")
    per = pd.read_sql(PER_SUBTYPE_SQL, eng, params={"src": LABEL_SOURCE})
    per["labelled_minus_designs"] = per.labelled - per.distinct_designs
    print(per.to_string(index=False))
    print(f"\ncolumn sums: listings={per.listings.sum():,} "
          f"distinct={per.distinct_designs.sum():,} "
          f"(any-subtype convention {per.designs_any_subtype.sum():,}) "
          f"labelled={per.labelled.sum():,}")
    phantom = int(per.designs_any_subtype.sum() - per.distinct_designs.sum())
    print(f"designs counted in more than one subtype: {phantom:,}")
    if int(per.distinct_designs.sum()) != scalars["distinct_designs_with_occasion"]:
        print("  NOTE: representative-convention sum should equal the "
              "corpus-wide classified design count")

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
    print(f"per-subtype labelled-design sum    {int(per.labelled_designs.sum()):>7,}")
    print(f"  labels - corpus-wide distinct    "
          f"{scalars['labelled_total'] - scalars['distinct_designs_labelled']:>7,}")
    print(f"  labels - per-subtype sum         "
          f"{scalars['labelled_total'] - int(per.labelled_designs.sum()):>7,}")
    print(f"  subtype sum - corpus-wide        "
          f"{int(per.labelled_designs.sum()) - scalars['distinct_designs_labelled']:>7,}"
          "   (designs whose cluster spans two subtypes)")
    print("These three gaps have different causes and the writeup must not quote")
    print("one as the explanation for another.")
    if scalars["labels_on_non_representatives"]:
        print("  This is why labelled can exceed distinct designs: some labels sit on")
        print("  listings that are not the representative selected now, so occasions or")
        print("  clusters changed after labelling ran.")

    emit_latex(src, per, scalars)

    print("\n=== LoRA training set, as the code computes it ===")
    n_images = 150
    subtypes = len(per)
    per_subtype = max(1, n_images // subtypes)
    print(f"n_images={n_images}, subtypes={subtypes} -> "
          f"{per_subtype} per subtype, {per_subtype * subtypes} total")


def emit_latex(src: pd.DataFrame, per: pd.DataFrame, scalars: dict[str, int]) -> None:
    """Print both corpus tables with the measured numbers, ready to paste."""
    print("\n=== LaTeX: Table 3.1 ===")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Source & Scraped & With subtype & Distinct designs \\")
    print(r"\midrule")
    for r in src.itertuples():
        # classified_designs, not distinct_designs: the column has to be drawn
        # from the same population as the one beside it, or a row reads as a
        # funnel whose last step goes up.
        print(f"{r.source} & {r.listings:,} & {r.classified:,} "
              f"& {r.classified_designs:,} " + r"\\")
    print(r"\midrule")
    print(f"Total & {scalars['total_listings']:,} & {scalars['listings_with_occasion']:,} "
          f"& {scalars['distinct_designs_with_occasion']:,} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    row_sum = int(src.classified_designs.sum())
    corpus = scalars["distinct_designs_with_occasion"]
    print(f"\n(design column: rows sum to {row_sum:,}, corpus-wide distinct is "
          f"{corpus:,}, difference {row_sum - corpus:,}; "
          f"{scalars['classified_designs_on_both_sources']:,} classified designs "
          f"appear on both marketplaces)")
    if row_sum != corpus and scalars["classified_designs_on_both_sources"] == 0:
        print("  UNEXPLAINED: rows disagree with the total but no design spans sources")

    print("\n=== LaTeX: Table 3.2 ===")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Subtype & Listings & Distinct designs & Labelled & Labelled designs \\")
    print(r"\midrule")
    for r in per.itertuples():
        name = r.occasion.replace("_", r"\_")
        print(f"\\texttt{{{name}}} & {r.listings:,} & {r.distinct_designs:,} "
              f"& {r.labelled:,} & {r.labelled_designs:,} " + r"\\")
    print(r"\midrule")
    print(f"Total & {per.listings.sum():,} & {per.distinct_designs.sum():,} "
          f"& {per.labelled.sum():,} & {per.labelled_designs.sum():,} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\n(per-subtype designs use the representative's subtype and sum to "
          f"{int(per.distinct_designs.sum()):,}, matching the classified corpus "
          f"as a whole)")

    print("\n=== LaTeX: the four counting bases ===")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Stage & Listings & Designs (all) & Designs (classified) & Label rows \\")
    print(r"\midrule")
    print(f"Scraped & {scalars['total_listings']:,} & "
          f"{scalars['distinct_designs_total']:,} & --- & --- " + r"\\")
    print(f"With a birthday subtype & {scalars['listings_with_occasion']:,} & --- & "
          f"{scalars['distinct_designs_with_occasion']:,} & --- " + r"\\")
    print(f"Labelled & --- & --- & {scalars['distinct_designs_labelled']:,} & "
          f"{scalars['labelled_total']:,} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
