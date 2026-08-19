"""Count how many duplicate pairs in the label pool straddle the train/test split.

Deduplication exists partly so that near-identical images cannot appear on both
sides of the split. One representative per duplicate cluster normally makes that
structurally impossible. Labelling ran before the corpus settled, so a number of
labels sit on listings that are not the representative the current queries pick,
and those are the rows where the guarantee does not hold: each shares a cluster
with another labelled listing.

Whether that costs anything depends on where the members land, which is a
property of the split rather than of the labels. Seller grouping keeps a
seller's listings together, so a pair from one seller cannot separate. Greetings
Island records no seller at all (`greetings_island.py`: "GI doesn't have
individual sellers") and holds most of the corpus's duplication, so both members
of such a pair become their own group and are free to separate. For two
independent groups under a 70/15/15 split the chance of landing on the same side
is 0.7^2 + 0.15^2 + 0.15^2, about 0.535, so a little under half of those pairs
are expected to straddle.

This measures it rather than assuming it. Reports, per duplicate cluster with
more than one labelled listing: how many members, which sides they land on, and
whether the cluster is split. The count that matters is `clusters straddling`,
and the rows it covers bound how optimistic the held-out correlation is.

Read-only. Needs the database, so run on a compute node with services up:

    source cluster/jobs/_start_services.sh
    python -m scripts.check_split_leakage
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from common.db import engine
from models.predictor.dataset import SplitConfig, load_training_frame, split_by_seller

LABEL_SOURCE = "llm_ssr_rubric_v2"

CLUSTER_SQL = """
SELECT l.listing_id,
       lf.duplicate_cluster_id,
       l.source,
       l.seller_id
FROM listings l
JOIN listing_features lf USING (listing_id)
WHERE lf.duplicate_cluster_id IS NOT NULL
"""


def main() -> None:
    eng = engine()
    df = load_training_frame()
    splits = split_by_seller(df, SplitConfig(seed=42))
    side_of: dict[str, str] = {}
    for name, rows in splits.items():
        for listing_id in rows["listing_id"].astype(str):
            side_of[listing_id] = name
    print(f"split: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))

    clusters = pd.read_sql(CLUSTER_SQL, eng)
    clusters["listing_id"] = clusters["listing_id"].astype(str)
    # Only listings that carry a label are in the pool the split partitions.
    in_pool = clusters[clusters.listing_id.isin(side_of)]

    members: dict[str, list[str]] = defaultdict(list)
    for row in in_pool.itertuples():
        members[str(row.duplicate_cluster_id)].append(row.listing_id)

    multi = {c: m for c, m in members.items() if len(m) > 1}
    straddling = {c: m for c, m in multi.items()
                  if len({side_of[x] for x in m}) > 1}

    print(f"\nlabelled listings in a duplicate cluster   {len(in_pool):>6,}")
    print(f"clusters holding more than one of them     {len(multi):>6,}")
    print(f"  of those, split across sides             {len(straddling):>6,}")
    exposed = sum(len(m) for m in straddling.values())
    print(f"  label rows in a straddling cluster       {exposed:>6,}"
          f"  ({exposed / max(len(df), 1):.2%} of the pool)")

    if not multi:
        print("\nNo cluster holds more than one labelled listing: the "
              "one-representative-per-cluster rule held, and no duplicate can "
              "cross the split.")
        return
    if not straddling:
        print("\nEvery such cluster fell entirely on one side, so no "
              "near-duplicate pair crosses the split. This is a property of "
              "this seed rather than a guarantee: the members are independent "
              "groups wherever the source records no seller.")
        return

    print("\nStraddling clusters, by source and side:")
    src = dict(zip(in_pool.listing_id, in_pool.source))
    for cluster, m in sorted(straddling.items())[:20]:
        sides = ", ".join(f"{x[:8]}:{side_of[x]}" for x in m)
        print(f"  {cluster[:12]:12s} [{src.get(m[0], '?')}] {sides}")
    if len(straddling) > 20:
        print(f"  ... and {len(straddling) - 20:,} more")

    print("\nThese are near-identical images on both sides of the split. The "
          "held-out correlation is optimistic by an amount bounded by the row "
          "count above, and this is the failure deduplication exists to "
          "prevent.")


if __name__ == "__main__":
    main()
