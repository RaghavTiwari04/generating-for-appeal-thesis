"""Read-only audit: do stored occasion labels match the current keyword rules?

`listing_features.feature_version` cannot answer this — the classifier's
ON CONFLICT clause does not update that column, so pre-existing rows keep
whatever the complexity/palette passes last wrote.

So recompute labels with the rules that are in the tree right now and diff
them against what is in the database. Writes nothing.

    python -m scripts.audit_occasions
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from common.db import engine
from data.features.occasion_classifier import pick_best_occasion, weak_label

_SQL = """
SELECT l.listing_id,
       l.source,
       COALESCE(l.title, '')       AS title,
       COALESCE(l.description, '') AS description,
       lf.occasion                 AS stored
FROM listings l
JOIN listing_features lf USING (listing_id)
WHERE lf.occasion IS NOT NULL;
"""


def main() -> None:
    df = pd.read_sql(_SQL, engine())
    if df.empty:
        print("No rows with a stored occasion.")
        return

    text = (df["title"] + " " + df["description"]).str.strip()
    df["recomputed"] = [pick_best_occasion(weak_label(t)) for t in text]
    # Same text the classifier would see if it used titles only.
    df["title_only"] = [pick_best_occasion(weak_label(t)) for t in df["title"]]

    agree = (df["stored"] == df["recomputed"]).sum()
    total = len(df)
    print(f"\nRows with stored occasion : {total}")
    print(f"Match current keyword rules: {agree}  ({agree / total * 100:.1f}%)")
    print(f"Disagree                   : {total - agree}\n")

    print("=== stored vs recomputed (title + description) ===")
    print(pd.crosstab(df["stored"], df["recomputed"], dropna=False).to_string())

    print("\n=== label distribution: stored / recomputed / title-only ===")
    for name in ("stored", "recomputed", "title_only"):
        counts = Counter(df[name].dropna())
        print(f"\n{name}:")
        for occ, n in counts.most_common():
            print(f"  {occ:28s} {n}")

    bad = df[df["stored"] != df["recomputed"]]
    if not bad.empty:
        print("\n=== sample disagreements ===")
        for _, r in bad.head(20).iterrows():
            print(f"  [{r['source']}] stored={r['stored']} -> rules={r['recomputed']}")
            print(f"      {r['title'][:90]}")

    # Per-source agreement — a source at ~0% means its labels came from
    # something other than these rules.
    print("\n=== agreement by source ===")
    for source, grp in df.groupby("source"):
        a = (grp["stored"] == grp["recomputed"]).sum()
        print(f"  {source:20s} {a}/{len(grp)}  ({a / len(grp) * 100:.1f}%)")


if __name__ == "__main__":
    main()
