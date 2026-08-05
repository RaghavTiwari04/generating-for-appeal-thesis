"""Export duplicate clusters as an HTML gallery, one row per cluster.

Dedup collapses 3905 listings to 2446 distinct designs, and nothing downstream
will use that unless the clusters are trustworthy. Seeing the members side by
side is the only practical check: genuine print-on-demand repeats and
personalised series (same artwork, different name) should look alike, while
over-merged clusters will obviously not.

    python -m scripts.export_cluster_gallery --out ./cluster_gallery
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import typer

from common.db import engine
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)

_SQL = """
SELECT ranked.duplicate_cluster_id AS cluster_id,
       ranked.duplicate_cluster_size AS cluster_size,
       ranked.title,
       ranked.source,
       ranked.occasion,
       ranked.storage_path
FROM (
    SELECT lf.duplicate_cluster_id,
           lf.duplicate_cluster_size,
           l.title,
           l.source,
           lf.occasion,
           li.storage_path,
           ROW_NUMBER() OVER (
               PARTITION BY lf.duplicate_cluster_id ORDER BY l.listing_id
           ) AS member_rank,
           DENSE_RANK() OVER (
               ORDER BY lf.duplicate_cluster_size DESC, lf.duplicate_cluster_id
           ) AS cluster_rank
    FROM listing_features lf
    JOIN listings l USING (listing_id)
    JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
    WHERE lf.duplicate_cluster_size > 1
      AND li.storage_path IS NOT NULL
) ranked
WHERE ranked.member_rank <= %(per_cluster)s
  AND ranked.cluster_rank <= %(clusters)s
-- Inner column names, not the outer aliases: ranked exposes the
-- subquery's own names here.
ORDER BY ranked.duplicate_cluster_size DESC,
         ranked.duplicate_cluster_id,
         ranked.member_rank;
"""


def export(out_dir: Path, clusters: int, per_cluster: int) -> None:
    df = pd.read_sql(
        _SQL, engine(), params={"clusters": clusters, "per_cluster": per_cluster}
    )
    if df.empty:
        print("No duplicate clusters found. Has dedup run?")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    exported = failed = 0

    for _cluster_id, group in df.groupby("cluster_id", sort=False):
        size = int(group["cluster_size"].iloc[0])
        occasion = group["occasion"].iloc[0] or "unlabelled"
        cards = []
        for _, row in group.iterrows():
            try:
                data = get_object(row["storage_path"])
            except Exception as e:
                log.warning(f"Could not load {row['storage_path']}: {e}")
                failed += 1
                continue
            fname = f"{hashlib.sha256(data).hexdigest()[:10]}.png"
            (out_dir / fname).write_bytes(data)
            exported += 1
            title = (row["title"] or "")[:44].replace("<", "&lt;")
            cards.append(
                f'<figure><img src="{fname}" loading="lazy">'
                f'<figcaption>{title}<br><span class="src">{row["source"]}</span>'
                f"</figcaption></figure>"
            )
        if not cards:
            continue
        shown = len(cards)
        more = f" — showing {shown} of {size}" if size > shown else ""
        rows.append(
            f'<section><h2>{size} members{more} '
            f'<span class="occ">{occasion}</span></h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    html = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}",
        "h1{border-bottom:2px solid #444;padding-bottom:10px}",
        "section{margin-top:26px;border-top:1px solid #2a2a2a;padding-top:12px}",
        "h2{color:#8af;font-size:15px;margin:0 0 10px}",
        ".occ{color:#666;font-weight:400;font-size:12px;margin-left:8px}",
        ".grid{display:flex;flex-wrap:wrap;gap:10px}",
        "figure{margin:0;width:150px;background:#1b1b1b;border-radius:6px;padding:6px}",
        "img{width:100%;border-radius:3px;display:block}",
        "figcaption{font-size:10px;color:#bbb;margin-top:5px;line-height:1.3}",
        ".src{color:#666}",
        "</style></head><body>",
        f"<h1>Duplicate clusters ({len(rows)} shown, {exported} images)</h1>",
        "<p style='color:#999;font-size:12px'>Members of one cluster should be the "
        "same design — print-on-demand repeats, or a personalised series differing "
        "only by name. Visibly unrelated cards in a cluster mean the similarity "
        "thresholds are too permissive.</p>",
    ]
    html.extend(rows)
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"Exported {len(rows)} clusters, {exported} images to {out_dir} ({failed} failed)")


def main(
    out: str = "./cluster_gallery",
    clusters: int = 40,
    per_cluster: int = 12,
) -> None:
    export(Path(out), clusters, per_cluster)


if __name__ == "__main__":
    typer.run(main)
