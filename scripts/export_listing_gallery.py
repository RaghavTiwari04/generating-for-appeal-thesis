"""Export scraped listings as an HTML gallery grouped by assigned occasion.

Lets you eyeball two things at once: whether the occasion classifier is
labelling sensibly, and whether the stored cover is flat artwork rather than a
tilted product mockup.

    python -m scripts.export_listing_gallery --out ./listing_gallery --per-occasion 40

Then copy the folder off the cluster and open index.html.
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
SELECT card.occasion, card.source, card.title, card.storage_path
FROM (
    SELECT lf.occasion,
           l.source,
           l.title,
           li.storage_path,
           ROW_NUMBER() OVER (PARTITION BY lf.occasion ORDER BY l.listing_id) AS rn
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
    WHERE li.storage_path IS NOT NULL
      AND (%(all_occasions)s OR lf.occasion LIKE 'birthday/%%')
) card
WHERE card.rn <= %(per_occasion)s
ORDER BY card.occasion, card.rn;
"""


def export(out_dir: Path, per_occasion: int, all_occasions: bool) -> None:
    df = pd.read_sql(
        _SQL, engine(),
        params={"per_occasion": per_occasion, "all_occasions": all_occasions},
    )
    if df.empty:
        print("No labelled listings with images found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, list[str]] = {}
    exported = failed = 0

    for _, row in df.iterrows():
        occasion = row["occasion"] or "unlabelled"
        try:
            data = get_object(row["storage_path"])
        except Exception as e:
            log.warning(f"Could not load {row['storage_path']}: {e}")
            failed += 1
            continue

        slug = occasion.replace("/", "_")
        sub = out_dir / slug
        sub.mkdir(parents=True, exist_ok=True)
        fname = f"{hashlib.sha256(data).hexdigest()[:10]}.png"
        (sub / fname).write_bytes(data)
        exported += 1

        title = (row["title"] or "")[:70].replace("<", "&lt;")
        sections.setdefault(occasion, []).append(
            f'<figure><img src="{slug}/{fname}" loading="lazy">'
            f'<figcaption>{title}<br><span class="src">{row["source"]}</span>'
            f'</figcaption></figure>'
        )

    html = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}",
        "h2{color:#8af;border-bottom:1px solid #333;padding-bottom:6px;margin-top:32px}",
        ".grid{display:flex;flex-wrap:wrap;gap:14px}",
        "figure{margin:0;width:190px;background:#1b1b1b;border-radius:8px;padding:8px}",
        "img{width:100%;border-radius:4px;display:block}",
        "figcaption{font-size:11px;color:#bbb;margin-top:6px;line-height:1.35}",
        ".src{color:#666}",
        "</style></head><body>",
        f"<h1>Scraped listings by occasion ({exported} images)</h1>",
    ]
    for occasion in sorted(sections):
        html.append(f"<h2>{occasion} — {len(sections[occasion])} shown</h2>")
        html.append('<div class="grid">' + "".join(sections[occasion]) + "</div>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"Exported {exported} images to {out_dir} ({failed} failed)")
    for occasion in sorted(sections):
        print(f"  {occasion:28s} {len(sections[occasion])}")


def main(
    out: str = "./listing_gallery",
    per_occasion: int = 40,
    all_occasions: bool = typer.Option(False, help="Include non-birthday occasions too"),
) -> None:
    export(Path(out), per_occasion, all_occasions)


if __name__ == "__main__":
    typer.run(main)
