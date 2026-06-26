"""Export eval cards (A/B/C/D) from MinIO to a local gallery folder.

Usage (run on compute node with MinIO running):
    python -m eval.export_gallery --out /path/to/card_gallery
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

from common.db import engine
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)

_GEN_SQL = """
SELECT gc.card_id::text AS card_key,
       gc.condition_tag,
       gc.cover_path,
       gc.headline_text,
       COALESCE(gc.brief->'request'->>'occasion', gc.brief->>'occasion') AS occasion
FROM generated_cards gc
WHERE gc.condition_tag = ANY(%(conditions)s)
"""

_HUMAN_SQL = """
SELECT li.listing_id::text AS card_key,
       'D_human_bestseller' AS condition_tag,
       li.storage_path AS cover_path,
       l.title AS headline_text,
       lf.occasion
FROM listings l
JOIN listing_features lf USING (listing_id)
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN saleability_labels sl
  ON sl.listing_id = l.listing_id AND sl.label_source = 'vlm_5head_v1'
WHERE lf.occasion = ANY(%(occasions)s)
  AND l.is_bestseller = TRUE
ORDER BY COALESCE(sl.score, 0) DESC
LIMIT %(limit)s
"""


def _fetch_image_bytes(cover_path: str) -> bytes | None:
    try:
        if cover_path.startswith("s3://") or cover_path.startswith("greeting-cards"):
            return get_object(cover_path)
        return Path(cover_path).read_bytes()
    except Exception as e:
        log.warning(f"Failed to load {cover_path}: {e}")
        return None


def export(
    out_dir: Path,
    occasions: list[str],
    conditions: tuple[str, ...] = (
        "A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank",
    ),
    d_per_occasion: int = 5,
) -> None:
    out_dir = Path(out_dir)

    # Load A/B/C from generated_cards
    gen_df = pd.read_sql(_GEN_SQL, engine(), params={"conditions": list(conditions)})
    log.info(f"Loaded {len(gen_df)} generated cards (A/B/C)")

    # Load D from listings
    human_df = pd.read_sql(
        _HUMAN_SQL, engine(),
        params={"occasions": occasions, "limit": d_per_occasion * len(occasions)},
    )
    log.info(f"Loaded {len(human_df)} human bestseller cards (D)")

    all_cards = pd.concat([gen_df, human_df], ignore_index=True)

    exported = 0
    html_sections: dict[str, list[str]] = {}

    for _, row in all_cards.iterrows():
        cond = row["condition_tag"]
        occasion = row.get("occasion", "unknown") or "unknown"
        cover_path = row.get("cover_path")
        headline = row.get("headline_text", "") or ""

        if not cover_path:
            continue

        data = _fetch_image_bytes(cover_path)
        if not data:
            continue

        cond_dir = out_dir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        slug = occasion.replace("/", "_")
        digest = hashlib.sha256(data).hexdigest()[:8]
        fname = f"{slug}_{digest}.png"
        fpath = cond_dir / fname

        fpath.write_bytes(data)
        exported += 1

        if cond not in html_sections:
            html_sections[cond] = []
        html_sections[cond].append(
            f'<div style="text-align:center">'
            f'<img src="{cond}/{fname}" width=200><br>'
            f'<small>{occasion}<br>{headline[:30]}</small></div>'
        )

    # Write gallery.html
    html = '<html><body style="font-family:sans-serif;background:#111;color:#eee">'
    html += "<h1>Generated Cards Gallery</h1>"
    for cond in sorted(html_sections):
        html += f"<h2>{cond}</h2>"
        html += '<div style="display:flex;flex-wrap:wrap;gap:10px">'
        html += "".join(html_sections[cond])
        html += "</div>"
    html += "</body></html>"
    (out_dir / "gallery.html").write_text(html)

    log.info(f"Exported {exported} cards to {out_dir}")


if __name__ == "__main__":
    import typer

    def cli(
        out: str = "./card_gallery",
        occasions: str = "birthday/general,birthday/milestone,birthday/kids,birthday/relationship",
        d_per_occasion: int = 5,
    ) -> None:
        occ_list = [o.strip() for o in occasions.split(",")]
        export(Path(out), occ_list, d_per_occasion=d_per_occasion)

    typer.run(cli)
