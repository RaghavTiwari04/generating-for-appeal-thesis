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
    conditions: tuple[str, ...] = (
        "A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller",
    ),
) -> None:
    out_dir = Path(out_dir)

    all_cards = pd.read_sql(_GEN_SQL, engine(), params={"conditions": list(conditions)})
    log.info(f"Loaded {len(all_cards)} cards (conditions: {', '.join(conditions)})")

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
    ) -> None:
        export(Path(out))

    typer.run(cli)
