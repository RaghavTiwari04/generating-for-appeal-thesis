"""Export eval cards (A/B/C/D) from MinIO to a local gallery folder with LLM scores.

Usage (run on compute node with MinIO running):
    python -m eval.export_gallery --out /path/to/card_gallery
    python -m eval.export_gallery --out /path/to/card_gallery --ratings ./artifacts/llm_system_eval/raw_ratings.csv
"""
from __future__ import annotations

import hashlib
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

# Rank WITHIN each occasion — a global ORDER BY score fills the sample from
# the highest-scoring occasions and drops the rest, so the gallery would not
# show the same occasion mix as the balanced A/B/C conditions.
_HUMAN_SQL = """
SELECT card_key, condition_tag, cover_path, headline_text, occasion
FROM (
    SELECT li.listing_id::text AS card_key,
           'D_human_bestseller' AS condition_tag,
           li.storage_path AS cover_path,
           l.title AS headline_text,
           lf.occasion,
           ROW_NUMBER() OVER (
               PARTITION BY lf.occasion
               ORDER BY COALESCE(sl.score, 0) DESC, l.listing_id
           ) AS rn
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
    LEFT JOIN saleability_labels sl
      ON sl.listing_id = l.listing_id AND sl.label_source = 'vlm_5head_v1'
    WHERE lf.occasion = ANY(%(occasions)s)
) ranked
WHERE rn <= %(per_occasion)s
"""

DIMS = ["purchase_intent", "occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness"]
DIM_SHORT = {"purchase_intent": "PI", "occasion_fit": "OF", "aesthetic": "AE",
             "emotional_resonance": "ER", "distinctiveness": "DI"}


def _fetch_image_bytes(cover_path: str) -> bytes | None:
    try:
        if cover_path.startswith("s3://") or cover_path.startswith("greeting-cards"):
            return get_object(cover_path)
        return Path(cover_path).read_bytes()
    except Exception as e:
        log.warning(f"Failed to load {cover_path}: {e}")
        return None


def _score_bar(val: float) -> str:
    pct = int(val * 100)
    color = "#4caf50" if val >= 0.7 else "#ff9800" if val >= 0.5 else "#f44336"
    return (
        f'<div style="background:#333;border-radius:3px;height:8px;width:100%;margin:1px 0">'
        f'<div style="background:{color};height:8px;border-radius:3px;width:{pct}%"></div></div>'
    )


def export(
    out_dir: Path,
    occasions: list[str],
    d_per_occasion: int = 5,
    ratings_path: str | None = None,
) -> None:
    out_dir = Path(out_dir)

    gen_conditions = ["A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank"]
    gen_df = pd.read_sql(_GEN_SQL, engine(), params={"conditions": gen_conditions})
    log.info(f"Loaded {len(gen_df)} generated cards (A/B/C)")

    human_df = pd.read_sql(
        _HUMAN_SQL, engine(),
        params={"occasions": occasions, "per_occasion": d_per_occasion},
    )
    log.info(f"Loaded {len(human_df)} human bestseller cards (D)")

    all_cards = pd.concat([gen_df, human_df], ignore_index=True)

    ratings_df = None
    if ratings_path:
        rp = Path(ratings_path)
        if rp.exists():
            ratings_df = pd.read_csv(rp)
            log.info(f"Loaded {len(ratings_df)} ratings from {rp}")
        else:
            log.warning(f"Ratings file not found: {rp}")

    exported = 0
    html_sections: dict[str, list[str]] = {}

    for _, row in all_cards.iterrows():
        cond = row["condition_tag"]
        occasion = row.get("occasion", "unknown") or "unknown"
        cover_path = row.get("cover_path")
        headline = row.get("headline_text", "") or ""
        card_key = row["card_key"]

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

        scores_html = ""
        if ratings_df is not None:
            match = ratings_df[ratings_df["card_key"] == card_key]
            if not match.empty:
                r = match.iloc[0]
                scores_html = '<div style="text-align:left;width:200px;margin-top:4px">'
                for dim in DIMS:
                    val = r.get(dim, float("nan"))
                    if pd.notna(val):
                        label = DIM_SHORT.get(dim, dim[:2].upper())
                        scores_html += (
                            f'<div style="display:flex;align-items:center;gap:4px;font-size:10px">'
                            f'<span style="width:20px">{label}</span>'
                            f'<div style="flex:1">{_score_bar(val)}</div>'
                            f'<span style="width:28px;text-align:right">{val:.2f}</span></div>'
                        )
                scores_html += "</div>"

        if cond not in html_sections:
            html_sections[cond] = []
        html_sections[cond].append(
            f'<div style="text-align:center;border:1px solid #333;border-radius:8px;'
            f'padding:8px;background:#1a1a1a">'
            f'<img src="{cond}/{fname}" width=200 style="border-radius:4px"><br>'
            f'<small style="color:#aaa">{occasion}<br>{headline[:40]}</small>'
            f'{scores_html}</div>'
        )

    cond_labels = {
        "A_naive_ai": "A — Naive AI (no LoRA, no pipeline)",
        "B_pipeline_no_rerank": "B — Pipeline, no rerank (N=1)",
        "C_pipeline_rerank": "C — Pipeline + rerank (N=8)",
        "D_human_bestseller": "D — Human Bestsellers",
    }

    html = (
        '<html><head><style>'
        'body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:20px}'
        'h1{border-bottom:2px solid #444;padding-bottom:10px}'
        'h2{color:#88aaff;margin-top:30px}'
        '.grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}'
        '</style></head><body>'
    )
    html += "<h1>Card Gallery with LLM Scores</h1>"
    if ratings_df is not None:
        html += "<p>PI=Purchase Intent, OF=Occasion Fit, AE=Aesthetic, ER=Emotional Resonance, DI=Distinctiveness</p>"

    for cond in ["A_naive_ai", "B_pipeline_no_rerank", "C_pipeline_rerank", "D_human_bestseller"]:
        if cond not in html_sections:
            continue
        label = cond_labels.get(cond, cond)
        n = len(html_sections[cond])
        html += f'<h2>{label} ({n} cards)</h2>'
        html += '<div class="grid">'
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
        ratings: str = "./artifacts/llm_system_eval/raw_ratings.csv",
    ) -> None:
        occ_list = [o.strip() for o in occasions.split(",")]
        export(Path(out), occ_list, d_per_occasion=d_per_occasion, ratings_path=ratings)

    typer.run(cli)
