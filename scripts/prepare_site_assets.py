"""Turn an exported card gallery into the images the demo site ships.

Run once by hand after `eval.export_gallery`, and commit what it writes. This
is content production, not a build step: the site has no build, and a page
that regenerated its own illustrations at deploy time would be a worse idea
than it sounds.

Choosing which cards appear is a human job and stays one. This script does
the mechanical part around that choice: crop, resize, convert, and record
where every image came from.

Two modes.

    python -m scripts.prepare_site_assets --list --gallery ./eval_gallery

prints what is available, with scores, so the cards can be picked by eye.

    python -m scripts.prepare_site_assets --gallery ./eval_gallery

reads `site/assets/cards/curation.json`, processes what it names, and writes
`manifest.json` beside the images plus the quiz rounds.

On cropping. Roughly 70 per cent of the human reference images carry
marketplace mockup framing and some carry seller watermarks. In the guessing
game that framing is the answer, so those cards are cropped to the card face
or left out. The crop is given per card in the curation file rather than
guessed here, because an automatic crop that clips artwork would be worse
than no crop at all.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil

import typer

CONDITIONS = {
    "A_naive_ai": "naive",
    "B_pipeline_no_rerank": "pipeline",
    "C_pipeline_rerank": "pipeline_reranked",
    "D_human_reference": "human",
    # The exporter used this name before the rename; accepted so an older
    # gallery still works rather than silently exporting three conditions.
    "D_human_bestseller": "human",
}

# The widths the page actually asks for, via srcset. 1240 rather than a round
# 1280 because that is the full width of a generated card and there is nothing
# above it to resample from. Anything wider than the source is skipped, so the
# scraped listings, which are narrower, come out at 640 alone.
WIDTHS = (640, 1240)
JPEG_QUALITY = 85

SITE = pathlib.Path("site")
OUT = SITE / "assets" / "cards"
CURATION = OUT / "curation.json"

_BLOCK = re.compile(
    r'<img src="([^"]+)"[^>]*>(.*?)(?=<div style="text-align:center|</div></body>|<h2)',
    re.S,
)
_TAG = re.compile(r"<[^>]+>")

app = typer.Typer(add_completion=False)


def read_gallery(gallery: pathlib.Path) -> list[dict]:
    """Every card the export wrote, with what the gallery page records about it.

    The exported filenames carry the occasion and a hash of the image, not the
    card id, so the gallery page is the only place the scores and the file are
    joined. Parsing it beats re-querying a database that may not be reachable
    from wherever this is being run.
    """
    page = gallery / "gallery.html"
    if not page.exists():
        raise typer.BadParameter(f"No gallery.html in {gallery}")

    cards = []
    for src, body in _BLOCK.findall(page.read_text(encoding="utf-8", errors="replace")):
        parts = [t.strip() for t in _TAG.sub("\x00", body).split("\x00") if t.strip()]
        nums = re.findall(r"\b(0\.\d\d|1\.00)\b", " ".join(parts))
        cond_dir = src.split("/")[0]
        cards.append({
            "src": src,
            "condition_dir": cond_dir,
            "condition": CONDITIONS.get(cond_dir, cond_dir),
            "occasion": parts[0] if parts else "",
            "headline": parts[1] if len(parts) > 1 else "",
            # Order follows DIMS in the exporter: PI, OF, AE, ER, DI.
            "purchase_intent": float(nums[0]) if nums else None,
        })
    return cards


def process(src_path: pathlib.Path, dest_stem: pathlib.Path, crop) -> dict:
    """Crop, resize and write one card at every width the page asks for."""
    from PIL import Image

    im = Image.open(src_path).convert("RGB")

    if crop:
        left, top, right, bottom = crop
        w, h = im.size
        im = im.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))

    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for width in WIDTHS:
        if width > im.width:
            continue
        scaled = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        for suffix, kwargs in (
            (".jpg", {"quality": JPEG_QUALITY, "optimize": True}),
            (".webp", {"quality": JPEG_QUALITY, "method": 6}),
        ):
            out = dest_stem.with_name(f"{dest_stem.name}-{width}{suffix}")
            scaled.save(out, **kwargs)
            written.append(out.name)

    return {"width": im.width, "height": im.height, "files": written}


@app.command()
def main(
    gallery: pathlib.Path = typer.Option(..., help="An eval_gallery directory from eval.export_gallery."),
    list_only: bool = typer.Option(False, "--list", help="Print what is available and stop."),
    limit: int = typer.Option(12, help="Rows per condition when listing."),
) -> None:
    cards = read_gallery(gallery)
    if not cards:
        raise typer.BadParameter("Parsed no cards out of gallery.html")

    by_condition: dict[str, list[dict]] = {}
    for c in cards:
        by_condition.setdefault(c["condition_dir"], []).append(c)

    if list_only:
        for cond, rows in sorted(by_condition.items()):
            print(f"\n=== {cond} ({len(rows)} cards) ===")
            rows.sort(key=lambda r: -(r["purchase_intent"] or 0))
            for r in rows[:limit]:
                pi = f"{r['purchase_intent']:.2f}" if r["purchase_intent"] is not None else "  . "
                print(f"  PI={pi}  {r['occasion']:24s} {r['headline'][:34]:36s} {r['src']}")
        print("\nPut the ones you want into", CURATION)
        return

    if not CURATION.exists():
        raise typer.BadParameter(
            f"{CURATION} does not exist. Run with --list first, then write it."
        )

    plan = json.loads(CURATION.read_text(encoding="utf-8"))
    index = {c["src"]: c for c in cards}
    manifest: list[dict] = []
    quiz_rounds: list[dict] = []

    for section, entries in plan.items():
        if section.startswith("_"):
            continue
        if isinstance(entries, dict):
            entries = [entries]

        for i, entry in enumerate(entries):
            src = entry["src"]
            if src not in index:
                raise typer.BadParameter(f"{src} is not in this gallery")
            meta = index[src]

            stem = OUT / section / f"{section}-{i + 1:02d}"
            info = process(gallery / src, stem, entry.get("crop"))

            row = {
                "section": section,
                "stem": f"assets/cards/{section}/{stem.name}",
                "condition": meta["condition"],
                "occasion": meta["occasion"],
                "headline": meta["headline"],
                "purchase_intent": meta["purchase_intent"],
                "source_file": src,
                "cropped": bool(entry.get("crop")),
                # Set by hand, and only after opening the file. Says the image
                # was checked and carries no mockup frame or watermark, so it
                # needs no crop. Without it an uncropped listing is assumed
                # framed, which is the safe way round.
                "framing_checked": bool(entry.get("framing_checked")),
                **info,
            }
            manifest.append(row)

            if section == "quiz":
                quiz_rounds.append({
                    "file": f"{row['stem']}-640.jpg",
                    "condition": "human" if meta["condition"] == "human" else "generated",
                    "note": entry.get("note", ""),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "_note": (
                    "Written by scripts/prepare_site_assets.py from an exported "
                    "gallery. Every row records which card the image is and what "
                    "it scored, so any card on the site can be traced back."
                ),
                "cards": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if quiz_rounds:
        quiz_path = SITE / "data" / "quiz.json"
        doc = json.loads(quiz_path.read_text(encoding="utf-8"))
        doc["rounds"] = quiz_rounds
        quiz_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"wrote {len(quiz_rounds)} quiz rounds")

    unchecked = [
        r for r in manifest
        if r["section"] == "quiz" and r["condition"] == "human"
        and not r["cropped"] and not r["framing_checked"]
    ]
    if unchecked:
        print(
            f"\nWARNING: {len(unchecked)} marketplace cards are in the quiz "
            "with no crop and no framing check. Mockup framing gives the answer "
            "away. Open each one, then either crop it, set framing_checked, or "
            "drop it."
        )
        for r in unchecked:
            print(f"  {r['source_file']}")

    print(f"processed {len(manifest)} cards into {OUT}")


if __name__ == "__main__":
    app()
