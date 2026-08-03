"""Find an OCR configuration that can read card lettering.

`verify_headline` scored 0.00 on covers whose headline is plainly legible —
brush-script "Happy Birthday" across the top of the card. Tesseract's default
page segmentation is built for documents, and on an illustration it can discard
the text region before recognition runs at all.

Rather than guess a fix, this runs the candidate configurations over covers that
actually failed and reports what each reads back.

    python -m scripts.tune_headline_ocr --dir artifacts/rejected_covers

Filenames from `_save_rejected` carry the headline, so the expected text comes
from the filename: happy-birthday_0.00_3d59549f.png -> "happy birthday".
"""

from __future__ import annotations

from pathlib import Path

import pytesseract
import typer
from PIL import Image, ImageEnhance, ImageOps

from common.logging import get_logger
from generation.image.headline_text import match_score

log = get_logger(__name__)

# 3  document-style layout analysis, the current default
# 6  assume one uniform block of text
# 11 sparse text, find as much as possible in no particular order
# 12 sparse text with orientation detection
PSMS = (3, 6, 11, 12)


def _variants(img: Image.Image) -> dict[str, Image.Image]:
    """Preprocessing worth trying, cheapest first."""
    grey = ImageOps.grayscale(img)
    return {
        "raw": img,
        "grey": grey,
        # Card lettering is often a mid-tone colour on a cream ground, which is
        # low contrast once greyscaled.
        "grey+contrast": ImageEnhance.Contrast(grey).enhance(2.5),
        "grey+invert": ImageOps.invert(grey),
        # Tesseract's models expect roughly 300dpi text; card lettering rendered
        # at 896px wide is smaller than that.
        "grey+2x": grey.resize((grey.width * 2, grey.height * 2), Image.LANCZOS),
    }


def run(dir: str = "artifacts/rejected_covers", limit: int = 8) -> None:
    covers = sorted(Path(dir).glob("*.png"))[:limit]
    if not covers:
        raise SystemExit(f"No covers in {dir}")

    log.info(f"Trying {len(PSMS)} page-segmentation modes on {len(covers)} covers")
    scores: dict[str, list[float]] = {}

    for path in covers:
        headline = path.name.split("_")[0].replace("-", " ")
        img = Image.open(path).convert("RGB")
        log.info(f"--- {path.name} (expecting {headline!r}) ---")
        for vname, variant in _variants(img).items():
            for psm in PSMS:
                try:
                    text = pytesseract.image_to_string(variant, config=f"--psm {psm}")
                except Exception as e:
                    log.warning(f"  {vname} psm={psm}: {e}")
                    continue
                score = match_score(text, headline)
                key = f"{vname} psm={psm}"
                scores.setdefault(key, []).append(score)
                if score > 0:
                    flat = " ".join(text.split())[:60]
                    log.info(f"  {key:24s} {score:.2f}  {flat!r}")

    log.info("--- mean match over all covers, best first ---")
    ranked = sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    for key, vals in ranked[:10]:
        mean = sum(vals) / len(vals)
        log.info(f"  {key:24s} {mean:.2f}  (passes 0.8 on {sum(v >= 0.8 for v in vals)}/{len(vals)})")


if __name__ == "__main__":
    typer.run(run)
