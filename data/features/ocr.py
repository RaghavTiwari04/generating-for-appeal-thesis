"""Extract headline text from card cover image.

Tesseract via pytesseract is the default — fast and good enough for the
mostly-legible headline area. PaddleOCR can be swapped in for harder cases
(handwritten / stylised typography) at the cost of much heavier deps.

The OCR output feeds:
- the predictor as `extracted_text`
- the layout composer (to detect existing text bounding boxes when reprocessing
  a generated card for verification)
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pytesseract
from PIL import Image

from common.db import connection
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)


@dataclass
class OCRResult:
    text: str
    mean_confidence: float


def ocr_image(img: Image.Image | Path | bytes, *, lang: str = "eng") -> OCRResult:
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))
    img = img.convert("RGB")

    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confs: list[float] = []
    for text, conf in zip(data["text"], data["conf"], strict=False):
        text = (text or "").strip()
        if not text:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        words.append(text)
        confs.append(c)

    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return OCRResult(text=" ".join(words), mean_confidence=mean_conf)


# ---------------------------------------------------------------------------
# Bulk job
# ---------------------------------------------------------------------------
_SELECT_MISSING = """
SELECT l.listing_id, li.storage_path
FROM listings l
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.extracted_text IS NULL
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT = """
INSERT INTO listing_features (listing_id, extracted_text, feature_version)
VALUES (%(listing_id)s, %(text)s, %(version)s)
ON CONFLICT (listing_id) DO UPDATE
SET extracted_text = EXCLUDED.extracted_text,
    feature_version = EXCLUDED.feature_version,
    computed_at = NOW();
"""


def run_ocr_missing(limit: int = 1000, feature_version: str = "ocr-v1") -> int:
    processed = 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_MISSING, {"limit": limit})
        for row in cur.fetchall():
            try:
                data = get_object(row["storage_path"])
                result = ocr_image(data)
                cur.execute(
                    _UPSERT,
                    {
                        "listing_id": row["listing_id"],
                        "text": result.text,
                        "version": feature_version,
                    },
                )
                processed += 1
            except Exception as e:
                log.warning(f"OCR failed for {row['listing_id']}: {e}")
    return processed


if __name__ == "__main__":
    import typer

    typer.run(run_ocr_missing)
