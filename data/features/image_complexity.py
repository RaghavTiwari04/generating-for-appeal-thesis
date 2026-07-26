"""Image complexity metric for pricing features.

Complexity = edge density + frequency entropy.
- Edge density: fraction of pixels that are strong edges (Canny)
- Frequency entropy: Shannon entropy of the log-magnitude DFT spectrum

Both signals combined into a single [0, 1] score. Higher = more detailed image
(illustrated scene vs. plain typography) which correlates with higher price.

Stored in listing_features.image_complexity.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from common.db import connection
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)


def compute_complexity(img: Image.Image | Path | bytes, *, size: int = 256) -> float:
    """Return a [0, 1] image complexity score."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))

    img = img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)

    edge_density = _canny_density(arr)
    freq_entropy = _frequency_entropy(arr)

    # Normalise both to [0,1] empirically: edge_density rarely exceeds 0.3,
    # freq_entropy rarely exceeds 5.5 nats for typical card images.
    norm_edge = min(1.0, edge_density / 0.25)
    norm_freq = min(1.0, freq_entropy / 5.0)

    return float(0.5 * norm_edge + 0.5 * norm_freq)


def _canny_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are strong edges (pure numpy Canny-like)."""
    try:
        import cv2
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        return float(edges.sum()) / (255.0 * gray.size)
    except ImportError:
        # Fallback: Sobel-based
        from scipy.ndimage import sobel

        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        mag = np.hypot(sx, sy)
        thresh = mag.max() * 0.2
        return float((mag > thresh).sum()) / gray.size


def _frequency_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of log-magnitude DFT spectrum."""
    fft = np.abs(np.fft.fft2(gray)) + 1e-9
    log_mag = np.log(fft)
    log_mag -= log_mag.min()
    total = log_mag.sum()
    if total == 0:
        return 0.0
    p = log_mag / total
    entropy = -float((p * np.log(p + 1e-12)).sum())
    return entropy


# ---------------------------------------------------------------------------
# Bulk job
# ---------------------------------------------------------------------------
_SELECT_MISSING = """
SELECT l.listing_id, li.storage_path
FROM listings l
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.image_complexity IS NULL
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT = """
INSERT INTO listing_features (listing_id, image_complexity, feature_version)
VALUES (%(listing_id)s, %(complexity)s, %(version)s)
ON CONFLICT (listing_id) DO UPDATE
SET image_complexity = EXCLUDED.image_complexity,
    feature_version  = EXCLUDED.feature_version,
    computed_at      = NOW();
"""


def run_complexity_missing(limit: int = 100_000, feature_version: str = "complexity-v1") -> int:
    processed = 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_MISSING, {"limit": limit})
        for row in cur.fetchall():
            try:
                data = get_object(row["storage_path"])
                score = compute_complexity(data)
                cur.execute(
                    _UPSERT,
                    {
                        "listing_id": row["listing_id"],
                        "complexity": score,
                        "version": feature_version,
                    },
                )
                processed += 1
            except Exception as e:
                log.warning(f"Complexity failed for {row['listing_id']}: {e}")
    return processed


if __name__ == "__main__":
    import typer

    typer.run(run_complexity_missing)
