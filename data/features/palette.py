"""K-means LAB palette extraction.

For each card image:
- Convert to LAB
- K-means into k=5 clusters
- Output (L*, a*, b*, weight) per cluster, sorted by weight desc

Feeds the layout composer (contrast-aware text colour selection) and the
predictor (palette-derived aesthetic features).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans

from common.db import connection
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)


def extract_palette(
    img: Image.Image | Path | bytes,
    *,
    k: int = 5,
    sample_pixels: int = 20_000,
    seed: int = 42,
) -> list[dict]:
    """Return top-k LAB clusters: [{L, a, b, weight}, ...]."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))
    img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    lab = color.rgb2lab(arr).reshape(-1, 3)

    if lab.shape[0] > sample_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(lab.shape[0], size=sample_pixels, replace=False)
        lab = lab[idx]

    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    labels = km.fit_predict(lab)
    centres = km.cluster_centers_
    counts = np.bincount(labels, minlength=k)
    weights = counts / counts.sum()

    order = np.argsort(-weights)
    return [
        {
            "L": float(centres[i, 0]),
            "a": float(centres[i, 1]),
            "b": float(centres[i, 2]),
            "weight": float(weights[i]),
        }
        for i in order
    ]


_SELECT_MISSING = """
SELECT l.listing_id, li.storage_path
FROM listings l
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.palette_lab IS NULL
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT = """
INSERT INTO listing_features (listing_id, palette_lab, feature_version)
VALUES (%(listing_id)s, %(palette)s, %(version)s)
ON CONFLICT (listing_id) DO UPDATE
SET palette_lab = EXCLUDED.palette_lab,
    feature_version = EXCLUDED.feature_version,
    computed_at = NOW();
"""


def run_palette_missing(limit: int = 100_000, feature_version: str = "palette-v1") -> int:
    import json

    processed = 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_MISSING, {"limit": limit})
        for row in cur.fetchall():
            try:
                data = get_object(row["storage_path"])
                palette = extract_palette(data)
                cur.execute(
                    _UPSERT,
                    {
                        "listing_id": row["listing_id"],
                        "palette": json.dumps(palette),
                        "version": feature_version,
                    },
                )
                processed += 1
            except Exception as e:
                log.warning(f"Palette failed for {row['listing_id']}: {e}")
    return processed


if __name__ == "__main__":
    import typer

    typer.run(run_palette_missing)
