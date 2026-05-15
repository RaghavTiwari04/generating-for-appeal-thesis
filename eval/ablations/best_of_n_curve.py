"""Best-of-N saturation curve ablation.

For each N in {1, 2, 4, 8, 16}, sample best-of-N from the existing condition-C
candidate pool and report mean predicted saleability (and mean survey
purchase-intent where available). Produces a saturation curve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.db import engine
from common.logging import get_logger

log = get_logger(__name__)

DEFAULT_N_GRID = (1, 2, 4, 8, 16)


@dataclass
class Point:
    n: int
    mean_score_top1: float
    mean_score_topk: float


_QUERY = """
SELECT (brief->>'group_id') AS group_id,
       (predicted_scores->>'saleability_calibrated')::float AS sale
FROM generated_cards
WHERE condition_tag = %(condition)s
  AND predicted_scores ? 'saleability_calibrated';
"""


def run(condition: str = "C_pipeline_rerank", out_dir: str = "./artifacts/ablations") -> list[Point]:
    df = pd.read_sql(_QUERY, engine(), params={"condition": condition})
    if df.empty:
        raise SystemExit("No scored candidates found")
    rng = np.random.default_rng(0)

    out: list[Point] = []
    for n in DEFAULT_N_GRID:
        per_group: list[float] = []
        for _, grp in df.groupby("group_id"):
            if len(grp) < n:
                continue
            sample = rng.choice(grp["sale"].to_numpy(), size=n, replace=False)
            per_group.append(float(np.max(sample)))
        if not per_group:
            continue
        out.append(
            Point(
                n=n,
                mean_score_top1=float(np.mean(per_group)),
                mean_score_topk=float(np.mean(per_group)),
            )
        )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    (out_dir_p / "best_of_n_curve.json").write_text(
        json.dumps([{"n": p.n, "score": p.mean_score_top1} for p in out], indent=2)
    )
    return out


if __name__ == "__main__":
    import typer

    typer.run(run)
