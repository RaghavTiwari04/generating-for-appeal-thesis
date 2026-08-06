"""Measure the within-batch spread of predicted scores against the scraped set.

This settles the mechanism behind the reranking null. Best-of-$N$ gain is bounded
by how far apart the candidates are, not by how well the scorer ranks them, so
the claim that the pipeline's candidates are too alike for selection to help is
a claim about a distribution that has never been measured. Only the returned
card is persisted by a normal run, so the eight scores it chose between are
discarded.

This regenerates candidates for a sample of requests, keeps every predicted
score, and plots the within-batch spread against the spread of the held-out
scraped cards under the same predictor. Two outcomes are informative:

  narrow within-batch, wide scraped   the pool is the limit, as argued
  comparable spreads                  the predictor is the limit, and the
                                      argument in the writeup is wrong

The scraped comparison uses the test split rather than the whole corpus, because
that is the set best-of-N recovery was measured on.

Runs on the cluster: needs the predictor, the LoRA, a GPU and the database.

    python -m eval.reports.candidate_spread --n 8 --batches 12
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS

log = get_logger(__name__)

OUT_DEFAULT = Path("report/figures")
ARTEFACT = Path("artifacts/candidate_spread.json")

PI_KEYS = ("purchase_intent_calibrated", "purchase_intent")


def _pi(scores: dict | None) -> float | None:
    for k in PI_KEYS:
        if scores and scores.get(k) is not None:
            return float(scores[k])
    return None


def _scraped_scores(cfg) -> np.ndarray:
    """Predicted purchase intent over the held-out scraped cards."""
    from data.features.clip_embed import CLIPEmbedder
    from models.predictor.dataset import SplitConfig, load_training_frame, split_by_seller
    from models.predictor.infer import CardFeatures
    from pipeline.orchestrator import _load_predictor

    frame = load_training_frame()
    test = split_by_seller(frame, SplitConfig())["test"]
    log.info(f"scraped test split: {len(test)} cards")

    embedder = CLIPEmbedder()
    text_embs = embedder.embed_texts([str(t or "") for t in test["extracted_text"]])
    feats = [
        CardFeatures(
            image_emb=np.asarray(row.clip_embedding, dtype=np.float32),
            text_emb=text_embs[i],
            occasion=row.occasion,
        )
        for i, row in enumerate(test.itertuples())
    ]
    predictor = _load_predictor(cfg, None)
    scored = [_pi(s) for s in predictor.score(feats)]
    return np.asarray([v for v in scored if v is not None], dtype=float)


def _candidate_batches(n: int, batches: int, seed: int, cfg) -> list[np.ndarray]:
    """Predicted purchase intent for all n candidates of each generated batch."""
    from pipeline.orchestrator import generate

    out = []
    for i in range(batches):
        occasion = ACTIVE_OCCASIONS[i % len(ACTIVE_OCCASIONS)]
        request = {"occasion": occasion, "seed": seed + i}
        # top_k unset, so `generate` returns every scored candidate rather than
        # only the winner. That is the entire point of this measurement.
        cands = generate(request, cfg)
        vals = [v for v in (_pi(c.scores) for c in cands) if v is not None]
        if len(vals) < 2:
            log.warning(f"batch {i} ({occasion}) returned {len(vals)} scored candidates")
            continue
        arr = np.asarray(vals, dtype=float)
        out.append(arr)
        log.info(
            f"batch {i} ({occasion}): n={len(arr)} "
            f"range={arr.max() - arr.min():.4f} sd={arr.std():.4f}"
        )
    return out


def run(
    n: int = typer.Option(8, help="candidates per batch, matching condition C"),
    batches: int = typer.Option(12, help="how many batches to generate"),
    seed: int = typer.Option(9000),
    out: Path = typer.Option(OUT_DEFAULT),
) -> None:
    from pipeline.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig()
    cfg.n_candidates = n
    cfg.top_k = n  # keep them all
    # `generate` persists everything it returns, so with top_k = n this writes
    # n cards per batch into generated_cards. Tag them distinctly: under the
    # default they would land as C_pipeline_rerank and sit alongside the
    # evaluated set, and anything that queried by condition_tag without also
    # filtering on eval_run would silently pick up probe cards.
    cfg.condition_tag = "probe_candidate_spread"

    scraped = _scraped_scores(cfg)
    log.info(f"scraped: n={len(scraped)} sd={scraped.std():.4f}")

    batch_scores = _candidate_batches(n, batches, seed, cfg)
    if not batch_scores:
        raise SystemExit("no batch produced scored candidates")

    within_sd = np.array([b.std() for b in batch_scores])
    within_range = np.array([b.max() - b.min() for b in batch_scores])

    summary = {
        "n_per_batch": n,
        "batches": len(batch_scores),
        "within_batch_sd_mean": float(within_sd.mean()),
        "within_batch_range_mean": float(within_range.mean()),
        "scraped_sd": float(scraped.std()),
        "scraped_range": float(scraped.max() - scraped.min()),
        # The quantity the argument turns on: how much of the corpus-wide spread
        # a single batch actually offers the selector.
        "spread_ratio": float(within_sd.mean() / scraped.std()),
    }
    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTEFACT.write_text(json.dumps(summary, indent=2))
    log.info(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(scraped, bins=40, density=True, alpha=0.55, color="#9ca3af",
            label=f"Scraped test set (sd {scraped.std():.3f})")
    # Each batch centred on its own mean, so the plot compares spreads rather
    # than locations: a batch sitting high is not the point, its width is.
    centred = np.concatenate([b - b.mean() + scraped.mean() for b in batch_scores])
    ax.hist(centred, bins=25, density=True, alpha=0.75, color="#34d399",
            label=f"Pipeline candidates, mean-centred (sd {within_sd.mean():.3f})")
    ax.set_xlabel("Predicted purchase intent")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "candidate_spread.pdf")
    plt.close(fig)
    log.info(f"wrote {out / 'candidate_spread.pdf'}")

    for k, v in summary.items():
        print(f"{k:26s} {v}")


if __name__ == "__main__":
    typer.run(run)
