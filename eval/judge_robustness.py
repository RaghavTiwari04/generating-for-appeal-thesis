"""Re-score the evaluated cards with other judge families and re-test equivalence.

The central threat to this evaluation is circularity: one instrument produced the
predictor's training labels and the reported scores, and cross-family judge
agreement sits around rho = 0.6, which is enough disagreement to move a result on
its own. That makes the headline equivalence a claim about one judge until it is
checked against another.

This re-scores the same 159 cards with a different judge family and re-runs the
pairwise and equivalence tests on the new scores. The comparison of interest is
not whether the absolute means move, which they will, but whether the verdicts
survive: B equivalent to D, B equivalent to C, A below all three.

What is and is not swapped. The VLM that answers as each persona and applies the
rubric changes. The SSR embedding model does not: it is the measuring scale
rather than the respondent, held fixed on purpose so the two judges are compared
on one ruler. Purchase intent therefore swaps the respondent and keeps the scale,
which is the comparison worth making.

Reads cards from an exported gallery, so it needs no database.

    python -m eval.judge_robustness --gallery ~/Desktop/eval_gallery --provider anthropic
    python -m eval.judge_robustness --gallery ~/Desktop/eval_gallery --provider openai
    python -m eval.judge_robustness --compare        # after both have run
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import typer
from PIL import Image

from common.logging import get_logger
from eval.llm_system_eval import (
    JUDGE_LONG_EDGE,
    _hodges_lehmann,
    _tost_equivalence,
    pairwise_holm,
)
from eval.reports.thesis_card_figures import load_gallery

log = get_logger(__name__)

OUT_DIR = Path("artifacts/judge_robustness")
DIMS = ["purchase_intent", "occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness"]
CONTRASTS = [
    ("B_pipeline_no_rerank", "D_human_reference", "B vs D"),
    ("B_pipeline_no_rerank", "C_pipeline_rerank", "B vs C"),
    ("C_pipeline_rerank", "D_human_reference", "C vs D"),
]


def _slug(provider: str, model: str | None) -> str:
    """Stable filename fragment identifying one judge."""
    return re.sub(r"[^a-z0-9]+", "_", f"{provider}_{model}".lower()).strip("_") if model else provider


def _load_image(path: str) -> Image.Image:
    """Same normalisation the reported run used, so resolution is not a variable."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > JUDGE_LONG_EDGE:
        scale = JUDGE_LONG_EDGE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def rescore(gallery: Path, provider: str, model: str | None, workers: int) -> pd.DataFrame:
    from scoring import CardScorer

    cards = load_gallery(gallery)
    # Keyed on the model, not the provider. Two OpenAI judges are two judges:
    # keying on "openai" alone meant a second model resumed from the first
    # model's completed rows, scored nothing, and wrote out the first model's
    # ratings under a name implying they were the second's.
    out_path = OUT_DIR / f"ratings_{_slug(provider, model)}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume rather than restart: a run is thousands of paid calls and a
    # transient failure partway through should not repeat the ones that worked.
    done: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = {r["cover_path"]: r.to_dict() for _, r in prev.iterrows()}
        log.info(f"resuming: {len(done)} cards already scored in {out_path}")

    scorer = CardScorer(provider=provider, model=model)
    todo = [r for r in cards.itertuples() if r.cover_path not in done]
    log.info(f"scoring {len(todo)} cards with provider={provider} model={model or 'default'}")

    def one(row) -> dict | None:
        try:
            scores = scorer.score(_load_image(row.cover_path), occasion=row.occasion)
        except Exception as e:
            log.warning(f"score failed for {Path(row.cover_path).name}: {e}")
            return None
        rec = {
            "cover_path": row.cover_path,
            "condition": row.condition,
            "occasion": row.occasion,
        }
        for d in DIMS:
            rec[d] = scores.get(d)
        return rec

    results = list(done.values())
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, rec in enumerate(pool.map(one, todo), start=1):
                if rec:
                    results.append(rec)
                if i % 20 == 0:
                    log.info(f"  {i}/{len(todo)}")
                    pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} cards)")
    return df


def analyse(df: pd.DataFrame, label: str) -> dict:
    df = df.dropna(subset=["purchase_intent"])
    means = df.groupby("condition").purchase_intent.mean().round(4).to_dict()
    holm, _ = pairwise_holm(df)

    verdicts = {}
    for a, b, name in CONTRASTS:
        sa = df[df.condition == a].purchase_intent
        sb = df[df.condition == b].purchase_intent
        if len(sa) < 3 or len(sb) < 3:
            continue
        t = _tost_equivalence(sa, sb, delta=0.02)
        verdicts[name] = {
            "hodges_lehmann": round(_hodges_lehmann(sa, sb), 4),
            "p_tost": round(t["p_tost"], 4),
            "equivalent": t["equivalent"],
        }

    a_below = {
        k: round(v, 4) for k, v in holm.items() if k.startswith("A_naive_ai")
    }
    return {
        "judge": label,
        "n": len(df),
        "means": means,
        "equivalence": verdicts,
        "A_contrasts_holm_p": a_below,
    }


def run(
    gallery: Path = typer.Option(None, help="exported eval gallery directory"),
    provider: str = typer.Option("anthropic", help="anthropic | openai | glm | gemini"),
    model: str = typer.Option(None, help="override the provider's default model"),
    workers: int = typer.Option(4, help="concurrent cards; each is 10 VLM calls"),
    compare: bool = typer.Option(False, help="summarise every ratings_*.csv already written"),
) -> None:
    if compare:
        rows = []
        for path in sorted(OUT_DIR.glob("ratings_*.csv")):
            rows.append(analyse(pd.read_csv(path), path.stem.replace("ratings_", "")))
        if not rows:
            raise SystemExit(f"no ratings_*.csv in {OUT_DIR}")
        (OUT_DIR / "comparison.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2))
        return

    if gallery is None:
        raise SystemExit("--gallery is required unless --compare is passed")
    df = rescore(gallery, provider, model, workers)
    label = _slug(provider, model)
    summary = analyse(df, label)
    (OUT_DIR / f"summary_{label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(run)
