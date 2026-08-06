"""Validate the headline contrasts by pairwise comparison instead of scoring.

Every number in the evaluation comes from Scoring Evaluation: each card is rated
independently and conditions are compared through their means.
\\citet{chen2024mllm} benchmarks multimodal judges on three tasks and finds they
show human-like discernment in Pair Comparison while diverging from human
preference in Scoring Evaluation and Batch Ranking. That is a direct objection
to the instrument, and the judge-robustness runs corroborate it from the other
side: per-card agreement between judges is 0.454 while condition-level agreement
is 0.8 to 1.0, which is what an unreliable per-item measure that aggregates well
looks like.

This does not replace the scoring pipeline. It asks whether the conclusions
drawn from it survive under the modality the literature says is reliable, by
putting cards head to head and counting wins.

Design:

  pairs        within subtype, so a comparison is never confounded by occasion
  both orders  every pair is judged twice with the sides swapped, because
               position bias is one of the failure modes that paper reports
  ties allowed forcing a choice would manufacture a winner between cards the
               judge considers equivalent, and equivalence is the hypothesis
  control      A vs B is included alongside the two equivalence contrasts.
               Scoring calls it significant at r = +0.495, so if pairwise
               cannot detect it either, the method is underpowered here and its
               nulls carry no information. Without this a null result is
               uninterpretable.

    python -m eval.pairwise_validation --gallery ~/eval_gallery_160 \
        --provider anthropic --model claude-sonnet-4-6
    python -m eval.pairwise_validation --analyse
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import typer
from PIL import Image, ImageDraw, ImageFont

from common.logging import get_logger
from common.vlm import call_vlm, image_to_b64
from eval.judge_robustness import _load_image, _slug
from eval.reports.thesis_card_figures import load_gallery

log = get_logger(__name__)

OUT_DIR = Path("artifacts/pairwise_validation")
FONT = Path("generation/layout/fonts/google/Rubik-SemiBold.ttf")

# (left condition, right condition, label). A vs B is the positive control.
CONTRASTS = [
    ("B_pipeline_no_rerank", "D_human_reference", "B vs D"),
    ("B_pipeline_no_rerank", "C_pipeline_rerank", "B vs C"),
    ("A_naive_ai", "B_pipeline_no_rerank", "A vs B (control)"),
]

SYSTEM = (
    "You are a shopper choosing a greeting card. You will see two cards side by "
    "side, numbered 1 and 2. Judge only which you would be more likely to buy."
)

TEMPLATE = """Two greeting cards for {occasion}, shown side by side and numbered 1 and 2.

If you were buying a card for this occasion, which would you buy?

Give one or two sentences of reasoning, then your answer strictly as [[1]], \
[[2]], or [[TIE]]. Use [[TIE]] only if you genuinely have no preference.
Example: Rating: [[1]]"""

_VERDICT = re.compile(r"\[\[\s*(1|2|TIE)\s*\]\]", re.I)


def _label(img: Image.Image, text: str) -> Image.Image:
    """Put a number above a card so the prompt can refer to it unambiguously."""
    band = 46
    out = Image.new("RGB", (img.width, img.height + band), "white")
    out.paste(img, (0, band))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(str(FONT), 34)
    except OSError:
        font = ImageFont.load_default()
    draw.text((img.width // 2, band // 2), text, fill="black",
              font=font, anchor="mm")
    return out


def _side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """One image holding both cards, since the transport sends a single image.

    Heights are matched so neither card is favoured by size, and the gap keeps
    the judge from reading them as one composition.
    """
    h = min(left.height, right.height, 900)
    lw = round(left.width * h / left.height)
    rw = round(right.width * h / right.height)
    left = _label(left.resize((lw, h), Image.LANCZOS), "1")
    right = _label(right.resize((rw, h), Image.LANCZOS), "2")
    gap = 28
    canvas = Image.new("RGB", (lw + rw + gap, left.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (lw + gap, 0))
    return canvas


def _build_pairs(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Match cards within subtype, deterministically, for every contrast."""
    rows = []
    for cond_a, cond_b, label in CONTRASTS:
        for occ in sorted(df.occasion.dropna().unique()):
            a = df[(df.condition == cond_a) & (df.occasion == occ)].sort_values("cover_path")
            b = df[(df.condition == cond_b) & (df.occasion == occ)].sort_values("cover_path")
            n = min(len(a), len(b))
            if n == 0:
                log.warning(f"{label} {occ}: no pairs ({len(a)} vs {len(b)})")
                continue
            if len(a) != len(b):
                log.warning(f"{label} {occ}: {len(a)} vs {len(b)}, pairing {n}")
            for i in range(n):
                ra, rb = a.iloc[i], b.iloc[i]
                rows.append({
                    "contrast": label, "occasion": occ, "pair_index": i,
                    "cond_a": cond_a, "cond_b": cond_b,
                    "path_a": ra.cover_path, "path_b": rb.cover_path,
                    "pi_a": ra.purchase_intent, "pi_b": rb.purchase_intent,
                })
    pairs = pd.DataFrame(rows)
    # Each pair judged in both orders. `a_on_left` records which way round it
    # was shown, so position bias can be measured rather than assumed away.
    both = pd.concat([pairs.assign(a_on_left=True), pairs.assign(a_on_left=False)],
                     ignore_index=True)
    return both.sort_values(["contrast", "occasion", "pair_index", "a_on_left"]).reset_index(drop=True)


def _judge_one(row, provider: str, model: str | None) -> str | None:
    left_path, right_path = (row.path_a, row.path_b) if row.a_on_left else (row.path_b, row.path_a)
    try:
        img = _side_by_side(_load_image(left_path), _load_image(right_path))
    except Exception as e:
        log.warning(f"could not build pair image: {e}")
        return None
    reply = call_vlm(
        image_to_b64(img), SYSTEM,
        TEMPLATE.format(occasion=row.occasion),
        provider=provider, model=model, temperature=0.0, max_tokens=400,
    )
    m = _VERDICT.search(reply or "")
    if not m:
        log.warning(f"no verdict parsed from: {(reply or '')[:120]!r}")
        return None
    return m.group(1).upper()


def collect(gallery: Path, provider: str, model: str | None, workers: int, seed: int) -> pd.DataFrame:
    df = load_gallery(gallery)
    pairs = _build_pairs(df, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"verdicts_{_slug(provider, model)}.csv"

    done: dict[tuple, dict] = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = {(r.contrast, r.occasion, r.pair_index, bool(r.a_on_left)): r._asdict()
                for r in prev.itertuples()}
        log.info(f"resuming: {len(done)} verdicts already in {out_path}")

    todo = [r for r in pairs.itertuples()
            if (r.contrast, r.occasion, r.pair_index, bool(r.a_on_left)) not in done]
    log.info(f"{len(pairs)} comparisons, {len(todo)} to judge "
             f"({provider} {model or 'default'})")

    results = [dict(v) for v in done.values()]

    def one(row):
        verdict = _judge_one(row, provider, model)
        if verdict is None:
            return None
        # Normalise to "which CONDITION won", independent of side shown.
        if verdict == "TIE":
            winner = "tie"
        else:
            shown_left_is_a = row.a_on_left
            picked_left = verdict == "1"
            winner = "a" if picked_left == shown_left_is_a else "b"
        return {
            "contrast": row.contrast, "occasion": row.occasion,
            "pair_index": row.pair_index, "a_on_left": row.a_on_left,
            "cond_a": row.cond_a, "cond_b": row.cond_b,
            "verdict_side": verdict, "winner": winner,
            "pi_a": row.pi_a, "pi_b": row.pi_b,
        }

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, rec in enumerate(pool.map(one, todo), start=1):
                if rec:
                    results.append(rec)
                if i % 20 == 0:
                    log.info(f"  {i}/{len(todo)}")
                    pd.DataFrame(results).to_csv(out_path, index=False)

    out = pd.DataFrame(results)
    out.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(out)} verdicts)")
    return out


def analyse_one(df: pd.DataFrame, judge: str) -> list[dict]:
    from scipy.stats import binomtest

    rows = []
    for _, _, label in CONTRASTS:
        sub = df[df.contrast == label]
        if sub.empty:
            continue
        a_wins = int((sub.winner == "a").sum())
        b_wins = int((sub.winner == "b").sum())
        ties = int((sub.winner == "tie").sum())
        decided = a_wins + b_wins

        # Position bias: how often the card shown on the left was chosen,
        # regardless of which condition it belonged to.
        left_picked = int((sub.verdict_side == "1").sum())
        sided = int((sub.verdict_side != "TIE").sum())

        # Order consistency: a pair judged both ways should give the same
        # winner. Flips are the judge disagreeing with itself.
        consistent = flipped = 0
        for _, g in sub.groupby(["occasion", "pair_index"]):
            if len(g) != 2:
                continue
            w = set(g.winner)
            if len(w) == 1:
                consistent += 1
            else:
                flipped += 1

        p = binomtest(a_wins, decided, 0.5).pvalue if decided else float("nan")
        rows.append({
            "judge": judge, "contrast": label,
            "n": len(sub), "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "a_win_rate": round(a_wins / decided, 3) if decided else None,
            "p_binom": round(float(p), 4) if decided else None,
            "left_pick_rate": round(left_picked / sided, 3) if sided else None,
            "order_consistent": consistent, "order_flipped": flipped,
            "consistency": round(consistent / (consistent + flipped), 3)
            if (consistent + flipped) else None,
        })
    return rows


def run(
    gallery: Path = typer.Option(None, help="exported eval gallery directory"),
    provider: str = typer.Option("anthropic"),
    model: str = typer.Option(None),
    workers: int = typer.Option(4),
    seed: int = typer.Option(42),
    analyse: bool = typer.Option(False, help="summarise every verdicts_*.csv"),
) -> None:
    if analyse:
        rows = []
        for path in sorted(OUT_DIR.glob("verdicts_*.csv")):
            rows += analyse_one(pd.read_csv(path), path.stem.replace("verdicts_", ""))
        if not rows:
            raise SystemExit(f"no verdicts_*.csv in {OUT_DIR}")
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
        (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
        return

    if gallery is None:
        raise SystemExit("--gallery is required unless --analyse is passed")
    df = collect(gallery, provider, model, workers, seed)
    print(pd.DataFrame(analyse_one(df, _slug(provider, model))).to_string(index=False))


if __name__ == "__main__":
    typer.run(run)
