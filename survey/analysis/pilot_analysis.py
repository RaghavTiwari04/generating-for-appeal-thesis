"""Pilot survey analysis: ICC, distributions, instrument refinement notes.

Run after the Prolific pilot (n=50) is complete. Outputs:
- ICC(3,1) and ICC(3,k) per dimension
- Mean + SD per dimension
- Ceiling/floor effect flags
- Response time distribution
- Free-text sample for qualitative review
- Suggested instrument revisions

Usage:
    python -m survey.analysis.pilot_analysis --study-id pilot_v1
    python -m survey.analysis.pilot_analysis --study-id pilot_v1 --out-dir artifacts/pilot
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer

from common.db import engine
from common.logging import get_logger
from survey.analysis.icc import SURVEY_DIMENSIONS, compute_icc
from survey.analysis.survey_loader import (
    aggregate_ratings,
    load_ratings,
    response_time_filter,
)

log = get_logger(__name__)

CEILING_FLOOR_THRESHOLD = 0.60  # > 60% at max/min = flag
ICC_TARGET = 0.50


def run(study_id: str = "pilot_v1", out_dir: str = "./artifacts/pilot") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = load_ratings(study_id, exclude_failed_attention=False)
    if df_raw.empty:
        raise SystemExit(f"No ratings for study_id={study_id!r}")

    log.info(f"Pilot raw: {len(df_raw)} ratings, {df_raw['participant_id'].nunique()} participants")

    # --- Exclusions ---
    n_attn_fail = df_raw[df_raw["attention_check_pass"] == False]["participant_id"].nunique()
    df = load_ratings(study_id, exclude_failed_attention=True)
    df = response_time_filter(df, min_ms=3000)
    log.info(f"After exclusions: {len(df)} ratings, {df['participant_id'].nunique()} participants")

    results: dict = {
        "study_id": study_id,
        "n_raw_participants": df_raw["participant_id"].nunique(),
        "n_excluded_attention": n_attn_fail,
        "n_included_participants": df["participant_id"].nunique(),
        "n_ratings": len(df),
        "icc": {},
        "descriptives": {},
        "ceiling_floor_flags": [],
        "timing": {},
    }

    # --- ICC per dimension ---
    for dim in SURVEY_DIMENSIONS:
        col = dim
        if col not in df.columns or df[col].notna().sum() < 10:
            continue
        try:
            icc = compute_icc(df, rating_col=col)
            results["icc"][dim] = {
                "icc31": icc.icc31,
                "icc3k": icc.icc3k,
                "ci_low": icc.ci_low,
                "ci_high": icc.ci_high,
                "pass": icc.icc3k >= ICC_TARGET,
            }
            flag = "✓" if icc.icc3k >= ICC_TARGET else "✗ BELOW TARGET"
            log.info(
                f"{dim}: ICC(3,k)={icc.icc3k:.3f} "
                f"[{icc.ci_low:.3f},{icc.ci_high:.3f}] {flag}"
            )
        except Exception as e:
            log.warning(f"ICC failed for {dim}: {e}")

    # --- Descriptives + ceiling/floor ---
    for dim in SURVEY_DIMENSIONS:
        if dim not in df.columns:
            continue
        s = df[dim].dropna()
        if len(s) == 0:
            continue
        desc = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "pct_floor": float((s == 1).mean()),
            "pct_ceiling": float((s == 7).mean()),
        }
        results["descriptives"][dim] = desc
        if desc["pct_floor"] > CEILING_FLOOR_THRESHOLD:
            results["ceiling_floor_flags"].append(f"{dim}: FLOOR ({desc['pct_floor']:.0%} at 1)")
        if desc["pct_ceiling"] > CEILING_FLOOR_THRESHOLD:
            results["ceiling_floor_flags"].append(f"{dim}: CEILING ({desc['pct_ceiling']:.0%} at 7)")

    # --- Response time ---
    if "response_time_ms" in df.columns:
        rt = df["response_time_ms"].dropna()
        results["timing"] = {
            "median_ms": float(rt.median()),
            "pct_under_3s": float((rt < 3000).mean()),
            "pct_under_10s": float((rt < 10000).mean()),
        }

    # --- Free text sample ---
    free_texts = df["free_text"].dropna().str.strip().loc[lambda s: s.str.len() > 0]
    sample = free_texts.sample(min(20, len(free_texts)), random_state=0).tolist() if len(free_texts) else []
    results["free_text_sample"] = sample

    # --- Write outputs ---
    (out / "pilot_report.json").write_text(json.dumps(results, indent=2))
    log.info(f"Pilot report saved to {out / 'pilot_report.json'}")

    # Print summary
    print("\n=== PILOT SUMMARY ===")
    print(f"Participants: {results['n_included_participants']} included, {n_attn_fail} excluded (attention)")
    print(f"Ratings: {results['n_ratings']}")
    print("\nICC(3,k) per dimension:")
    for dim, v in results["icc"].items():
        flag = "✓" if v["pass"] else "✗"
        print(f"  {flag} {dim}: {v['icc3k']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    if results["ceiling_floor_flags"]:
        print("\n⚠ Ceiling/floor flags:")
        for f in results["ceiling_floor_flags"]:
            print(f"  {f}")
    print(f"\nMedian response time: {results['timing'].get('median_ms', '?')/1000:.1f}s")


if __name__ == "__main__":
    typer.run(run)
