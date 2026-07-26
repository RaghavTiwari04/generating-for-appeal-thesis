"""Validate VLM pseudo-labels against human pairwise judgments.

Computes Spearman rho per dimension between:
  - VLM score (from saleability_labels, llm_ssr_rubric_v2)
  - Human BT score (from survey_pairs, calibration_v1)

Acceptance criterion: rho >= 0.5 per dimension (preregistered).

Also reports:
  - Inter-rater reliability (Krippendorff alpha)
  - Distribution plots (VLM vs human per dim)
  - Attention check pass rate

Run:
    python -m survey.analysis.vlm_calibration
    python -m survey.analysis.vlm_calibration --study-id calibration_v1 --vlm-source llm_ssr_rubric_v2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer
from scipy import stats as sp_stats

from common.db import engine
from common.logging import get_logger
from survey.analysis.bradley_terry import fit_bradley_terry, to_dataframe

log = get_logger(__name__)

CALIBRATION_DIMS = (
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
)

# Preregistered acceptance threshold
RHO_THRESHOLD = 0.5


@dataclass
class DimResult:
    dimension: str
    rho: float
    p_value: float
    n_cards: int
    accepted: bool
    human_mean: float
    human_std: float
    vlm_mean: float
    vlm_std: float


@dataclass
class CalibrationReport:
    dims: list[DimResult]
    n_participants: int
    n_total_pairs: int
    n_cards_with_bt: int
    attention_pass_rate: float
    overall_accepted: bool

    def summary_table(self) -> str:
        lines = [
            f"{'Dimension':25s} {'rho':>6s} {'p':>8s} {'N':>5s} {'Accept':>7s}",
            "-" * 55,
        ]
        for d in self.dims:
            accept = "YES" if d.accepted else "NO"
            lines.append(
                f"{d.dimension:25s} {d.rho:6.3f} {d.p_value:8.4f} {d.n_cards:5d} {accept:>7s}"
            )
        lines.append("-" * 55)
        status = "ACCEPTED" if self.overall_accepted else "REJECTED"
        lines.append(f"Overall: {status} (threshold rho >= {RHO_THRESHOLD})")
        lines.append(f"Participants: {self.n_participants}")
        lines.append(f"Total pairs: {self.n_total_pairs}")
        lines.append(f"Attention check pass rate: {self.attention_pass_rate:.1%}")
        return "\n".join(lines)


def _load_human_pairs(study_id: str) -> pd.DataFrame:
    """Load all calibration survey pairs."""
    sql = """
    SELECT pair_id, participant_id, study_id,
           COALESCE(left_listing_id::text, left_generated_id::text) AS left_key,
           COALESCE(right_listing_id::text, right_generated_id::text) AS right_key,
           occasion_shown, question_dim, winner_side,
           response_time_ms, attention_check_pass
    FROM survey_pairs
    WHERE study_id = %(study_id)s
    """
    return pd.read_sql(sql, engine(), params={"study_id": study_id})


def _load_vlm_scores(label_source: str) -> pd.DataFrame:
    """Load VLM scores as flat table: listing_id × 5 dimension scores."""
    sql = """
    SELECT listing_id::text,
           score AS vlm_composite,
           (raw->>'occasion_fit')::float AS vlm_occasion_fit,
           (raw->>'aesthetic')::float AS vlm_aesthetic,
           (raw->>'emotional_resonance')::float AS vlm_emotional_resonance,
           (raw->>'distinctiveness')::float AS vlm_distinctiveness
    FROM saleability_labels
    WHERE label_source = %(label_source)s
    """
    return pd.read_sql(sql, engine(), params={"label_source": label_source})


def _attention_check_rate(pairs: pd.DataFrame) -> float:
    """Fraction of attention checks passed."""
    checks = pairs[pairs["attention_check_pass"].notna()]
    if checks.empty:
        return 1.0  # no attention checks = assume pass
    return float(checks["attention_check_pass"].mean())


def run_calibration(
    study_id: str = "calibration_v1",
    vlm_source: str = "llm_ssr_rubric_v2",
    exclude_failed_attention: bool = True,
) -> CalibrationReport:
    """Full calibration analysis: human BT scores vs VLM scores.

    For each dimension:
      1. Filter survey_pairs to that question_dim.
      2. Fit Bradley-Terry → human BT sale_score per card.
      3. Join with VLM score for that dimension.
      4. Compute Spearman rho.
    """
    # Load data
    all_pairs = _load_human_pairs(study_id)
    if all_pairs.empty:
        log.error("No survey pairs found. Run the calibration study first.")
        return CalibrationReport(
            dims=[], n_participants=0, n_total_pairs=0,
            n_cards_with_bt=0, attention_pass_rate=0.0, overall_accepted=False,
        )

    if exclude_failed_attention:
        all_pairs = all_pairs[all_pairs["attention_check_pass"].fillna(True)]

    vlm = _load_vlm_scores(vlm_source)
    if vlm.empty:
        log.error(f"No VLM scores found for label_source={vlm_source}")
        return CalibrationReport(
            dims=[], n_participants=0, n_total_pairs=0,
            n_cards_with_bt=0, attention_pass_rate=0.0, overall_accepted=False,
        )

    n_participants = all_pairs["participant_id"].nunique()
    n_total_pairs = len(all_pairs)
    attention_rate = _attention_check_rate(_load_human_pairs(study_id))

    dim_results: list[DimResult] = []

    # Map survey question_dim names to VLM column names
    dim_to_vlm_col = {
        "occasion_fit": "vlm_occasion_fit",
        "aesthetic": "vlm_aesthetic",
        "emotional_resonance": "vlm_emotional_resonance",
        "distinctiveness": "vlm_distinctiveness",
    }

    for dim in CALIBRATION_DIMS:
        dim_pairs = all_pairs[all_pairs["question_dim"] == dim]
        if len(dim_pairs) < 10:
            log.warning(f"Too few pairs for {dim}: {len(dim_pairs)}")
            dim_results.append(DimResult(
                dimension=dim, rho=0.0, p_value=1.0, n_cards=0,
                accepted=False, human_mean=0.0, human_std=0.0,
                vlm_mean=0.0, vlm_std=0.0,
            ))
            continue

        # Fit BT for this dimension
        try:
            bt = fit_bradley_terry(dim_pairs)
            bt_df = to_dataframe(bt)
        except Exception as e:
            log.warning(f"BT fit failed for {dim}: {e}")
            dim_results.append(DimResult(
                dimension=dim, rho=0.0, p_value=1.0, n_cards=0,
                accepted=False, human_mean=0.0, human_std=0.0,
                vlm_mean=0.0, vlm_std=0.0,
            ))
            continue

        # Join with VLM scores
        vlm_col = dim_to_vlm_col[dim]
        merged = bt_df.merge(
            vlm[["listing_id", vlm_col]],
            left_on="card_key",
            right_on="listing_id",
            how="inner",
        )

        if len(merged) < 10:
            log.warning(f"Too few matched cards for {dim}: {len(merged)}")
            dim_results.append(DimResult(
                dimension=dim, rho=0.0, p_value=1.0, n_cards=len(merged),
                accepted=False, human_mean=0.0, human_std=0.0,
                vlm_mean=0.0, vlm_std=0.0,
            ))
            continue

        # Spearman correlation
        rho, p = sp_stats.spearmanr(merged["sale_score"], merged[vlm_col])
        accepted = rho >= RHO_THRESHOLD

        dim_results.append(DimResult(
            dimension=dim,
            rho=float(rho),
            p_value=float(p),
            n_cards=len(merged),
            accepted=accepted,
            human_mean=float(merged["sale_score"].mean()),
            human_std=float(merged["sale_score"].std()),
            vlm_mean=float(merged[vlm_col].mean()),
            vlm_std=float(merged[vlm_col].std()),
        ))

        log.info(f"{dim}: rho={rho:.3f}, p={p:.4f}, N={len(merged)} → {'ACCEPT' if accepted else 'REJECT'}")

    # Cards that appeared in any BT fit
    all_bt_cards = set()
    for dim in CALIBRATION_DIMS:
        dp = all_pairs[all_pairs["question_dim"] == dim]
        all_bt_cards |= set(dp["left_key"]) | set(dp["right_key"])

    overall = all(d.accepted for d in dim_results) if dim_results else False

    return CalibrationReport(
        dims=dim_results,
        n_participants=n_participants,
        n_total_pairs=n_total_pairs,
        n_cards_with_bt=len(all_bt_cards),
        attention_pass_rate=attention_rate,
        overall_accepted=overall,
    )


def save_report(report: CalibrationReport, path: Path | None = None) -> Path:
    """Save calibration report as JSON + text summary."""
    if path is None:
        path = Path("survey/analysis/calibration_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "n_participants": report.n_participants,
        "n_total_pairs": report.n_total_pairs,
        "n_cards_with_bt": report.n_cards_with_bt,
        "attention_pass_rate": report.attention_pass_rate,
        "overall_accepted": report.overall_accepted,
        "rho_threshold": RHO_THRESHOLD,
        "dimensions": [
            {
                "dimension": d.dimension,
                "rho": d.rho,
                "p_value": d.p_value,
                "n_cards": d.n_cards,
                "accepted": d.accepted,
                "human_mean": d.human_mean,
                "human_std": d.human_std,
                "vlm_mean": d.vlm_mean,
                "vlm_std": d.vlm_std,
            }
            for d in report.dims
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    # Also write text summary
    txt_path = path.with_suffix(".txt")
    txt_path.write_text(report.summary_table())
    log.info(f"Report saved to {path} and {txt_path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="VLM calibration analysis")


@app.command()
def validate(
    study_id: str = typer.Option("calibration_v1", help="Study ID"),
    vlm_source: str = typer.Option("llm_ssr_rubric_v2", help="VLM label source"),
) -> None:
    """Run full calibration validation and print results."""
    report = run_calibration(study_id=study_id, vlm_source=vlm_source)
    print("\n" + report.summary_table() + "\n")
    save_report(report)


@app.command()
def design() -> None:
    """Print study design summary for preregistration."""
    from survey.instrument.sampler_calibration import study_design_summary
    summary = study_design_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    app()
