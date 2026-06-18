"""Response quality checks and exclusion criteria.

Applied before any analysis or label generation. Pre-registered exclusions
(see survey/preregistration/system_eval_v1.md §6):

1. Failed > 1 attention check
2. Median per-card response time < 3 000 ms for > 20% of cards
3. Straight-lining: all ratings on any one dimension are identical
4. Participated in an excluded study (enforced at Prolific recruitment, but
   double-checked here)

Usage:
    df = load_ratings("main_v1")
    report = quality_report(df)
    df_clean = apply_exclusions(df, report)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from common.logging import get_logger
from survey.analysis.survey_loader import LIKERT_DIMENSIONS

log = get_logger(__name__)

FAST_RESPONSE_MS = 3_000
FAST_RESPONSE_THRESHOLD = 0.20   # > 20% of cards too fast → exclude
ATTENTION_CHECK_MAX_FAILS = 1


@dataclass
class ParticipantQuality:
    participant_id: str
    n_ratings: int
    n_attention_checks: int
    n_attention_failures: int
    median_response_ms: float
    fast_response_fraction: float
    straight_line_dimensions: list[str]
    excluded: bool
    exclusion_reasons: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    study_id: str
    n_participants_total: int
    n_excluded: int
    n_retained: int
    exclusion_counts: dict[str, int]
    per_participant: list[ParticipantQuality]

    def excluded_ids(self) -> set[str]:
        return {p.participant_id for p in self.per_participant if p.excluded}

    def retained_ids(self) -> set[str]:
        return {p.participant_id for p in self.per_participant if not p.excluded}


def _check_straight_lining(group: pd.DataFrame) -> list[str]:
    """Return dimensions where participant gave identical rating to ALL cards."""
    bad = []
    for dim in LIKERT_DIMENSIONS:
        col = group[dim].dropna()
        if len(col) >= 5 and col.nunique() == 1:
            bad.append(dim)
    return bad


def quality_report(df: pd.DataFrame, study_id: str = "") -> QualityReport:
    participants = df["participant_id"].unique()
    results: list[ParticipantQuality] = []
    exclusion_counts: dict[str, int] = {
        "attention_checks": 0,
        "too_fast": 0,
        "straight_liner": 0,
    }

    for pid in participants:
        grp = df[df["participant_id"] == pid]
        n = len(grp)

        # Attention checks
        attn = grp[grp["attention_check_pass"].notna()]
        n_checks = len(attn)
        n_fails = int((~attn["attention_check_pass"].astype(bool)).sum())

        # Response speed
        rt = grp["response_time_ms"].dropna()
        med_rt = float(rt.median()) if len(rt) else float("nan")
        fast_frac = float((rt < FAST_RESPONSE_MS).sum() / n) if n else 0.0

        # Straight-lining
        sl_dims = _check_straight_lining(grp)

        reasons = []
        if n_fails > ATTENTION_CHECK_MAX_FAILS:
            reasons.append(f"attention_checks_failed={n_fails}")
            exclusion_counts["attention_checks"] += 1
        if fast_frac > FAST_RESPONSE_THRESHOLD:
            reasons.append(f"fast_response_fraction={fast_frac:.2f}")
            exclusion_counts["too_fast"] += 1
        if sl_dims:
            reasons.append(f"straight_lining={sl_dims}")
            exclusion_counts["straight_liner"] += 1

        results.append(ParticipantQuality(
            participant_id=pid,
            n_ratings=n,
            n_attention_checks=n_checks,
            n_attention_failures=n_fails,
            median_response_ms=med_rt,
            fast_response_fraction=fast_frac,
            straight_line_dimensions=sl_dims,
            excluded=bool(reasons),
            exclusion_reasons=reasons,
        ))

    n_excl = sum(1 for r in results if r.excluded)
    return QualityReport(
        study_id=study_id,
        n_participants_total=len(participants),
        n_excluded=n_excl,
        n_retained=len(participants) - n_excl,
        exclusion_counts=exclusion_counts,
        per_participant=results,
    )


def apply_exclusions(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Return df with excluded participants removed."""
    excl = report.excluded_ids()
    if excl:
        log.info(
            f"Excluding {len(excl)} participants "
            f"({report.exclusion_counts}). "
            f"Retained: {report.n_retained}/{report.n_participants_total}"
        )
    return df[~df["participant_id"].isin(excl)].reset_index(drop=True)


def flag_suspicious_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Add `suspicious` bool column per rating row.

    Flags: response_time_ms < 1000 (almost certainly not looked at image).
    """
    df = df.copy()
    df["suspicious"] = df["response_time_ms"].fillna(9999) < 1_000
    n = df["suspicious"].sum()
    if n:
        log.info(f"Flagged {n} suspicious ratings (response_time_ms < 1s)")
    return df


def quality_summary_df(report: QualityReport) -> pd.DataFrame:
    from dataclasses import asdict
    rows = [asdict(p) for p in report.per_participant]
    return pd.DataFrame(rows).sort_values("excluded", ascending=False)
