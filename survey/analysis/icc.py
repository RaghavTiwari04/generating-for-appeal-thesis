"""Inter-rater reliability via Intraclass Correlation Coefficient.

Uses ICC(3,k) — two-way mixed, absolute agreement, average measures.
This is the standard choice for fixed raters rating fixed items on a
continuous/ordinal scale.

Also computes ICC(3,1) — single-measure variant, for reporting.

Reference: Shrout & Fleiss (1979); implemented via pingouin.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ICCResult:
    icc31: float       # ICC(3,1) — single measure
    icc3k: float       # ICC(3,k) — average measures
    ci_low: float
    ci_high: float
    f_value: float
    p_value: float
    n_raters: int
    n_items: int


def compute_icc(
    df: pd.DataFrame,
    *,
    item_col: str = "listing_id",
    rater_col: str = "participant_id",
    rating_col: str = "purchase_intent",
) -> ICCResult:
    """Compute ICC on a long-format ratings dataframe.

    df must have columns [item_col, rater_col, rating_col].
    """
    import pingouin as pg

    wide = df.pivot_table(
        index=item_col,
        columns=rater_col,
        values=rating_col,
        aggfunc="mean",
    )
    n_items, n_raters = wide.shape

    icc_df = pg.intraclass_corr(
        data=df,
        targets=item_col,
        raters=rater_col,
        ratings=rating_col,
    ).set_index("Type")

    row1 = icc_df.loc["ICC3"]
    rowk = icc_df.loc["ICC3k"]

    return ICCResult(
        icc31=float(row1["ICC"]),
        icc3k=float(rowk["ICC"]),
        ci_low=float(rowk["CI95%"][0]),
        ci_high=float(rowk["CI95%"][1]),
        f_value=float(row1["F"]),
        p_value=float(row1["pval"]),
        n_raters=n_raters,
        n_items=n_items,
    )


def icc_per_dimension(df: pd.DataFrame, dimensions: list[str], **kwargs) -> dict[str, ICCResult]:
    """Compute ICC for each Likert dimension."""
    return {dim: compute_icc(df, rating_col=dim, **kwargs) for dim in dimensions}


SURVEY_DIMENSIONS = [
    "purchase_intent",
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
]
