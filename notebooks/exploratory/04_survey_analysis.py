# %% [markdown]
# # 04 — Survey Analysis (Pilot + Main)
#
# After a Prolific study run: ICC, mean scores per dimension,
# comparison to proxy labels.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.db import engine
from survey.analysis.icc import compute_icc, SURVEY_DIMENSIONS
from survey.analysis.survey_loader import (
    load_ratings,
    aggregate_ratings,
    response_time_filter,
)

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

STUDY_ID = "main_v1"  # change to 'pilot_v1' for pilot

# %%
try:
    df = load_ratings(STUDY_ID)
    df = response_time_filter(df, min_ms=3000)
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Ratings: {len(df)}")
    print(f"Cards: {df['card_key'].nunique()}")
except Exception as e:
    print(f"DB error (run docker compose up first): {e}")
    df = pd.DataFrame()

# %% [markdown]
# ## ICC per dimension

# %%
if not df.empty:
    for dim in SURVEY_DIMENSIONS:
        if df[dim].notna().sum() < 20:
            continue
        try:
            result = compute_icc(df, rating_col=dim)
            print(f"{dim:25s}  ICC(3,1)={result.icc31:.3f}  ICC(3,k)={result.icc3k:.3f}  "
                  f"[{result.ci_low:.3f}, {result.ci_high:.3f}]  n_raters={result.n_raters}")
        except Exception as e:
            print(f"{dim}: {e}")

# %% [markdown]
# ## Mean scores per occasion

# %%
if not df.empty:
    by_occ = df.groupby("occasion_shown")[SURVEY_DIMENSIONS].mean()
    print(by_occ.round(2).to_string())

    fig, ax = plt.subplots(figsize=(12, 4))
    by_occ["purchase_intent"].sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(4.0, ls="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Mean purchase intent (1–7)")
    ax.set_title("Purchase intent by occasion")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Survey vs proxy correlation

# %%
if not df.empty:
    agg = aggregate_ratings(df)
    df_proxy = pd.read_sql("""
        SELECT listing_id::text AS card_key, score AS proxy_score
        FROM saleability_labels WHERE label_source = 'proxy_v1'
    """, engine())
    merged = agg.per_card.reset_index().merge(df_proxy, on="card_key", how="inner")
    if len(merged) > 10:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(merged["purchase_intent_mean"], merged["proxy_score"])
        print(f"Survey PI vs proxy: Spearman ρ = {rho:.3f}  p = {pval:.4f}  n = {len(merged)}")
    else:
        print("Not enough overlap between survey and proxy labels for correlation")
