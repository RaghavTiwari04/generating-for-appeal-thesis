# %% [markdown]
# # 02 — Predictor Analysis
#
# After training the predictor, inspect head performance, calibration,
# and failure modes.

# %%
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

ARTIFACTS = Path("../../artifacts/predictor")

# %% [markdown]
# ## 1. Test metrics

# %%
import json

metrics_path = ARTIFACTS / "test_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    df_metrics = pd.DataFrame([metrics]).T.rename(columns={0: "value"})
    print(df_metrics.to_string())
else:
    print(f"No test_metrics.json at {metrics_path} — train first: `make train-predictor`")

# %% [markdown]
# ## 2. Calibration reliability plot

# %%
cal_path = ARTIFACTS / "calibration.json"
if cal_path.exists():
    cal = json.loads(cal_path.read_text())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(cal["bin_predicted"], cal["bin_observed"], "o-", label=f"ECE={cal['ece']:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Reliability diagram — saleability head")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No calibration.json yet")

# %% [markdown]
# ## 3. Per-head Spearman by occasion

# %%
try:
    from common.db import engine

    # Labels come from the vision-language judge. The survey_ratings table this
    # once read is dropped by migration 0005; the study was never run.
    # `score` is only the sortable summary, so the dimensions are read out of
    # `raw`, the same way models/predictor/dataset.py does it.
    df_labels = pd.read_sql("""
        SELECT sl.listing_id, lf.occasion,
               (sl.raw->>'purchase_intent')::float      AS purchase_intent,
               (sl.raw->>'aesthetic')::float            AS aesthetic,
               (sl.raw->>'emotional_resonance')::float  AS emotional_resonance,
               (sl.raw->>'distinctiveness')::float      AS distinctiveness,
               (sl.raw->>'occasion_fit')::float         AS occasion_fit
        FROM saleability_labels sl
        JOIN listing_features lf USING (listing_id)
        WHERE sl.label_source = 'llm_ssr_rubric_v2'
    """, engine())
    print(f"Labelled cards: {len(df_labels)}")
    print(df_labels.groupby("occasion").size().sort_values(ascending=False).head(10))
except Exception as e:
    print(f"DB not available: {e}")
