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

    df_survey = pd.read_sql("""
        SELECT sr.listing_id, sr.occasion_shown,
               AVG(sr.purchase_intent) AS purchase_intent,
               AVG(sr.aesthetic) AS aesthetic,
               AVG(sr.emotional_resonance) AS emotional_resonance,
               AVG(sr.distinctiveness) AS distinctiveness,
               AVG(sr.occasion_fit) AS occasion_fit
        FROM survey_ratings sr
        WHERE sr.study_id IN ('pilot_v1','main_v1')
        GROUP BY sr.listing_id, sr.occasion_shown
    """, engine())
    print(f"Survey cards: {len(df_survey)}")
    print(df_survey.groupby("occasion_shown").size().sort_values(ascending=False).head(10))
except Exception as e:
    print(f"DB not available: {e}")
