# %% [markdown]
# # 05 — Pricing Model Analysis
#
# Train and evaluate the LightGBM price-band regressor.
# Requires `make embed-features` and `make proxy-labels` to have run first.

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from common.db import engine
from common.fx_rates import normalise_price_column

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

# %% [markdown]
# ## 1. Load pricing dataset

# %%
df = pd.read_sql("""
    SELECT
        l.listing_id,
        l.source AS marketplace,
        l.price_minor_units,
        l.currency,
        lf.occasion,
        lf.image_complexity,
        sl_proxy.score       AS proxy_score,
        sr_agg.aesthetic     AS aesthetic_score,
        sr_agg.distinctiveness AS distinctiveness_score
    FROM listings l
    JOIN listing_features lf USING (listing_id)
    LEFT JOIN saleability_labels sl_proxy
           ON sl_proxy.listing_id = l.listing_id AND sl_proxy.label_source = 'proxy_v1'
    LEFT JOIN (
        SELECT listing_id,
               AVG(aesthetic) AS aesthetic,
               AVG(distinctiveness) AS distinctiveness
        FROM survey_ratings GROUP BY listing_id
    ) sr_agg USING (listing_id)
    WHERE l.price_minor_units IS NOT NULL
      AND l.currency IS NOT NULL
      AND lf.occasion IS NOT NULL
""", engine())

normalise_price_column(df)
df = df[df["price_gbp"].between(0.5, 20.0)].reset_index(drop=True)
print(f"Rows: {len(df)}  Occasions: {df['occasion'].nunique()}  Sources: {df['marketplace'].nunique()}")

# %% [markdown]
# ## 2. Price distribution by occasion

# %%
top_occs = df.groupby("occasion")["listing_id"].count().nlargest(8).index.tolist()
fig, ax = plt.subplots(figsize=(10, 4))
for occ in top_occs:
    sub = df[df["occasion"] == occ]["price_gbp"]
    ax.plot(sorted(sub), np.linspace(0, 1, len(sub)), label=occ.replace("/", " / "), linewidth=1.2)
ax.set_xlabel("Price (GBP)")
ax.set_ylabel("CDF")
ax.set_title("Price CDF by top-8 occasions")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train pricing model

# %%
from models.pricing.train_pricing import FEATURE_COLS, CAT_COLS
from models.pricing.price_model import train, predict, _band

# Fill missing features with defaults
df["aesthetic_score"] = df["aesthetic_score"].fillna(0.5)
df["distinctiveness_score"] = df["distinctiveness_score"].fillna(0.5)
df["image_complexity"] = df["image_complexity"].fillna(0.3)

train_df = df.dropna(subset=["price_gbp"] + [c for c in FEATURE_COLS if c in df.columns])
bundle = train(train_df, feature_cols=FEATURE_COLS, categorical_cols=CAT_COLS)
print(f"Trained on {len(train_df)} listings")

# %% [markdown]
# ## 4. Predictions vs actuals

# %%
preds = predict(bundle, train_df)
train_df = train_df.copy()
train_df["pred_gbp"] = preds["price_gbp_median"].values
train_df["band_pred"] = preds["band"].values
train_df["band_actual"] = train_df["price_gbp"].apply(_band)

# Band confusion matrix
conf = pd.crosstab(train_df["band_actual"], train_df["band_pred"],
                   rownames=["Actual"], colnames=["Predicted"])
print(conf)

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(train_df["price_gbp"], train_df["pred_gbp"], alpha=0.15, s=8, color="#3b82f6")
lo, hi = train_df["price_gbp"].min(), train_df["price_gbp"].max()
ax.plot([lo, hi], [lo, hi], "--", color="#6b7280", linewidth=0.8)
ax.set_xlabel("Actual price (GBP)")
ax.set_ylabel("Predicted price (GBP)")
ax.set_title("Pricing model: predicted vs actual")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Feature importances

# %%
importances = pd.Series(
    bundle.median.feature_importance(importance_type="gain"),
    index=FEATURE_COLS,
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(6, 3))
importances.plot(kind="barh", ax=ax, color="#3b82f6")
ax.set_title("Feature importance (gain) — median booster")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Save model

# %%
MODEL_DIR = Path("../../artifacts/pricing")
bundle.save(MODEL_DIR)
print(f"Saved pricing model to {MODEL_DIR}")
