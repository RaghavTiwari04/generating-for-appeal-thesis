# %% [markdown]
# # 01 — Data Overview
#
# Exploratory notebook: counts, distributions, and spot-checks after
# initial scraping + feature extraction.
#
# Run with: `jupyter lab` or as a VS Code Jupyter notebook.
# All cells are self-contained and read-only w.r.t. the database.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from common.db import engine

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

# %% [markdown]
# ## 1. Listings counts by source and occasion

# %%
df_counts = pd.read_sql("""
    SELECT l.source,
           lf.occasion,
           COUNT(*) AS n,
           SUM(l.is_bestseller::int) AS n_bestseller,
           AVG(l.review_count) AS avg_reviews,
           AVG(l.favourite_count) AS avg_favourites
    FROM listings l
    LEFT JOIN listing_features lf USING (listing_id)
    GROUP BY l.source, lf.occasion
    ORDER BY n DESC
""", engine())

print(df_counts.head(20).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 4))
by_source = df_counts.groupby("source")["n"].sum().sort_values(ascending=False)
ax.bar(by_source.index, by_source.values)
ax.set_title("Listings per source")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Price distributions

# %%
df_price = pd.read_sql("""
    SELECT source, currency, price_minor_units
    FROM listings
    WHERE price_minor_units IS NOT NULL
      AND price_minor_units BETWEEN 50 AND 5000
""", engine())

df_price["price_gbp"] = df_price["price_minor_units"] / 100.0

fig, axes = plt.subplots(1, len(df_price["source"].unique()), figsize=(12, 3), sharey=False)
for ax, (src, grp) in zip(
    axes if hasattr(axes, "__len__") else [axes],
    df_price.groupby("source")
):
    ax.hist(grp["price_gbp"], bins=30, edgecolor="white", linewidth=0.5)
    ax.set_title(src)
    ax.set_xlabel("Price (GBP equiv.)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("£%.0f"))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Occasion class balance

# %%
occ_counts = (
    df_counts.groupby("occasion")["n"].sum()
    .sort_values(ascending=False)
    .dropna()
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(occ_counts.index, occ_counts.values)
ax.set_xlabel("Count")
ax.set_title("Listings per occasion")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Saleability proxy score distribution

# %%
df_proxy = pd.read_sql("""
    SELECT sl.score, lf.occasion
    FROM saleability_labels sl
    JOIN listing_features lf USING (listing_id)
    WHERE sl.label_source = 'proxy_v1'
""", engine())

if df_proxy.empty:
    print("No proxy labels yet — run `make proxy-labels`")
else:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df_proxy["score"], bins=50, edgecolor="white")
    ax.set_xlabel("Proxy saleability score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of proxy saleability labels")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Duplicate cluster sizes

# %%
df_dup = pd.read_sql("""
    SELECT duplicate_cluster_size, COUNT(*) AS n_clusters
    FROM listing_features
    WHERE duplicate_cluster_size IS NOT NULL
    GROUP BY duplicate_cluster_size
    ORDER BY duplicate_cluster_size
""", engine())

if df_dup.empty:
    print("No dedup results yet — run `make dedup`")
else:
    print(df_dup.head(20).to_string(index=False))
    total_duped = df_dup[df_dup["duplicate_cluster_size"] > 1]["n_clusters"].sum()
    print(f"\nListings in clusters > 1: {total_duped}")
