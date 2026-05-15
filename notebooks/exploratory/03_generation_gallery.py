# %% [markdown]
# # 03 — Generation Gallery
#
# Pull generated cards from the DB and display them in a grid,
# grouped by condition and occasion.

# %%
import warnings
warnings.filterwarnings("ignore")

import io
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from common.db import engine
from common.storage import get_object

# %%
df = pd.read_sql("""
    SELECT card_id, condition_tag,
           (brief->'request'->>'occasion') AS occasion,
           headline_text,
           cover_path,
           (predicted_scores->>'saleability_calibrated')::float AS sale_score,
           generated_at
    FROM generated_cards
    WHERE cover_path IS NOT NULL
    ORDER BY generated_at DESC
    LIMIT 40
""", engine())

print(f"Cards: {len(df)}")
print(df.groupby("condition_tag").size())

# %% [markdown]
# ## Grid display

# %%
def show_grid(subset: pd.DataFrame, ncols: int = 4, title: str = "") -> None:
    nrows = -(-len(subset) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, subset.iterrows()):
        try:
            data = get_object(row["cover_path"])
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((300, 400))
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, fontsize=7)
        score = row.get("sale_score")
        score_str = f"{score:.2f}" if score is not None and score == score else "?"
        ax.set_title(
            f"{row.get('occasion','?')}\n{score_str}",
            fontsize=7,
            pad=2,
        )
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


for cond, grp in df.groupby("condition_tag"):
    show_grid(grp.head(8), title=cond)
