"""Mine market signals to inject into the brief-generator prompt.

Four signals:
- top tropes for the occasion: cluster the top-saleability listings via
  CLIP embeddings, surface the cluster headline themes.
- bestseller subjects: frequency-ranked visual subjects from bestseller
  titles/descriptions — gives the LLM a data-grounded palette to riff on.
- coverage gaps: (occasion × tone) pairs underrepresented in the dataset.
- longevity caution: short prompt-side warning about time-bound references.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common.db import engine

LONGEVITY_CAUTION = (
    "Avoid current-events references, brand mentions, or anything that will "
    "feel dated within twelve months."
)

_SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "fox": ["fox", "foxes", "foxy"],
    "hedgehog": ["hedgehog", "hedgehogs"],
    "owl": ["owl", "owls"],
    "elephant": ["elephant", "elephants", "ellie"],
    "dinosaur": ["dinosaur", "dinosaurs", "dino", "dinos", "t-rex"],
    "unicorn": ["unicorn", "unicorns"],
    "cat": ["cat", "cats", "kitten", "kittens", "kitty"],
    "dog": ["dog", "dogs", "puppy", "puppies", "pup"],
    "bear": ["bear", "bears", "teddy"],
    "rabbit": ["rabbit", "rabbits", "bunny", "bunnies"],
    "penguin": ["penguin", "penguins"],
    "flamingo": ["flamingo", "flamingos"],
    "turtle": ["turtle", "turtles", "tortoise"],
    "sloth": ["sloth", "sloths"],
    "koala": ["koala", "koalas"],
    "panda": ["panda", "pandas"],
    "giraffe": ["giraffe", "giraffes"],
    "lion": ["lion", "lions"],
    "monkey": ["monkey", "monkeys"],
    "deer": ["deer", "stag", "fawn"],
    "bee": ["bee", "bees", "bumblebee"],
    "butterfly": ["butterfly", "butterflies"],
    "ladybug": ["ladybug", "ladybird"],
    "frog": ["frog", "frogs"],
    "duck": ["duck", "ducks", "duckling"],
    "llama": ["llama", "llamas", "alpaca"],
    "highland cow": ["highland cow", "highland"],
    "otter": ["otter", "otters"],
    "dragon": ["dragon", "dragons"],
    "mermaid": ["mermaid", "mermaids"],
    "astronaut": ["astronaut", "space", "rocket"],
    "pirate": ["pirate", "pirates"],
    "fairy": ["fairy", "fairies"],
    "robot": ["robot", "robots"],
    "floral arrangement": ["floral", "flowers", "flower", "botanical", "bouquet"],
    "roses": ["roses", "rose"],
    "sunflower": ["sunflower", "sunflowers"],
    "daisy": ["daisy", "daisies"],
    "jungle animals": ["jungle", "safari", "tropical"],
    "ocean scene": ["ocean", "sea", "underwater"],
}


@dataclass
class MarketSignals:
    top_tropes: list[str]
    bestseller_subjects: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    longevity_caution: str = LONGEVITY_CAUTION


_TOP_BY_OCCASION_SQL = """
SELECT lf.listing_id, lf.clip_embedding, COALESCE(lf.extracted_text, l.title) AS headline_text
FROM listing_features lf
JOIN listings l USING (listing_id)
LEFT JOIN saleability_labels sl
  ON sl.listing_id = lf.listing_id AND sl.label_source = 'vlm_4head_v1'
WHERE lf.occasion = %(occasion)s
  AND lf.clip_embedding IS NOT NULL
ORDER BY COALESCE(sl.score, 0) DESC
LIMIT %(limit)s;
"""


def top_tropes_for_occasion(occasion: str, *, k_clusters: int = 5, top_n: int = 100) -> list[str]:
    df = pd.read_sql(
        _TOP_BY_OCCASION_SQL,
        engine(),
        params={"occasion": occasion, "limit": top_n},
    )
    if df.empty:
        return []
    embeds = np.stack(df["clip_embedding"].apply(np.asarray).to_list())
    k = min(k_clusters, len(df))
    if k < 2:
        return [str(t)[:160] for t in df["headline_text"].head(5).tolist() if t]

    km = KMeans(n_clusters=k, n_init=4, random_state=0)
    labels = km.fit_predict(embeds)
    df["cluster"] = labels
    tropes: list[str] = []
    for c in range(k):
        sub = df[df["cluster"] == c]
        sample = (
            sub["headline_text"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .head(3)
            .tolist()
        )
        if sample:
            tropes.append(f"cluster_{c}: {' | '.join(s[:80] for s in sample)}")
    return tropes


_BESTSELLER_SUBJECTS_SQL = """
SELECT l.title, l.description
FROM listings l
JOIN listing_features lf USING (listing_id)
WHERE lf.occasion = %(occasion)s
  AND l.is_bestseller = TRUE
ORDER BY COALESCE(l.review_count, 0) + COALESCE(l.favourite_count, 0) DESC
LIMIT 500;
"""


def bestseller_subjects_for_occasion(occasion: str) -> list[str]:
    """Return visual subjects ranked by frequency in bestseller titles/descriptions."""
    df = pd.read_sql(_BESTSELLER_SUBJECTS_SQL, engine(), params={"occasion": occasion})
    if df.empty:
        return []
    counter: Counter = Counter()
    total = len(df)
    for _, row in df.iterrows():
        text = f"{row['title'] or ''} {row['description'] or ''}".lower()
        for subject, keywords in _SUBJECT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                counter[subject] += 1
    return [
        f"{subj} ({count}/{total} bestsellers, {count/total*100:.0f}%)"
        for subj, count in counter.most_common(20)
    ]


_GAP_SQL = """
SELECT lf.occasion, COUNT(*) AS n
FROM listing_features lf
GROUP BY lf.occasion
ORDER BY n ASC
LIMIT 5;
"""


def coverage_gaps() -> list[str]:
    df = pd.read_sql(_GAP_SQL, engine())
    return [f"low_volume:{r.occasion} (n={int(r.n)})" for r in df.itertuples()]


def gather(occasion: str) -> MarketSignals:
    try:
        tropes = top_tropes_for_occasion(occasion)
    except Exception:
        tropes = []
    try:
        subjects = bestseller_subjects_for_occasion(occasion)
    except Exception:
        subjects = []
    try:
        gaps = coverage_gaps()
    except Exception:
        gaps = []
    return MarketSignals(top_tropes=tropes, bestseller_subjects=subjects, coverage_gaps=gaps)


def render_for_prompt(signals: MarketSignals) -> dict[str, str]:
    return {
        "top_tropes": "\n".join(f"- {t}" for t in signals.top_tropes) or "  (none yet)",
        "bestseller_subjects": "\n".join(f"- {s}" for s in signals.bestseller_subjects) or "  (none yet)",
        "coverage_gaps": "\n".join(f"- {g}" for g in signals.coverage_gaps) or "  (none yet)",
        "longevity_caution": signals.longevity_caution,
    }
