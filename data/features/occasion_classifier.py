"""Multi-label occasion classifier.

Two-stage training:
1. Weak supervision from keyword rules → produce a noisy initial dataset
2. Fine-tune DistilBERT multi-label on confident keyword-rule positives;
   bootstrap: iteratively retrain on model-confident predictions.

Input:  listing title + description + tags (space-joined)
Output: multi-label over OCCASIONS with per-label confidence

Predictions stored in listing_features.occasion (top-1), occasion_confidence,
and occasion_multilabel (full probability vector).

Usage:
    # Train
    python -m data.features.occasion_classifier train --epochs 5
    # Infer missing
    python -m data.features.occasion_classifier infer
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from common.db import connection
from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

log = get_logger(__name__)

app = typer.Typer()

MODEL_ID = "distilbert-base-uncased"
CKPT_PATH = Path("./artifacts/occasion_classifier.pt")

OCCASION_TO_IDX = {o: i for i, o in enumerate(OCCASIONS)}
IDX_TO_OCCASION = {i: o for i, o in enumerate(OCCASIONS)}


# ---------------------------------------------------------------------------
# Keyword rules (weak supervision seed)
# ---------------------------------------------------------------------------
_RULES: dict[str, list[str]] = {
    "birthday/general": ["birthday", "bday", "happy birthday"],
    "birthday/milestone": ["18th", "21st", "30th", "40th", "50th", "60th", "70th", "80th", "milestone"],
    "birthday/kids": [
        "kids birthday", "birthday kids", "children's birthday", "birthday children",
        "birthday child", "child's birthday", "birthday boy", "birthday girl",
        "birthday son", "son birthday", "birthday daughter", "daughter birthday",
        "birthday nephew", "birthday niece", "birthday grandson", "birthday granddaughter",
        "1st birthday", "2nd birthday", "3rd birthday", "4th birthday", "5th birthday",
        "6th birthday", "7th birthday", "8th birthday", "9th birthday", "10th birthday",
        "11th birthday", "12th birthday",
        "first birthday", "second birthday", "third birthday",
        "age 1", "age 2", "age 3", "age 4", "age 5", "age 6", "age 7",
        "age 8", "age 9", "age 10", "age 11", "age 12",
        "little one", "toddler", "baby birthday",
    ],
    "birthday/relationship": [
        "boyfriend birthday", "birthday boyfriend", "boyfriend card", "boyfriend happy",
        "girlfriend birthday", "birthday girlfriend", "girlfriend card", "girlfriend happy",
        "husband birthday", "birthday husband", "husband card", "hubby birthday", "hubby card",
        "wife birthday", "birthday wife", "wife card", "wifey birthday", "wifey card",
        "partner birthday", "birthday partner", "partner card",
        "fiance birthday", "birthday fiance", "fiancee birthday", "birthday fiancee",
        "for him birthday", "birthday for him", "for her birthday", "birthday for her",
        "other half", "soulmate", "love of my life",
    ],
    "christmas/general": ["christmas", "xmas", "festive", "merry christmas"],
    "christmas/humorous": ["christmas funny", "funny christmas", "humorous christmas"],
    "mothers_day": ["mother's day", "mothers day", "mum birthday", "mom birthday"],
    "fathers_day": ["father's day", "fathers day", "dad birthday"],
    "valentines_day": ["valentines", "valentine", "love you", "be mine"],
    "easter": ["easter", "happy easter", "easter bunny"],
    "anniversary/general": ["anniversary", "years together", "years married"],
    "wedding/congratulations": ["wedding", "newly wed", "congratulations on your wedding"],
    "wedding/engagement": ["engagement", "engaged", "congratulations engaged"],
    "new_baby": ["new baby", "baby shower", "congratulations baby", "newborn"],
    "sympathy/bereavement": ["sympathy", "condolences", "sorry for your loss", "bereavement", "with deepest sympathy"],
    "sympathy/get_well": ["get well", "get better", "speedy recovery", "feel better"],
    "thank_you": ["thank you", "thanks", "grateful"],
    "congratulations/general": ["congratulations", "congrats", "well done"],
    "congratulations/exam": ["exam", "results", "a-level", "gcse", "degree"],
    "congratulations/new_job": ["new job", "promotion", "new role"],
    "congratulations/new_home": ["new home", "moving", "housewarming"],
    "leaving/retirement": ["retirement", "retiring", "happy retirement"],
    "leaving/job": ["leaving", "farewell", "goodbye", "new adventure"],
    "graduation": ["graduation", "graduate", "well done graduate"],
    "encouragement": ["thinking of you", "you've got this", "keep going"],
    "just_because": ["just because", "no occasion", "thinking of you"],
}

_COOCCURRENCE_RULES: dict[str, list[tuple[str, ...]]] = {
    "birthday/kids": [
        ("birthday", "kid"), ("birthday", "child"), ("birthday", "children"),
        ("birthday", "son"), ("birthday", "daughter"),
        ("birthday", "nephew"), ("birthday", "niece"),
        ("birthday", "grandson"), ("birthday", "granddaughter"),
        ("birthday", "toddler"), ("birthday", "baby"),
        ("birthday", "young"), ("birthday", "little"),
        ("bday", "kid"), ("bday", "child"), ("bday", "son"), ("bday", "daughter"),
    ],
    "birthday/relationship": [
        ("birthday", "husband"), ("birthday", "wife"),
        ("birthday", "boyfriend"), ("birthday", "girlfriend"),
        ("birthday", "partner"), ("birthday", "fiance"), ("birthday", "fiancee"),
        ("birthday", "hubby"), ("birthday", "wifey"),
        ("birthday", "for him"), ("birthday", "for her"),
        ("birthday", "soulmate"), ("birthday", "other half"),
        ("bday", "husband"), ("bday", "wife"),
        ("bday", "boyfriend"), ("bday", "girlfriend"),
    ],
}


def weak_label(text: str) -> list[str]:
    text_l = text.lower()
    labels = []
    for occasion, keywords in _RULES.items():
        if any(kw in text_l for kw in keywords):
            labels.append(occasion)
    for occasion, word_groups in _COOCCURRENCE_RULES.items():
        if occasion in labels:
            continue
        for words in word_groups:
            if all(w in text_l for w in words):
                labels.append(occasion)
                break
    return labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _model_cls():
    """Build the DistilBERT classifier class on demand.

    torch and transformers are needed only by this model path — `infer` runs
    pure keyword rules. Importing them at module scope cost several minutes
    on NFS for every caller that just wanted `weak_label`.
    """
    import torch
    from torch import nn
    from transformers import AutoModel

    class OccasionClassifier(nn.Module):
        def __init__(self, n_labels: int = len(OCCASIONS)):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(MODEL_ID)
            hidden = self.encoder.config.hidden_size
            self.head = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_labels),
            )

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            return torch.sigmoid(self.head(cls))

    return OccasionClassifier


def load_model(ckpt: Path | None = None):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = _model_cls()()
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.command()
def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    ckpt: Path = CKPT_PATH,
) -> None:
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from common.db import engine

    df = pd.read_sql(
        "SELECT listing_id, title, description FROM listings", engine()
    )
    texts = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.strip().tolist()

    model, tokenizer = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ys = []
    keep = []
    for i, t in enumerate(texts):
        lbls = weak_label(t)
        if not lbls:
            continue
        vec = [0.0] * len(OCCASIONS)
        for lbl in lbls:
            if lbl in OCCASION_TO_IDX:
                vec[OCCASION_TO_IDX[lbl]] = 1.0
        ys.append(vec)
        keep.append(i)

    if not keep:
        log.error("No weak labels found — check listings table is populated")
        raise SystemExit(1)

    texts_keep = [texts[i] for i in keep]
    enc = tokenizer(texts_keep, truncation=True, padding=True, max_length=128, return_tensors="pt")
    y_tensor = torch.tensor(ys, dtype=torch.float32)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            pred = model(ids, mask)
            loss = loss_fn(pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        log.info(f"epoch={epoch} loss={running/len(loader):.4f}")

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    log.info(f"Saved {ckpt}")


# ---------------------------------------------------------------------------
# Inference over missing listings
# ---------------------------------------------------------------------------
_SELECT_MISSING = """
SELECT l.listing_id,
       COALESCE(l.title,'') || ' ' || COALESCE(l.description,'') AS text
FROM listings l
LEFT JOIN listing_features lf ON lf.listing_id = l.listing_id
WHERE lf.occasion IS NULL
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_SELECT_ALL = """
SELECT l.listing_id,
       COALESCE(l.title,'') || ' ' || COALESCE(l.description,'') AS text
FROM listings l
JOIN listing_features lf ON lf.listing_id = l.listing_id
ORDER BY l.last_seen_at DESC
LIMIT %(limit)s;
"""

_UPSERT = """
INSERT INTO listing_features (listing_id, occasion, occasion_confidence, occasion_multilabel, feature_version)
VALUES (%(listing_id)s, %(occasion)s, %(confidence)s, %(multilabel)s, 'keyword-v2')
ON CONFLICT (listing_id) DO UPDATE
SET occasion = EXCLUDED.occasion,
    occasion_confidence = EXCLUDED.occasion_confidence,
    occasion_multilabel = EXCLUDED.occasion_multilabel,
    computed_at = NOW();
"""


_SPECIFICITY_ORDER = [o for o in OCCASIONS if not o.endswith("/general")] + \
                     [o for o in OCCASIONS if o.endswith("/general")]


def pick_best_occasion(labels: list[str]) -> str | None:
    """Pick most specific occasion from a list of keyword matches.

    Sub-occasions (kids, relationship, milestone) win over /general
    when both match.
    """
    if not labels:
        return None
    for occ in _SPECIFICITY_ORDER:
        if occ in labels:
            return occ
    return labels[0]


@app.command()
def infer(
    limit: int = 50000,
    reclassify_all: bool = typer.Option(True, help="Re-classify all listings"),
) -> None:
    """Classify listings using keyword rules (no model needed)."""
    query = _SELECT_ALL if reclassify_all else _SELECT_MISSING
    with connection() as conn, conn.cursor() as cur:
        cur.execute(query, {"limit": limit})
        rows = cur.fetchall()

    log.info(f"Classifying {len(rows)} listings with keyword rules")
    processed = 0
    stats: dict[str, int] = {}

    with connection() as conn, conn.cursor() as cur:
        for r in rows:
            text = r["text"] or ""
            labels = weak_label(text)
            occasion = pick_best_occasion(labels)
            if not occasion:
                continue
            confidence = 1.0 if len(labels) == 1 else 0.8
            multilabel = {o: (1.0 if o in labels else 0.0) for o in OCCASIONS}
            cur.execute(
                _UPSERT,
                {
                    "listing_id": r["listing_id"],
                    "occasion": occasion,
                    "confidence": confidence,
                    "multilabel": json.dumps(multilabel),
                },
            )
            stats[occasion] = stats.get(occasion, 0) + 1
            processed += 1

    log.info(f"Keyword classification complete: {processed} listings")
    for occ, n in sorted(stats.items(), key=lambda x: -x[1]):
        log.info(f"  {occ}: {n}")


if __name__ == "__main__":
    app()
