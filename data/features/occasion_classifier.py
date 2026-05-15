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
from typing import Iterable

import numpy as np
import torch
import typer
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common.db import connection
from common.logging import get_logger
from common.occasions import OCCASIONS

log = get_logger(__name__)

app = typer.Typer()

MODEL_ID = "distilbert-base-uncased"
CKPT_PATH = Path("./artifacts/occasion_classifier.pt")
THRESHOLD = 0.4

OCCASION_TO_IDX = {o: i for i, o in enumerate(OCCASIONS)}
IDX_TO_OCCASION = {i: o for i, o in enumerate(OCCASIONS)}


# ---------------------------------------------------------------------------
# Keyword rules (weak supervision seed)
# ---------------------------------------------------------------------------
_RULES: dict[str, list[str]] = {
    "birthday/general": ["birthday", "bday", "happy birthday"],
    "birthday/milestone": ["18th", "21st", "30th", "40th", "50th", "60th", "70th", "80th", "milestone"],
    "birthday/kids": ["kids birthday", "children's birthday", "age 1", "age 2", "age 3", "age 4", "age 5"],
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


def weak_label(text: str) -> list[str]:
    text_l = text.lower()
    labels = []
    for occasion, keywords in _RULES.items():
        if any(kw in text_l for kw in keywords):
            labels.append(occasion)
    return labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return torch.sigmoid(self.head(cls))


def load_model(ckpt: Path | None = None) -> tuple[OccasionClassifier, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OccasionClassifier()
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
        for l in lbls:
            if l in OCCASION_TO_IDX:
                vec[OCCASION_TO_IDX[l]] = 1.0
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

_UPSERT = """
INSERT INTO listing_features (listing_id, occasion, occasion_confidence, occasion_multilabel, feature_version)
VALUES (%(listing_id)s, %(occasion)s, %(confidence)s, %(multilabel)s, 'occ-clf-v1')
ON CONFLICT (listing_id) DO UPDATE
SET occasion = EXCLUDED.occasion,
    occasion_confidence = EXCLUDED.occasion_confidence,
    occasion_multilabel = EXCLUDED.occasion_multilabel,
    computed_at = NOW();
"""


@app.command()
def infer(limit: int = 2000, ckpt: Path = CKPT_PATH, batch: int = 64) -> None:
    model, tokenizer = load_model(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    with connection() as conn, conn.cursor() as cur:
        cur.execute(_SELECT_MISSING, {"limit": limit})
        rows = cur.fetchall()

    log.info(f"Inferring occasions for {len(rows)} listings")
    processed = 0

    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        texts = [r["text"] or "" for r in chunk]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.inference_mode():
            probs = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = probs.cpu().numpy()

        with connection() as conn, conn.cursor() as cur:
            for r, prob_vec in zip(chunk, probs):
                top_idx = int(np.argmax(prob_vec))
                cur.execute(
                    _UPSERT,
                    {
                        "listing_id": r["listing_id"],
                        "occasion": IDX_TO_OCCASION[top_idx],
                        "confidence": float(prob_vec[top_idx]),
                        "multilabel": json.dumps(
                            {IDX_TO_OCCASION[i]: float(p) for i, p in enumerate(prob_vec)}
                        ),
                    },
                )
        processed += len(chunk)

    log.info(f"Occasion inference complete: {processed} listings")


if __name__ == "__main__":
    app()
