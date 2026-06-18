"""Multi-head saleability predictor.

Frozen vision-language backbone (consumed via cached CLIP features) +
occasion embedding + price scalar → MLP trunk → five heads.

Heads 1-4 (occasion_fit, aesthetic, emotional_resonance, distinctiveness)
are supervised by VLM labels on all ~2,377 cards. Head 5 (purchase_intent)
is supervised by Bradley-Terry scores from a Prolific 2AFC study on a
~500-card subsample. Training uses masked multi-task loss so each head
only backpropagates on samples where its label exists (Ruder 2017).

Backbone is **not** loaded here — embeddings are cached in
`listing_features.clip_embedding` by `data/features/clip_embed.py`. This module
operates purely on cached features for speed and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

# Heads 1-4: VLM-labelled perceptual quality (all ~2,377 cards)
# Head 5: human purchase_intent (Prolific 2AFC, ~500 card subsample)
HEAD_NAMES: tuple[str, ...] = (
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
    "purchase_intent",
)

# Which heads have VLM labels (all cards) vs human labels (subsample only)
VLM_HEADS: tuple[str, ...] = ("occasion_fit", "aesthetic", "emotional_resonance", "distinctiveness")
HUMAN_HEADS: tuple[str, ...] = ("purchase_intent",)


@dataclass
class PredictorConfig:
    image_dim: int = 768
    text_dim: int = 768
    occasion_vocab: int = len(OCCASIONS)
    occasion_emb_dim: int = 32
    price_dim: int = 1
    trunk_hidden: int = 512
    head_hidden: int = 128
    dropout: float = 0.1
    head_names: tuple[str, ...] = field(default_factory=lambda: HEAD_NAMES)


class SaleabilityPredictor(nn.Module):
    """Multi-head saleability predictor.

    Forward inputs (all batched):
      image_emb:     (B, image_dim)   — CLIP image embedding (normalised)
      text_emb:      (B, text_dim)    — joint headline+inside-message text emb
      occasion_idx:  (B,)             — long, index into OCCASIONS
      price_rel:     (B, 1)           — log-price relative to occasion median

    Output: dict[head_name -> (B,)] of sigmoid-bounded scalars.
    """

    def __init__(self, cfg: PredictorConfig | None = None):
        super().__init__()
        self.cfg = cfg or PredictorConfig()

        self.occasion_emb = nn.Embedding(self.cfg.occasion_vocab, self.cfg.occasion_emb_dim)

        in_dim = (
            self.cfg.image_dim
            + self.cfg.text_dim
            + self.cfg.occasion_emb_dim
            + self.cfg.price_dim
        )

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.cfg.trunk_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.trunk_hidden, self.cfg.trunk_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
        )

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.cfg.trunk_hidden, self.cfg.head_hidden),
                    nn.GELU(),
                    nn.Linear(self.cfg.head_hidden, 1),
                )
                for name in self.cfg.head_names
            }
        )

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        occasion_idx: torch.Tensor,
        price_rel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        occ = self.occasion_emb(occasion_idx)
        x = torch.cat([image_emb, text_emb, occ, price_rel], dim=-1)
        z = self.trunk(x)
        return {name: torch.sigmoid(head(z)).squeeze(-1) for name, head in self.heads.items()}


def head_loss_weights(purchase_intent_factor: float = 2.0) -> dict[str, float]:
    """Purchase-intent head weighted 2× by default (§4.4).

    Upweight compensates for the smaller label set (~500 vs ~2,377).
    Masked loss already handles missing labels; this factor controls
    relative importance in the total loss.
    """
    return {name: (purchase_intent_factor if name == "purchase_intent" else 1.0) for name in HEAD_NAMES}
