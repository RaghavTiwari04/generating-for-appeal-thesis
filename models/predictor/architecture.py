"""Multi-head saleability predictor.

Frozen vision-language backbone (consumed via cached CLIP features) +
occasion embedding → MLP trunk → five heads.

Price is deliberately not an input. It cannot be known for a generated card —
the thing the predictor exists to rank — and on the training corpus its
missingness is informative: free listings have no price, priced ones do, so a
price channel lets the trunk read the scrape source and use it as a shortcut
for quality.

All five heads are supervised by the VLM labels in `saleability_labels`:
the rubric judge for the four quality dimensions, SSR for purchase intent.
There is no human study — SSR stands in for one, on the strength of the
paper's own validation against human survey distributions.

The loss is masked per head (Ruder 2017), so a dimension the judge failed to
produce for a card contributes no gradient for that card rather than being
imputed.

Backbone is **not** loaded here — embeddings are cached in
`listing_features.clip_embedding` by `data/features/clip_embed.py`. This module
operates purely on cached features for speed and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from common.occasions import ACTIVE_OCCASIONS as OCCASIONS

HEAD_NAMES: tuple[str, ...] = (
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
    "purchase_intent",
)

# The rubric judge supplies these four; SSR supplies purchase_intent, which is
# read from the same `raw` payload but by a different instrument.
VLM_HEADS: tuple[str, ...] = (
    "occasion_fit",
    "aesthetic",
    "emotional_resonance",
    "distinctiveness",
)


@dataclass
class PredictorConfig:
    image_dim: int = 768
    text_dim: int = 768
    occasion_vocab: int = len(OCCASIONS)
    occasion_emb_dim: int = 32
    trunk_hidden: int = 512
    head_hidden: int = 128
    dropout: float = 0.1
    # Both default off: measured, both hurt. The idea was that a linear path
    # from input to output would make the network a superset of the ridge probe
    # that beats it, and that normalising the input would fix conditioning for
    # L2-normalised embeddings whose components sit near 0.04. Instead every
    # head fell — occasion_fit by 0.120 — and seed spread on that head went from
    # 0.015 to 0.090, which is optimisation instability, not a capacity gain. A
    # 1568-to-1 path straight into the logit is a large gradient route at
    # lr 1e-2, and LayerNorm across the concatenated vector mixes image, text
    # and occasion blocks of different scales.
    #
    # Kept switchable because the ablation is worth reporting.
    skip_connection: bool = False
    input_norm: bool = False
    # Per-dimension z-scoring using statistics from the training split, held as
    # a buffer so inference applies the same shift. This is not `input_norm`:
    # LayerNorm normalises each sample across a vector that concatenates image,
    # text and occasion blocks of different scales, which is what made it hurt.
    # Standardising per dimension against fixed training statistics leaves the
    # blocks alone and fixes conditioning, which is the part ridge gets for free
    # by choosing its penalty per head.
    standardise: bool = False
    head_names: tuple[str, ...] = field(default_factory=lambda: HEAD_NAMES)


class SaleabilityPredictor(nn.Module):
    """Multi-head saleability predictor.

    Forward inputs (all batched):
      image_emb:     (B, image_dim)   — CLIP image embedding (normalised)
      text_emb:      (B, text_dim)    — joint headline+inside-message text emb
      occasion_idx:  (B,)             — long, index into OCCASIONS

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
        )

        self.input_norm = (
            nn.LayerNorm(in_dim) if self.cfg.input_norm else nn.Identity()
        )

        # Buffers, not parameters: they travel with the state dict, so a loaded
        # checkpoint standardises exactly as it did in training, and the
        # optimiser never touches them. Identity until `set_feature_stats`
        # supplies the training statistics.
        #
        # Covers the image and text blocks only. The occasion block is a learned
        # embedding whose distribution moves during training, so statistics
        # frozen at step 0 would describe nothing and would fight the embedding.
        cached_dim = self.cfg.image_dim + self.cfg.text_dim
        self.register_buffer("feat_mean", torch.zeros(cached_dim))
        self.register_buffer("feat_std", torch.ones(cached_dim))

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

        self.skips = (
            nn.ModuleDict(
                {name: nn.Linear(in_dim, 1) for name in self.cfg.head_names}
            )
            if self.cfg.skip_connection
            else None
        )

    @torch.no_grad()
    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install training-split statistics for the cached feature blocks.

        Clamped because an embedding dimension that is constant across the
        corpus has zero variance, and dividing by it would put infinities into
        the first forward pass.
        """
        self.feat_mean.copy_(mean.to(self.feat_mean))
        self.feat_std.copy_(std.to(self.feat_std).clamp(min=1e-6))

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        occasion_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cached = torch.cat([image_emb, text_emb], dim=-1)
        if self.cfg.standardise:
            cached = (cached - self.feat_mean) / self.feat_std
        occ = self.occasion_emb(occasion_idx)
        x = self.input_norm(torch.cat([cached, occ], dim=-1))
        z = self.trunk(x)
        out = {}
        for name, head in self.heads.items():
            logit = head(z)
            if self.skips is not None:
                logit = logit + self.skips[name](x)
            out[name] = torch.sigmoid(logit).squeeze(-1)
        return out


def head_loss_weights(purchase_intent_factor: float = 2.0) -> dict[str, float]:
    """Weight the purchase-intent head above the rest.

    Every head now has the same number of labels, so this is no longer
    compensating for a smaller set. It reflects which head the pipeline
    actually uses: purchase intent ranks candidates at rerank time and orders
    condition D, while the other four are reported but never decide anything.
    """
    return {name: (purchase_intent_factor if name == "purchase_intent" else 1.0) for name in HEAD_NAMES}
