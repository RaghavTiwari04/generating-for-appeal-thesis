"""Layout/inpainting ControlNet helpers.

We use a simple binary mask reserving a rectangular region for the headline
text. The diffusion model is conditioned (via ControlNet inpaint or via the
`image=` argument on an inpainting pipeline variant) to keep that region
low-detail, so the typography composer has clean whitespace to overlay text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageDraw


HeadlineRegion = Literal["top", "top-left", "top-right", "bottom", "centre"]


@dataclass
class LayoutMaskSpec:
    width: int = 1024
    height: int = 1024
    region: HeadlineRegion = "top-left"
    padding_frac: float = 0.06
    height_frac: float = 0.25     # vertical share reserved for headline
    width_frac: float = 0.55      # horizontal share reserved
    feather_px: int = 24


def build_headline_mask(spec: LayoutMaskSpec | None = None) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return (mask_image, bbox).

    Mask is white where the headline must remain low-detail, black elsewhere.
    Diffusers inpaint pipelines treat white = regenerate / keep blank.
    """
    spec = spec or LayoutMaskSpec()
    w, h = spec.width, spec.height
    pad = int(min(w, h) * spec.padding_frac)
    box_w = int(w * spec.width_frac)
    box_h = int(h * spec.height_frac)

    if spec.region == "top":
        x0, y0 = (w - box_w) // 2, pad
    elif spec.region == "top-left":
        x0, y0 = pad, pad
    elif spec.region == "top-right":
        x0, y0 = w - pad - box_w, pad
    elif spec.region == "bottom":
        x0, y0 = (w - box_w) // 2, h - pad - box_h
    elif spec.region == "centre":
        x0, y0 = (w - box_w) // 2, (h - box_h) // 2
    else:
        raise ValueError(f"Unknown region {spec.region!r}")

    x1, y1 = x0 + box_w, y0 + box_h
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=255)

    if spec.feather_px:
        from PIL import ImageFilter

        mask = mask.filter(ImageFilter.GaussianBlur(radius=spec.feather_px))

    return mask, (x0, y0, x1, y1)
