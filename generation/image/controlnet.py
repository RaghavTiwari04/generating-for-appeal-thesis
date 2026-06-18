"""Backward-compat shim — moved to headline_mask.py.

Deprecated: import from generation.image.headline_mask directly.
"""
from generation.image.headline_mask import (  # noqa: F401
    HeadlineRegion,
    LayoutMaskSpec,
    build_headline_mask,
)
