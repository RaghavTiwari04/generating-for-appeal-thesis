"""Rule-based mapping (tone, style_tags) -> ordered font candidates.

Fonts are referenced by family name and a relative path under `fonts/`. We
ship only license-clean fonts (Google Fonts default) and a small set of paid
faces that must be installed under `fonts/paid/` separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FontSpec:
    family: str
    path: str
    weight: str = "Regular"


# Google Fonts (SIL OFL — commercial use allowed)
FONTS: dict[str, FontSpec] = {
    "caveat": FontSpec("Caveat", "fonts/google/Caveat-Bold.ttf", "Bold"),
    "sacramento": FontSpec("Sacramento", "fonts/google/Sacramento-Regular.ttf"),
    "cormorant": FontSpec("Cormorant Garamond", "fonts/google/CormorantGaramond-SemiBold.ttf"),
    "spectral": FontSpec("Spectral", "fonts/google/Spectral-SemiBold.ttf"),
    "bebas": FontSpec("Bebas Neue", "fonts/google/BebasNeue-Regular.ttf"),
    "anton": FontSpec("Anton", "fonts/google/Anton-Regular.ttf"),
    "playfair": FontSpec("Playfair Display", "fonts/google/PlayfairDisplay-Bold.ttf"),
    "lora": FontSpec("Lora", "fonts/google/Lora-SemiBold.ttf"),
    "rubik": FontSpec("Rubik", "fonts/google/Rubik-SemiBold.ttf"),
    "amatic": FontSpec("Amatic SC", "fonts/google/AmaticSC-Bold.ttf"),
    "permanent_marker": FontSpec("Permanent Marker", "fonts/google/PermanentMarker-Regular.ttf"),
    "great_vibes": FontSpec("Great Vibes", "fonts/google/GreatVibes-Regular.ttf"),
    "dancing_script": FontSpec("Dancing Script", "fonts/google/DancingScript-Bold.ttf"),
    "alfa_slab_one": FontSpec("Alfa Slab One", "fonts/google/AlfaSlabOne-Regular.ttf"),
    "righteous": FontSpec("Righteous", "fonts/google/Righteous-Regular.ttf"),
}


# (tone, style_tag) -> ordered list of font keys (best fit first).
# style_tag vocabulary (from brief_v1.txt): watercolour, illustrated,
# photographic, typographic, bold-graphic, minimalist, hand-drawn, retro,
# modern-serif. An empty style_tag "" is the per-tone fallback used by
# select_fonts when no tag-specific rule matches — every tone has one, so a
# card never silently drops to the global _DEFAULT.
RULES: dict[tuple[str, str], tuple[str, ...]] = {
    # -- warm-sincere (the eval default tone) — full tag coverage --
    ("warm-sincere", "watercolour"): ("great_vibes", "lora", "cormorant"),
    ("warm-sincere", "illustrated"): ("lora", "cormorant", "dancing_script"),
    ("warm-sincere", "photographic"): ("lora", "spectral", "cormorant"),
    ("warm-sincere", "typographic"): ("playfair", "cormorant", "rubik"),
    ("warm-sincere", "bold-graphic"): ("anton", "bebas", "playfair"),
    ("warm-sincere", "minimalist"): ("cormorant", "spectral", "lora"),
    ("warm-sincere", "hand-drawn"): ("caveat", "dancing_script", "great_vibes"),
    ("warm-sincere", "retro"): ("righteous", "rubik", "lora"),
    ("warm-sincere", "modern-serif"): ("playfair", "cormorant", "spectral"),
    ("warm-sincere", ""): ("great_vibes", "lora", "cormorant"),
    # -- warm-humorous --
    ("warm-humorous", "watercolour"): ("caveat", "sacramento", "amatic"),
    ("warm-humorous", "illustrated"): ("caveat", "dancing_script", "amatic"),
    ("warm-humorous", "hand-drawn"): ("caveat", "amatic", "permanent_marker"),
    ("warm-humorous", "bold-graphic"): ("permanent_marker", "anton", "righteous"),
    ("warm-humorous", "typographic"): ("righteous", "anton", "rubik"),
    ("warm-humorous", ""): ("caveat", "dancing_script", "amatic"),
    # -- funny-irreverent --
    ("funny-irreverent", "bold-graphic"): ("bebas", "anton", "alfa_slab_one"),
    ("funny-irreverent", "typographic"): ("anton", "alfa_slab_one", "righteous"),
    ("funny-irreverent", "hand-drawn"): ("permanent_marker", "amatic", "caveat"),
    ("funny-irreverent", "retro"): ("righteous", "alfa_slab_one", "bebas"),
    ("funny-irreverent", ""): ("anton", "bebas", "permanent_marker"),
    # -- formal-sincere --
    ("formal-sincere", "minimalist"): ("cormorant", "playfair", "spectral"),
    ("formal-sincere", "modern-serif"): ("playfair", "cormorant", "spectral"),
    ("formal-sincere", "typographic"): ("playfair", "spectral", "rubik"),
    ("formal-sincere", ""): ("playfair", "cormorant", "spectral"),
    # -- minimalist --
    ("minimalist", "minimalist"): ("rubik", "spectral", "lora"),
    ("minimalist", "modern-serif"): ("spectral", "lora", "playfair"),
    ("minimalist", "typographic"): ("rubik", "bebas", "spectral"),
    ("minimalist", ""): ("rubik", "spectral", "lora"),
    # -- religious --
    ("religious", "minimalist"): ("cormorant", "playfair", "spectral"),
    ("religious", "modern-serif"): ("playfair", "cormorant", "spectral"),
    ("religious", ""): ("cormorant", "playfair", "spectral"),
    # -- sentimental --
    ("sentimental", "hand-drawn"): ("dancing_script", "great_vibes", "caveat"),
    ("sentimental", "watercolour"): ("great_vibes", "dancing_script", "lora"),
    ("sentimental", "illustrated"): ("dancing_script", "lora", "cormorant"),
    ("sentimental", ""): ("dancing_script", "great_vibes", "caveat"),
}


_DEFAULT = ("rubik", "lora", "spectral")


def select_fonts(tone: str, style_tags: list[str]) -> tuple[FontSpec, ...]:
    """Return up to 3 candidate FontSpecs ordered best-first."""
    keys: list[str] = []
    for tag in style_tags:
        keys.extend(RULES.get((tone, tag), ()))
    keys.extend(RULES.get((tone, ""), ()))
    if not keys:
        keys = list(_DEFAULT)
    seen: set[str] = set()
    out: list[FontSpec] = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        if k in FONTS:
            out.append(FONTS[k])
        if len(out) >= 3:
            break
    return tuple(out)
