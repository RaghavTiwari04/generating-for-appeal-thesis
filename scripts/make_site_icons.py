"""Draw the site's icons and its social preview image.

Run by hand when the mark or the hero card changes, and commit what it writes.
Like `prepare_site_assets`, this is content production rather than a build
step: the site has no build, and these files are small enough to live in the
repository.

    python -m scripts.make_site_icons

The mark is two cards, the back one tilted the same way the hero card on the
page is. At sixteen pixels the tilt is most of what survives, which is enough
to read as cards rather than as a generic square.

Everything is drawn from the same palette as `site/css/site.css`. The SVG is
authored here rather than rasterised from a file so there is no dependency on
a separate SVG renderer, and the raster sizes redraw the identical geometry
with Pillow.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

SITE = pathlib.Path("site")
FONT = pathlib.Path("generation/layout/fonts/google/PlayfairDisplay-Bold.ttf")

PAPER = "#f8f5ef"
PAPER_RAISED = "#fdfbf7"
INK = "#17140f"
INK_SOFT = "#4f4941"
ACCENT = "#2e4a55"
ACCENT_BRIGHT = "#3d6070"
RULE = "#ddd5c7"

FAVICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="10" fill="{PAPER}"/>
  <g transform="rotate(-9 32 34)">
    <rect x="14" y="13" width="30" height="42" rx="3" fill="{ACCENT_BRIGHT}"/>
  </g>
  <rect x="22" y="15" width="30" height="42" rx="3" fill="{ACCENT}"/>
  <rect x="27" y="24" width="20" height="3.4" rx="1.7" fill="{PAPER}"/>
  <rect x="27" y="32" width="14" height="3.4" rx="1.7" fill="{PAPER}" opacity="0.72"/>
</svg>
"""


def _draw_mark(size: int) -> Image.Image:
    """The favicon geometry, drawn at any size. 64 units in the SVG map to `size`."""
    scale = size * 4 / 64  # supersample, then reduce, so the edges are clean
    big = int(size * 4)
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def box(x, y, w, h, r, fill):
        d.rounded_rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale],
                            radius=r * scale, fill=fill)

    box(0, 0, 64, 64, 10, PAPER)

    back = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(back).rounded_rectangle(
        [14 * scale, 13 * scale, 44 * scale, 55 * scale], radius=3 * scale, fill=ACCENT_BRIGHT
    )
    back = back.rotate(9, resample=Image.BICUBIC, center=(32 * scale, 34 * scale))
    im.alpha_composite(back)

    box(22, 15, 30, 42, 3, ACCENT)
    box(27, 24, 20, 3.4, 1.7, PAPER)
    box(27, 32, 14, 3.4, 1.7, "#cfd8dc")

    return im.resize((size, size), Image.LANCZOS)


def _fit(draw, text: str, font_path: pathlib.Path, target_w: int, start: int) -> ImageFont.FreeTypeFont:
    """Largest size at or below `start` that keeps `text` inside `target_w`."""
    size = start
    while size > 12:
        font = ImageFont.truetype(str(font_path), size)
        if draw.textlength(text, font=font) <= target_w:
            return font
        size -= 2
    return ImageFont.truetype(str(font_path), 12)


def build_og() -> None:
    """The 1200x630 card that link previews show.

    Portrait card on the right, the question on the left, because the hero card
    is 1240x1748 and dropping it in whole would letterbox badly.
    """
    W, H = 1200, 630
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)

    card_src = SITE / "assets" / "cards" / "hero" / "hero-01-1240.jpg"
    if card_src.exists():
        card = Image.open(card_src).convert("RGB")
        target_h = 470
        card = card.resize((round(card.width * target_h / card.height), target_h), Image.LANCZOS)
        card = card.rotate(-2.2, resample=Image.BICUBIC, expand=True, fillcolor=PAPER)
        cx = W - card.width - 90
        cy = (H - card.height) // 2
        d.rectangle([cx + 8, cy + 12, cx + card.width + 8, cy + card.height + 12], fill=RULE)
        im.paste(card, (cx, cy))
        text_w = cx - 150
    else:
        text_w = W - 200

    if FONT.exists():
        title = "Can a model learn what makes a birthday card worth choosing?"
        words, lines, cur = title.split(), [], ""
        font = _fit(d, "Can a model learn what", FONT, text_w, 58)
        for w in words:
            trial = f"{cur} {w}".strip()
            if d.textlength(trial, font=font) <= text_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)

        y = (H - len(lines) * (font.size + 14)) // 2
        for line in lines:
            d.text((90, y), line, font=font, fill=INK)
            y += font.size + 14

        small = ImageFont.truetype(str(FONT), 25)
        d.text((90, y + 18), "An MSc thesis demo", font=small, fill=INK_SOFT)

    im.save(SITE / "og.jpg", quality=88, optimize=True)
    print("wrote site/og.jpg")


def main() -> None:
    (SITE / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")
    print("wrote site/favicon.svg")

    # .ico for the browsers that still ask for /favicon.ico by name, and a
    # 180px PNG because iOS ignores both of the above.
    _draw_mark(64).save(SITE / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print("wrote site/favicon.ico")

    apple = Image.new("RGB", (180, 180), PAPER)
    apple.paste(_draw_mark(180), (0, 0), _draw_mark(180))
    apple.save(SITE / "apple-touch-icon.png", optimize=True)
    print("wrote site/apple-touch-icon.png")

    build_og()


if __name__ == "__main__":
    main()
