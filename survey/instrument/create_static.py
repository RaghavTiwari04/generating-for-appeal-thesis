"""Generate static assets needed by the survey app.

Run once at setup: python -m survey.instrument.create_static
"""
from pathlib import Path

from PIL import Image

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

# Placeholder card image (200×280 grey rectangle with centred text)
img = Image.new("RGB", (200, 280), color=(220, 220, 220))
from PIL import ImageDraw

draw = ImageDraw.Draw(img)
draw.rectangle([10, 10, 189, 269], outline=(180, 180, 180), width=2)
draw.text((60, 130), "Card image\nnot available", fill=(120, 120, 120))
img.save(STATIC / "placeholder.png")
print(f"Created {STATIC / 'placeholder.png'}")
