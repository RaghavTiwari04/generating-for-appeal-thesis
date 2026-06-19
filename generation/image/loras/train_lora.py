"""Per-occasion LoRA fine-tuning on Flux.1-dev (rank 8-16, ~1000 steps).

Trains a small LoRA on the top-saleability images for a single occasion.
PEFT-based; meant to run on a rented A100 in a training sprint. Saves LoRA
weights to `generation/image/loras/<occasion>/`.

This is a CLI wrapper around diffusers + peft. We keep training-script
configuration explicit in the CLI so each occasion's LoRA is reproducible
from the command line that produced it.

Usage (rented A100):
    python -m generation.image.loras.train_lora \
        --occasion birthday/general \
        --rank 8 \
        --steps 1000 \
        --lr 1e-4
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from common.db import engine
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)


_TOP_FOR_OCC_SQL = """
SELECT li.storage_path
FROM listings l
JOIN listing_features lf USING (listing_id)
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
LEFT JOIN saleability_labels sl
  ON sl.listing_id = l.listing_id AND sl.label_source = 'vlm_4head_v1'
WHERE lf.occasion = %(occasion)s
ORDER BY COALESCE(sl.score, 0) DESC
LIMIT %(limit)s;
"""


def _erase_text_regions(img: Image.Image) -> Image.Image:
    """Detect text bounding boxes via Tesseract and inpaint them out.

    Builds a binary mask from OCR word-level bboxes (dilated slightly to cover
    serifs/shadows), then uses OpenCV Telea inpainting to fill those regions
    with surrounding texture. This prevents LoRA from learning text-as-texture
    artifacts from marketplace card images.
    """
    import cv2
    import pytesseract

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for i, conf in enumerate(ocr_data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 10:
            continue
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        if w < 5 or h < 5:
            continue
        pad = max(4, int(0.15 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_cv.shape[1], x + w + pad)
        y1 = min(img_cv.shape[0], y + h + pad)
        mask[y0:y1, x0:x1] = 255

    if mask.sum() == 0:
        return img

    inpainted = cv2.inpaint(img_cv, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def _materialise_training_images(
    occasion: str, limit: int, dest: Path, *, erase_text: bool = True,
) -> list[Path]:
    import pandas as pd

    df = pd.read_sql(_TOP_FOR_OCC_SQL, engine(), params={"occasion": occasion, "limit": limit})
    paths: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        try:
            data = get_object(row["storage_path"])
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if erase_text:
                try:
                    img = _erase_text_regions(img)
                except Exception as te:
                    log.debug(f"Text erasure skipped (tesseract unavailable): {te}")
            out = dest / f"{i:04d}.png"
            img.save(out)
            paths.append(out)
        except Exception as e:
            log.warning(f"Skipping {row['storage_path']}: {e}")
    log.info(f"Materialised {len(paths)} training images to {dest}")
    return paths


def train(
    occasion: str = typer.Option(...),
    rank: int = 8,
    steps: int = 1000,
    lr: float = 1e-4,
    n_images: int = 150,
    erase_text: bool = typer.Option(True, help="Inpaint text regions out of training images"),
    base_model: str = "black-forest-labs/FLUX.1-dev",
    out_root: Path = Path(__file__).parent,
) -> None:
    """Train a single occasion LoRA on Flux.1-dev via DreamBooth.

    Delegates to diffusers' official train_dreambooth_lora_flux.py for
    reproducibility — we handle data prep + bookkeeping here.
    """
    import subprocess

    work_dir = Path("./artifacts/lora_train") / occasion.replace("/", "_")
    image_dir = work_dir / "images"
    out_dir = out_root / occasion.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    _materialise_training_images(occasion, n_images, image_dir, erase_text=erase_text)

    train_script = Path("diffusers/examples/dreambooth/train_dreambooth_lora_flux.py")
    if not train_script.exists():
        train_script.parent.mkdir(parents=True, exist_ok=True)
        import diffusers
        _ver = diffusers.__version__
        _url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{_ver}/examples/dreambooth/train_dreambooth_lora_flux.py"
        import urllib.request
        urllib.request.urlretrieve(_url, train_script)
        log.info(f"Downloaded training script from {_url}")

    instance_prompt = f"a greeting card for {occasion.replace('_', ' ').replace('/', ' ')}"
    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        str(train_script),
        f"--pretrained_model_name_or_path={base_model}",
        f"--instance_data_dir={image_dir}",
        f"--output_dir={out_dir}",
        f"--instance_prompt={instance_prompt}",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        f"--learning_rate={lr}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={steps}",
        f"--rank={rank}",
        "--checkpointing_steps=500",
        "--seed=42",
    ]
    log.info(f"Launching LoRA training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "occasion": occasion,
                "rank": rank,
                "steps": steps,
                "lr": lr,
                "n_images": n_images,
                "base_model": base_model,
            },
            indent=2,
        )
    )
    log.info(f"LoRA saved to {out_dir}")


if __name__ == "__main__":
    typer.run(train)
