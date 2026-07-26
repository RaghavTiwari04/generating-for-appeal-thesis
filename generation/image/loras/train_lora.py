"""Per-occasion LoRA fine-tuning on Flux.1-dev.

Trains a style LoRA on the top-saleability images for a single occasion and
saves the weights to `generation/image/loras/<occasion>/`.

Thin CLI wrapper around diffusers' official `train_dreambooth_lora_flux.py`
(flow-matching loss, timestep sampling and weight serialisation all come from
there). We own only data prep and bookkeeping, so each occasion's LoRA stays
reproducible from the command line that produced it.

Note: text encoders are NOT trained (no `--train_text_encoder`), so "TOK" is
not a learned token — it is a consistent conditioning phrase that the
transformer LoRA keys on. It is kept identical between training captions and
inference prompts so that association holds.

Usage — mirrors cluster/jobs/04_train_lora.sh:
    python -m generation.image.loras.train_lora \
        --occasion birthday/general \
        --rank 32 \
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
  ON sl.listing_id = l.listing_id AND sl.label_source = 'llm_ssr_rubric_v1'
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


def _pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """Fit the whole card into a square canvas without cropping.

    The diffusers dreambooth script resizes to `--resolution` and then crops
    square. Greeting cards are portrait, so that crop discards the top and
    bottom — exactly the border and headline regions that define card
    composition. Pre-padding to square makes that crop a no-op.

    Padding uses the median colour of the image border, which for card art is
    usually the background itself, so the LoRA is not taught hard letterbox
    edges as a feature.
    """
    w, h = img.size
    if w == h == size:
        return img
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)

    arr = np.array(img)
    edges = np.concatenate(
        [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
    )
    bg = tuple(int(v) for v in np.median(edges, axis=0))

    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _caption_images(paths: list[Path]) -> list[str] | None:
    """BLIP-caption each training image; None if captioning is unavailable.

    Training every image on one fixed caption teaches the LoRA the *average*
    of the set and weakens prompt conditioning. Per-image captions keep the
    text-to-image association intact so detailed inference prompts still steer
    the output.
    """
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except Exception as e:
        log.warning(f"Captioning unavailable ({e}), falling back to a fixed instance prompt")
        return None

    model_id = "Salesforce/blip-image-captioning-base"
    try:
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
    except Exception as e:
        log.warning(f"Could not load {model_id} ({e}), falling back to a fixed instance prompt")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    captions: list[str] = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40)
            captions.append(processor.decode(out[0], skip_special_tokens=True).strip())
        except Exception as e:
            log.warning(f"Captioning failed for {p.name} ({e}), using empty caption")
            captions.append("")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    log.info(f"Captioned {len(captions)} training images")
    return captions


def _materialise_training_images(
    occasion: str, limit: int, dest: Path, *, erase_text: bool = True,
    resolution: int = 1024,
) -> list[Path]:
    import pandas as pd

    df = pd.read_sql(_TOP_FOR_OCC_SQL, engine(), params={"occasion": occasion, "limit": limit})
    paths: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    # Clear prior runs. `--instance_data_dir` opens every file in the directory
    # as an image, so a leftover metadata.jsonl would crash training, and stale
    # PNGs from a different --resolution would silently join the training set.
    for stale in dest.iterdir():
        if stale.is_file():
            stale.unlink()
    for i, row in df.iterrows():
        try:
            data = get_object(row["storage_path"])
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if erase_text:
                try:
                    img = _erase_text_regions(img)
                except Exception as te:
                    log.debug(f"Text erasure skipped (tesseract unavailable): {te}")
            img = _pad_to_square(img, resolution)
            out = dest / f"{i:04d}.png"
            img.save(out)
            paths.append(out)
        except Exception as e:
            log.warning(f"Skipping {row['storage_path']}: {e}")
    log.info(f"Materialised {len(paths)} training images to {dest}")
    return paths


def train(
    occasion: str = typer.Option(...),
    rank: int = 32,
    steps: int = 1000,
    lr: float = 1e-4,
    n_images: int = 150,
    resolution: int = 1024,
    erase_text: bool = typer.Option(True, help="Inpaint text regions out of training images"),
    caption_images: bool = typer.Option(True, help="BLIP-caption each image instead of one fixed prompt"),
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

    paths = _materialise_training_images(
        occasion, n_images, image_dir, erase_text=erase_text, resolution=resolution,
    )
    if not paths:
        log.warning(f"No training images for occasion '{occasion}' — skipping LoRA training")
        return

    train_script = Path("diffusers/examples/dreambooth/train_dreambooth_lora_flux.py")
    if not train_script.exists():
        train_script.parent.mkdir(parents=True, exist_ok=True)
        import diffusers
        _ver = diffusers.__version__
        _url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{_ver}/examples/dreambooth/train_dreambooth_lora_flux.py"
        import urllib.request
        urllib.request.urlretrieve(_url, train_script)
        log.info(f"Downloaded training script from {_url}")

    occasion_tag = occasion.replace("_", " ").replace("/", " ")
    instance_prompt = f"a TOK greeting card for {occasion_tag}"
    warmup_steps = max(1, steps // 10)

    # Per-image captions go through the dataset path; instance_data_dir applies
    # one fixed prompt to every image and cannot carry them.
    captions = _caption_images(paths) if caption_images else None
    if captions:
        metadata = image_dir / "metadata.jsonl"
        with metadata.open("w", encoding="utf-8") as fh:
            for path, cap in zip(paths, captions):
                text = f"TOK {cap}, a greeting card for {occasion_tag}" if cap else instance_prompt
                fh.write(json.dumps({"file_name": path.name, "prompt": text}) + "\n")
        log.info(f"Wrote per-image captions to {metadata}")
        data_args = [
            f"--dataset_name={image_dir}",
            "--image_column=image",
            "--caption_column=prompt",
        ]
    else:
        data_args = [f"--instance_data_dir={image_dir}"]

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        str(train_script),
        f"--pretrained_model_name_or_path={base_model}",
        *data_args,
        f"--output_dir={out_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--resolution={resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        f"--learning_rate={lr}",
        "--lr_scheduler=cosine",
        f"--lr_warmup_steps={warmup_steps}",
        f"--max_train_steps={steps}",
        f"--rank={rank}",
        "--checkpointing_steps=250",
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
                "resolution": resolution,
                "captioned": bool(captions),
                "base_model": base_model,
            },
            indent=2,
        )
    )
    log.info(f"LoRA saved to {out_dir}")


if __name__ == "__main__":
    typer.run(train)
