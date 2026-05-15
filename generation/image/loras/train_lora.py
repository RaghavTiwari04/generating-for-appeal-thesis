"""Per-occasion LoRA fine-tuning script (rank 8-16, ~1000 steps).

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

import json
from pathlib import Path

import typer
from PIL import Image

from common.db import engine
from common.logging import get_logger
from common.storage import get_object
from data.features.clip_embed import EMBED_DIM  # noqa: F401 — ensures shared imports work

log = get_logger(__name__)


_TOP_FOR_OCC_SQL = """
SELECT li.storage_path
FROM listings l
JOIN listing_features lf USING (listing_id)
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
JOIN saleability_labels sl
  ON sl.listing_id = l.listing_id AND sl.label_source = 'proxy_v1'
WHERE lf.occasion = %(occasion)s
ORDER BY sl.score DESC
LIMIT %(limit)s;
"""


def _materialise_training_images(occasion: str, limit: int, dest: Path) -> list[Path]:
    import pandas as pd

    df = pd.read_sql(_TOP_FOR_OCC_SQL, engine(), params={"occasion": occasion, "limit": limit})
    paths: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        try:
            data = get_object(row["storage_path"])
            img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
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
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    out_root: Path = Path(__file__).parent,
) -> None:
    """Train a single occasion LoRA.

    NOTE: this calls into diffusers' `train_dreambooth_lora_sdxl.py`-style
    training loop. The actual training code is deliberately delegated to the
    official diffusers example script for reproducibility — we drive it as a
    subprocess and only handle data prep + bookkeeping here.
    """
    import subprocess

    work_dir = Path("./artifacts/lora_train") / occasion.replace("/", "_")
    image_dir = work_dir / "images"
    out_dir = out_root / occasion.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    _materialise_training_images(occasion, n_images, image_dir)

    instance_prompt = f"a greeting card for {occasion.replace('_', ' ').replace('/', ' ')}"
    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=fp16",
        "diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py",
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
