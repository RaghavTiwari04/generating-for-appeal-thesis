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
import re
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from common.db import engine
from common.logging import get_logger
from common.storage import get_object

log = get_logger(__name__)


# One image per duplicate cluster, and the cluster's best-scoring member stands
# for it.
#
# Print-on-demand catalogues carry the same design in many colourways: dedup
# found 1,111 redundant copies among 1,816 listings. Ordering by score without
# collapsing them puts near-identical copies next to each other — they score
# near-identically because they are the same artwork — so the top 150 could be
# a few dozen designs repeated, and the LoRA would overfit to those while the
# count suggested otherwise. This is the same collapse `data/labels/vlm_labels`
# applies before spending judge calls.
#
# The representative is always the labelled member. Labelling keeps the lowest
# listing_id per cluster, and unlabelled rows COALESCE to score 0, so the
# labelled one sorts first — and on a genuine 0.0 the listing_id tiebreak
# returns that same row. So the ranking below is over judge scores rather than
# over a mix of scored and unscored cards.
#
# Selection is stratified: each subtype contributes its own top slice rather
# than competing in one pool. birthday/general holds 2,011 of the group's 2,463
# distinct designs, so a single ranking would fill the training set with it and
# leave kids, milestone and relationship barely represented — and with the text
# encoder frozen, a prompt cannot recover a style the LoRA never saw. For a
# single subtype the partition has one group and this is an ordinary top-N.
_TOP_FOR_OCC_SQL = """
SELECT storage_path, extracted_text
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY occasion ORDER BY score DESC, storage_path
           ) AS rank_in_occasion
    FROM (
        SELECT DISTINCT ON (COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text))
               li.storage_path,
               lf.occasion,
               -- OCR of the card front. Captions name the words the card
               -- actually shows, so the lettering is conditioned on the prompt
               -- instead of being averaged into the style.
               lf.extracted_text,
               COALESCE(sl.score, 0) AS score
        FROM listings l
        JOIN listing_features lf USING (listing_id)
        JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
        LEFT JOIN saleability_labels sl
          ON sl.listing_id = l.listing_id AND sl.label_source = 'llm_ssr_rubric_v2'
        -- `birthday/kids` selects that subtype; `birthday` selects all of them,
        -- so one LoRA can be trained over a whole group. The LoRA learns style
        -- only — the text encoder is not trained — and the four birthday
        -- subtypes share one visual idiom, so splitting them fits nearly the
        -- same distribution four times from a quarter of the data each.
        WHERE (lf.occasion = %(occasion)s OR split_part(lf.occasion, '/', 1) = %(occasion)s)
        ORDER BY COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text),
                 COALESCE(sl.score, 0) DESC,
                 l.listing_id
    ) representatives
) ranked
WHERE rank_in_occasion <= %(per_occasion)s
ORDER BY score DESC;
"""

# The subtypes a group covers, so the per-subtype quota can be worked out
# before selecting.
_SUBTYPES_SQL = """
SELECT lf.occasion,
       COUNT(DISTINCT COALESCE(lf.duplicate_cluster_id::text, l.listing_id::text)) AS designs
FROM listings l
JOIN listing_features lf USING (listing_id)
JOIN listing_images li ON li.listing_id = l.listing_id AND li.is_primary
WHERE (lf.occasion = %(occasion)s OR split_part(lf.occasion, '/', 1) = %(occasion)s)
GROUP BY 1
ORDER BY 1;
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


# Longer than a headline is body copy or OCR noise, and quoting a paragraph in
# the caption would teach the model to fill the card with text.
_MAX_CAPTION_WORDS = 8


def _card_text(raw: object) -> str:
    """The card's own greeting, cleaned for use inside a caption.

    OCR returns the whole front, so this keeps the leading words — the headline
    is set largest and reads first — and drops anything that would turn the
    caption into a paragraph. Quotes are stripped because the caption wraps the
    result in its own.
    """
    if not isinstance(raw, str):
        return ""
    words = re.sub(r'["""\'\n\r]+', " ", raw).split()
    return " ".join(words[:_MAX_CAPTION_WORDS]).strip()


def _training_caption(blip_caption: str, card_text: str, occasion_tag: str) -> str:
    """The prompt one training image is paired with.

    When the card's own words are named, the lettering becomes something the
    caption varies rather than a constant the style averages over. Without it a
    LoRA trained on birthday cards learns "Happy Birthday" as part of the look,
    and then fights the brief's actual headline at generation time.

    Phrased to match `generation.image.headline_text.augment_prompt`, which
    asks for a greeting "lettered into the design" — training and inference
    then describe the same thing the same way.
    """
    if not blip_caption:
        return ""
    caption = f"TOK {blip_caption}, a greeting card for {occasion_tag}"
    if card_text:
        caption += f', with the greeting "{card_text}" lettered into the design'
    return caption


def _materialise_training_images(
    occasion: str, limit: int, dest: Path, *, erase_text: bool = False,
    resolution: int = 1024,
) -> list[tuple[Path, str]]:
    """Fetch training images. Returns (path, card text) so captions can quote it."""
    import pandas as pd

    subtypes = pd.read_sql(_SUBTYPES_SQL, engine(), params={"occasion": occasion})
    if subtypes.empty:
        log.warning(f"No listings for occasion '{occasion}'")
        return []

    # Floor, so the quota never overshoots `limit` in total and the outer query
    # needs no second trim — a trim by score would drop whole subtypes, which is
    # the imbalance the stratification exists to prevent.
    per_occasion = max(1, limit // len(subtypes))
    df = pd.read_sql(
        _TOP_FOR_OCC_SQL,
        engine(),
        params={"occasion": occasion, "per_occasion": per_occasion},
    )

    # Whether the quota selects or just takes everything. Where it meets the
    # pool the saleability ordering does no work and the LoRA trains on every
    # design including the ones the judge scored worst — worth seeing in the log
    # rather than inferring from the image count afterwards.
    if len(subtypes) > 1:
        log.info(f"{occasion}: {per_occasion} per subtype across {len(subtypes)} subtypes")
    for row in subtypes.itertuples():
        if per_occasion >= row.designs:
            log.warning(
                f"  {row.occasion}: {row.designs} distinct designs, quota {per_occasion} — "
                f"whole pool, so saleability ranking selects nothing here"
            )
        else:
            log.info(
                f"  {row.occasion}: top {per_occasion} of {row.designs} by saleability"
            )

    items: list[tuple[Path, str]] = []
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
            # Blank when the text was erased: quoting words the image no longer
            # shows would train the model to letter cards that have none.
            items.append((out, "" if erase_text else _card_text(row["extracted_text"])))
        except Exception as e:
            log.warning(f"Skipping {row['storage_path']}: {e}")

    with_text = sum(1 for _, t in items if t)
    log.info(
        f"Materialised {len(items)} training images to {dest} "
        f"({with_text} carrying readable card text)"
    )
    return items


def train(
    occasion: str = typer.Option(...),
    rank: int = 32,
    steps: int = 1000,
    lr: float = 1e-4,
    n_images: int = 150,
    resolution: int = 1024,
    erase_text: bool = typer.Option(
        False,
        help="Inpaint text regions out of training images (captions then omit the card's words)",
    ),
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

    items = _materialise_training_images(
        occasion, n_images, image_dir, erase_text=erase_text, resolution=resolution,
    )
    if not items:
        log.warning(f"No training images for occasion '{occasion}' — skipping LoRA training")
        return
    paths = [p for p, _ in items]
    card_texts = [t for _, t in items]

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
        quoted = 0
        with metadata.open("w", encoding="utf-8") as fh:
            # strict: a caption list shorter than the image list would silently
            # drop training images off the end of the metadata file.
            for path, cap, card_text in zip(paths, captions, card_texts, strict=True):
                text = _training_caption(cap, card_text, occasion_tag) or instance_prompt
                quoted += bool(cap and card_text)
                fh.write(json.dumps({"file_name": path.name, "prompt": text}) + "\n")
        log.info(f"Wrote per-image captions to {metadata} ({quoted} naming the card's own words)")
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
