"""Upscale 1024×1024 diffusion output to print resolution.

Target: 1240×1748 px (A6 card at 300 DPI).

Two backends:
1. Real-ESRGAN (preferred — sharp, handles card art well)
2. Lanczos fallback (PIL — no extra deps, lower quality but always works)

The upscaler is a post-processing step applied after layout composition,
producing the final PNG ready for print-on-demand upload.

Usage:
    img = upscale_to_print(cover_1024)        # returns PIL Image
    img.save("card_print.png", dpi=(300, 300))
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

from PIL import Image

from common.logging import get_logger

log = get_logger(__name__)

PRINT_W = 1240
PRINT_H = 1748
PRINT_DPI = 300

Backend = Literal["realesrgan", "lanczos"]


def _upscale_realesrgan(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Use Real-ESRGAN ×4 then resize to exact target."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        scale = 4
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )
        model_path = Path(__file__).parent / "weights" / "RealESRGAN_x4plus.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Real-ESRGAN weights not found at {model_path}. "
                                    "Download from https://github.com/xinntao/Real-ESRGAN/releases")

        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
        import numpy as np
        import cv2
        arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        out_arr, _ = upsampler.enhance(arr, outscale=scale)
        out_rgb = cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB)
        upscaled = Image.fromarray(out_rgb)
        return upscaled.resize((target_w, target_h), Image.LANCZOS)

    except ImportError:
        raise ImportError(
            "realesrgan / basicsr not installed. "
            "pip install realesrgan  or use backend='lanczos'."
        )


def _upscale_lanczos(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return img.resize((target_w, target_h), Image.LANCZOS)


def upscale_to_print(
    img: Image.Image,
    *,
    backend: Backend | None = None,
    target_w: int = PRINT_W,
    target_h: int = PRINT_H,
) -> Image.Image:
    """Upscale `img` to print resolution using the best available backend.

    Falls back to Lanczos if Real-ESRGAN is unavailable (weights missing or
    package not installed) so the pipeline never hard-fails.
    """
    if backend is None:
        backend = "realesrgan"

    if backend == "realesrgan":
        try:
            return _upscale_realesrgan(img, target_w, target_h)
        except (ImportError, FileNotFoundError) as e:
            log.warning(f"Real-ESRGAN unavailable ({e}), falling back to Lanczos")
            return _upscale_lanczos(img, target_w, target_h)

    return _upscale_lanczos(img, target_w, target_h)


def download_realesrgan_weights(out_dir: Path | None = None) -> Path:
    """Download Real-ESRGAN x4plus weights (~67 MB)."""
    import httpx

    out_dir = out_dir or Path(__file__).parent / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    )
    dest = out_dir / "RealESRGAN_x4plus.pth"
    if dest.exists():
        log.info(f"Weights already present at {dest}")
        return dest
    log.info(f"Downloading Real-ESRGAN weights → {dest}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.0f}%", end="", flush=True)
    print()
    log.info(f"Saved {dest}")
    return dest


if __name__ == "__main__":
    import typer

    def cli(
        input_path: Path = typer.Argument(...),
        output_path: Path = typer.Argument(...),
        backend: Backend = "realesrgan",
    ) -> None:
        img = Image.open(input_path).convert("RGB")
        out = upscale_to_print(img, backend=backend)
        out.save(output_path, dpi=(PRINT_DPI, PRINT_DPI))
        print(f"Saved {output_path} ({out.width}×{out.height} @ {PRINT_DPI} DPI)")

    typer.run(cli)
