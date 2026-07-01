"""Flux (default) / SDXL image generation with per-occasion LoRA stacking.

Two-pass pipeline:
  1. FluxPipeline generates full cover art (no mask).
  2. FluxFillPipeline inpaints the headline region to create clean
     whitespace for the typography composer.

LoRA weights are stored under `generation/image/loras/<occasion>/`; they are
trained by `generation/image/loras/train_lora.py` (separate script, GPU-only).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)


LORA_ROOT = Path(__file__).parent / "loras"


@dataclass
class DiffusionConfig:
    backend: str = settings.diffusion_backend
    model_id: str = settings.sdxl_model_id
    revision: str | None = settings.sdxl_revision
    flux_model_id: str = settings.flux_model_id
    flux_fill_model_id: str = settings.flux_fill_model_id
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    height: int = 1024
    width: int = 1024
    gen_steps: int = 28
    gen_guidance: float = 3.5
    lora_scale: float = 0.65
    fill_steps: int = 50
    fill_guidance: float = 30.0


class DiffusionRunner:
    """Lazily-initialised diffusion pipeline. Reuses one pipeline per process."""

    def __init__(self, cfg: DiffusionConfig | None = None):
        self.cfg = cfg or DiffusionConfig()
        self._pipe: Any = None
        self._fill_pipe: Any = None
        self._active_loras: list[str] = []

    def _sdxl_load_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"torch_dtype": self.cfg.dtype, "use_safetensors": True}
        if self.cfg.revision:
            kw["revision"] = self.cfg.revision
        return kw

    def _init_pipe(self, pipe: Any) -> Any:
        pipe = pipe.to(self.cfg.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _load_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        if self.cfg.backend == "sdxl":
            from diffusers import StableDiffusionXLPipeline

            log.info(f"Loading SDXL: {self.cfg.model_id}")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.model_id, **self._sdxl_load_kwargs(),
            )
        elif self.cfg.backend == "flux":
            from diffusers import FluxPipeline

            log.info(f"Loading Flux: {self.cfg.flux_model_id}")
            pipe = FluxPipeline.from_pretrained(
                self.cfg.flux_model_id,
                torch_dtype=self.cfg.dtype,
                token=settings.hf_token,
            )
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

        self._pipe = self._init_pipe(pipe)
        return self._pipe

    def _load_fill_pipeline(self) -> Any:
        """Load inpainting pipeline (Flux Fill or SDXL Inpaint)."""
        if self._fill_pipe is not None:
            return self._fill_pipe

        if self.cfg.backend == "flux":
            from diffusers import FluxFillPipeline

            log.info(f"Loading Flux Fill: {self.cfg.flux_fill_model_id}")
            pipe = FluxFillPipeline.from_pretrained(
                self.cfg.flux_fill_model_id,
                torch_dtype=self.cfg.dtype,
                token=settings.hf_token,
            )
        elif self.cfg.backend == "sdxl":
            from diffusers import StableDiffusionXLInpaintPipeline

            log.info(f"Loading SDXL Inpaint: {self.cfg.model_id}")
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.cfg.model_id, **self._sdxl_load_kwargs(),
            )
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

        self._fill_pipe = self._init_pipe(pipe)
        return self._fill_pipe

    def _apply_loras(self, pipe: Any, occasion: str | None) -> None:
        if not occasion:
            return
        lora_dir = LORA_ROOT / occasion.replace("/", "_")
        if not lora_dir.exists():
            log.debug(f"No LoRA for occasion={occasion}, skipping")
            return
        if str(lora_dir) in self._active_loras:
            return
        try:
            pipe.load_lora_weights(str(lora_dir))
            pipe.fuse_lora(lora_scale=self.cfg.lora_scale)
            self._active_loras.append(str(lora_dir))
            log.info(f"Loaded LoRA: {lora_dir.name} (scale={self.cfg.lora_scale})")
        except Exception as e:
            log.warning(f"LoRA load failed for {occasion}: {e}")

    def unload_loras(self) -> None:
        for pipe in (self._pipe, self._fill_pipe):
            if pipe is not None and self._active_loras:
                try:
                    pipe.unload_lora_weights()
                except Exception as e:
                    log.debug(f"LoRA unload failed (non-fatal): {e}")
        self._active_loras = []

    def _free_pipeline(self) -> None:
        """Unload gen pipeline to free VRAM."""
        if self._pipe is not None:
            self.unload_loras()
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Freed gen pipeline VRAM")

    def _free_fill_pipeline(self) -> None:
        """Unload Fill pipeline to free VRAM."""
        if self._fill_pipe is not None:
            del self._fill_pipe
            self._fill_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Freed Fill pipeline VRAM")

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        occasion: str | None = None,
        seed: int | None = None,
        n: int = 1,
        mask_image: Image.Image | None = None,
        upscale_to_print_res: bool = True,
        **kwargs: Any,
    ) -> list[Image.Image]:
        base_seed = seed if seed is not None else int(torch.randint(0, 2**31, (1,)).item())

        # Pass 1: generate all cover images with FluxPipeline
        covers: list[Image.Image] = []
        for i in range(n):
            gen = [torch.Generator(device=self.cfg.device).manual_seed(base_seed + i)]
            cover = self._generate_plain(
                prompt, negative_prompt, occasion, 1, gen, **kwargs,
            )[0]

            arr = np.array(cover)
            if arr.max() == 0:
                log.warning(f"Image {i + 1}/{n} is blank, retrying with offset seed")
                retry_gen = [torch.Generator(device=self.cfg.device).manual_seed(base_seed + n + i)]
                cover = self._generate_plain(
                    prompt, negative_prompt, occasion, 1, retry_gen, **kwargs,
                )[0]

            covers.append(cover)
            log.info(f"Generated image {i + 1}/{n}")

        # Pass 2: inpaint headline regions with FluxFillPipeline
        if mask_image is not None:
            self._free_pipeline()
            for i, cover in enumerate(covers):
                covers[i] = self._inpaint_headline_region(cover, mask_image, prompt, base_seed + i)
                log.info(f"Inpainted headline {i + 1}/{n}")
            self._free_fill_pipeline()

        if upscale_to_print_res:
            from generation.image.upscaler import upscale_to_print
            covers = [upscale_to_print(img) for img in covers]

        return covers

    def _generate_plain(
        self, prompt: str, negative_prompt: str, occasion: str | None,
        n: int, generator: Any, **kwargs: Any,
    ) -> list[Image.Image]:
        pipe = self._load_pipeline()
        self._apply_loras(pipe, occasion)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "num_inference_steps": self.cfg.gen_steps,
            "guidance_scale": self.cfg.gen_guidance,
            "num_images_per_prompt": n,
            "generator": generator,
        }
        if self.cfg.backend == "sdxl" and negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        call_kwargs.update(kwargs)

        result = pipe(**call_kwargs)
        return list(result.images)

    def _inpaint_headline_region(
        self, cover: Image.Image, mask: Image.Image, prompt: str, seed: int,
    ) -> Image.Image:
        """Inpaint the headline region of an already-generated cover to create clean whitespace."""
        pipe = self._load_fill_pipeline()
        gen = [torch.Generator(device=self.cfg.device).manual_seed(seed)]

        from PIL import ImageOps
        fill_mask = ImageOps.invert(mask.convert("L"))

        inpaint_prompt = (
            "clean smooth empty whitespace area suitable for text overlay, "
            "no objects no details, matching the surrounding card style"
        )

        call_kwargs: dict[str, Any] = {
            "prompt": inpaint_prompt,
            "image": cover,
            "mask_image": fill_mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "num_inference_steps": self.cfg.fill_steps,
            "guidance_scale": self.cfg.fill_guidance,
            "num_images_per_prompt": 1,
            "generator": gen,
        }

        result = pipe(**call_kwargs)
        return result.images[0]


_runner: DiffusionRunner | None = None


def get_runner() -> DiffusionRunner:
    global _runner
    if _runner is None:
        _runner = DiffusionRunner()
    return _runner
