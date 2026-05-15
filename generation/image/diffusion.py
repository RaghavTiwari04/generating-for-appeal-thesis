"""SDXL / Flux image generation with per-occasion LoRA stacking.

Self-hosted via diffusers. ControlNet conditioning is applied through
`generation/image/controlnet.py`'s helper that supplies an inpainting mask
reserving the headline area.

LoRA weights are stored under `generation/image/loras/<occasion>/`; they are
trained by `generation/image/loras/train_lora.py` (separate script, GPU-only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.0


class DiffusionRunner:
    """Lazily-initialised diffusion pipeline. Reuses one pipeline per process."""

    def __init__(self, cfg: DiffusionConfig | None = None):
        self.cfg = cfg or DiffusionConfig()
        self._pipe: Any = None
        self._active_loras: list[str] = []

    def _load_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        if self.cfg.backend == "sdxl":
            from diffusers import StableDiffusionXLPipeline

            log.info(f"Loading SDXL: {self.cfg.model_id} revision={self.cfg.revision}")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.model_id,
                revision=self.cfg.revision,
                torch_dtype=self.cfg.dtype,
                use_safetensors=True,
            )
        elif self.cfg.backend == "flux":
            from diffusers import FluxPipeline

            log.info(f"Loading Flux: {settings.flux_model_id}")
            pipe = FluxPipeline.from_pretrained(
                settings.flux_model_id, torch_dtype=self.cfg.dtype
            )
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

        pipe = pipe.to(self.cfg.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _apply_loras(self, occasion: str | None) -> None:
        if not occasion:
            return
        lora_dir = LORA_ROOT / occasion.replace("/", "_")
        if not lora_dir.exists():
            log.debug(f"No LoRA for occasion={occasion}, skipping")
            return
        if str(lora_dir) in self._active_loras:
            return
        try:
            self._pipe.load_lora_weights(str(lora_dir))
            self._active_loras.append(str(lora_dir))
            log.info(f"Loaded LoRA: {lora_dir.name}")
        except Exception as e:
            log.warning(f"LoRA load failed for {occasion}: {e}")

    def unload_loras(self) -> None:
        if self._pipe is not None and self._active_loras:
            try:
                self._pipe.unload_lora_weights()
            except Exception:
                pass
        self._active_loras = []

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        occasion: str | None = None,
        seed: int | None = None,
        n: int = 1,
        controlnet_image: Image.Image | None = None,
        upscale_to_print_res: bool = True,
        **kwargs: Any,
    ) -> list[Image.Image]:
        pipe = self._load_pipeline()
        self._apply_loras(occasion)

        generator = None
        if seed is not None:
            generator = [
                torch.Generator(device=self.cfg.device).manual_seed(seed + i) for i in range(n)
            ]

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "num_inference_steps": self.cfg.num_inference_steps,
            "guidance_scale": self.cfg.guidance_scale,
            "num_images_per_prompt": n,
            "generator": generator,
        }
        if controlnet_image is not None:
            call_kwargs["image"] = controlnet_image
        call_kwargs.update(kwargs)

        result = pipe(**call_kwargs)
        images = list(result.images)

        if upscale_to_print_res:
            from generation.image.upscaler import upscale_to_print
            images = [upscale_to_print(img) for img in images]

        return images


_runner: DiffusionRunner | None = None


def get_runner() -> DiffusionRunner:
    global _runner
    if _runner is None:
        _runner = DiffusionRunner()
    return _runner
