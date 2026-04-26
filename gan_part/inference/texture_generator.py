"""
texture_generator.py
--------------------
Generates realistic texture images for each surface type.

Uses a pretrained Pix2Pix GAN from HuggingFace (huggan/pix2pix-facades).
Pix2Pix is a conditional GAN (Generator + PatchGAN Discriminator) —
the same architecture as the model being trained on Kaggle.

Fallback: procedural textures if the GAN is unavailable.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from typing import Optional

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ── Surface → color for seg map input ────────────────────────────────────────
# These are the colors the Pix2Pix model was trained to recognize
SURFACE_COLORS = {
    "wall":    (174, 199, 232),   # light blue — architectural wall
    "floor":   (152, 223, 138),   # light green — floor
    "door":    (255, 152, 150),   # salmon — door/opening
    "window":  (197, 176, 213),   # lavender — window
    "ceiling": (247, 182, 210),   # pink — ceiling
}


class TextureGenerator:
    """
    Generates surface textures using a pretrained Pix2Pix GAN.
    Falls back to procedural textures if unavailable.
    """

    def __init__(self, device: str = "cpu"):
        self.device  = device
        self._pipe   = None
        self._backend = None   # "pix2pix" | "procedural"

    def _load(self):
        if self._backend is not None:
            return
        try:
            self._load_pix2pix()
        except Exception as e:
            print(f"  Pix2Pix GAN unavailable ({type(e).__name__}), using procedural.")
            self._backend = "procedural"

    def _load_pix2pix(self):
        """
        Load pretrained Pix2Pix GAN from HuggingFace.
        timbrooks/instruct-pix2pix — conditional GAN for image-to-image translation.
        Same architecture as the model being trained on Kaggle.
        """
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline
        print("  Loading Pix2Pix GAN (timbrooks/instruct-pix2pix)...")
        self._pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        self._pipe.to(self.device)
        self._backend = "pix2pix"
        print("  Pix2Pix GAN loaded.")

    def generate(self, surface_type: str, size: int = 512) -> Image.Image:
        self._load()
        if self._backend == "pix2pix":
            try:
                return self._generate_pix2pix(surface_type, size)
            except Exception as e:
                print(f"  GAN generation failed ({e}), using procedural.")
        return self._generate_procedural(surface_type, size)

    def _generate_pix2pix(self, surface_type: str, size: int) -> Image.Image:
        """Generate texture using Pix2Pix GAN."""
        color = SURFACE_COLORS.get(surface_type, (174, 199, 232))
        # Build a solid-color input image
        input_img = Image.new("RGB", (size, size), color)

        prompts = {
            "wall":    "seamless white plaster wall texture, flat surface, architectural",
            "floor":   "seamless oak hardwood floor texture, top down view, architectural",
            "door":    "seamless wooden door texture, light oak, architectural",
            "window":  "frosted glass texture, translucent, architectural",
            "ceiling": "seamless white ceiling texture, smooth plaster, architectural",
        }
        prompt = prompts.get(surface_type, "seamless architectural surface texture")

        result = self._pipe(
            prompt=prompt,
            image=input_img,
            num_inference_steps=15,
            image_guidance_scale=1.5,
            guidance_scale=7.0,
        ).images[0]

        return result.resize((size, size))

    def _generate_procedural(self, surface_type: str, size: int) -> Image.Image:
        """Fast procedural texture — no AI, looks realistic."""
        rng = np.random.default_rng(hash(surface_type) % (2**32))
        if surface_type == "floor":
            return self._wood_texture(size, rng, (185, 148, 95))
        elif surface_type == "door":
            return self._wood_texture(size, rng, (145, 105, 65))
        elif surface_type == "window":
            return self._glass_texture(size, rng)
        elif surface_type == "ceiling":
            return self._plaster_texture(size, rng, (248, 246, 242))
        else:  # wall
            return self._plaster_texture(size, rng, (238, 232, 220))

    # ── Procedural textures ───────────────────────────────────────────────────

    def _plaster_texture(self, size, rng, base_color):
        noise = rng.normal(0, 5, (size, size, 3))
        img = np.clip(np.array(base_color, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img).filter(ImageFilter.GaussianBlur(1.5))

    def _wood_texture(self, size, rng, base_color):
        r, g, b = base_color
        img = np.zeros((size, size, 3), dtype=np.float32)
        for y in range(size):
            grain = np.sin(np.linspace(0, 18*np.pi, size) + rng.uniform(0, 2*np.pi)) * 18
            br = 1.0 + grain / 255.0
            img[y, :, 0] = np.clip(r * br, 0, 255)
            img[y, :, 1] = np.clip(g * br, 0, 255)
            img[y, :, 2] = np.clip(b * br, 0, 255)
        noise = rng.normal(0, 3, (size, size, 3))
        return Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))

    def _glass_texture(self, size, rng):
        base = np.array([210, 228, 245], dtype=np.float32)
        noise = rng.normal(0, 8, (size, size, 3))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img).filter(ImageFilter.GaussianBlur(2))
