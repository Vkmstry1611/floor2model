"""
texture_generator.py
--------------------
Generates realistic texture images for each surface type
using ControlNet (or Pix2Pix GAN when trained).

For each surface type (wall, floor, door, window) it produces
a tileable 512x512 texture image that gets applied to the
corresponding faces in the 3D model.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Optional


# ── Prompts per surface type ──────────────────────────────────────────────────

SURFACE_PROMPTS = {
    "wall": (
        "seamless tileable white plaster wall texture, smooth surface, "
        "architectural interior, high quality, 4k, no furniture, flat surface"
    ),
    "floor": (
        "seamless tileable light oak hardwood floor texture, top down view, "
        "architectural interior, high quality, 4k, flat surface"
    ),
    "door": (
        "seamless tileable wooden door texture, light oak wood grain, "
        "architectural interior, high quality, 4k, flat surface"
    ),
    "window": (
        "seamless tileable frosted glass texture, translucent, "
        "architectural interior, high quality, 4k, flat surface"
    ),
    "ceiling": (
        "seamless tileable white ceiling plaster texture, smooth, "
        "architectural interior, high quality, 4k, flat surface"
    ),
}

NEGATIVE_PROMPT = (
    "cartoon, anime, blurry, low quality, distorted, furniture, "
    "people, shadows, perspective, 3d render, photorealistic room"
)


class TextureGenerator:
    """
    Generates surface textures using ControlNet conditioned on
    a solid-color segmentation patch.

    Falls back to procedural textures if ControlNet is unavailable.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import (
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                UniPCMultistepScheduler,
            )
            print("  Loading ControlNet for texture generation...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-seg",
                torch_dtype=torch.float32,
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            pipe.to(self.device)
            self._pipe = pipe
            print("  ControlNet loaded.")
        except Exception as e:
            print(f"  ControlNet unavailable ({e}), using procedural textures.")
            self._pipe = None

    def generate(self, surface_type: str, size: int = 512) -> Image.Image:
        """
        Generate a texture image for the given surface type.

        Args:
            surface_type: 'wall', 'floor', 'door', 'window', 'ceiling'
            size:         Output texture size in pixels

        Returns:
            PIL Image (RGB, size x size)
        """
        self._load()

        if self._pipe is not None:
            return self._generate_controlnet(surface_type, size)
        else:
            return self._generate_procedural(surface_type, size)

    def _generate_controlnet(self, surface_type: str, size: int) -> Image.Image:
        """Generate texture using ControlNet with a solid-color seg map."""
        from gan_part.data.room_class_map import ADE20K_PALETTE

        # Build a solid-color seg map for this surface type
        color_map = {
            "wall":    ADE20K_PALETTE[3][:3],    # grey
            "floor":   ADE20K_PALETTE[4][:3],    # cyan
            "door":    ADE20K_PALETTE[14][:3],   # dark red
            "window":  ADE20K_PALETTE[8][:3],    # yellow
            "ceiling": ADE20K_PALETTE[5][:3],    # pink
        }
        color = color_map.get(surface_type, (120, 120, 120))
        seg_map = Image.new("RGB", (size, size), color)

        prompt   = SURFACE_PROMPTS.get(surface_type, SURFACE_PROMPTS["wall"])
        negative = NEGATIVE_PROMPT

        result = self._pipe(
            prompt,
            num_inference_steps=20,
            image=seg_map,
            negative_prompt=negative,
            guidance_scale=7.5,
        ).images[0]

        return result.resize((size, size))

    def _generate_procedural(self, surface_type: str, size: int) -> Image.Image:
        """
        Fast procedural texture fallback — no AI needed.
        Generates realistic-looking tileable textures using numpy.
        """
        rng = np.random.default_rng(hash(surface_type) % (2**32))

        if surface_type == "floor":
            return self._wood_texture(size, rng, base_color=(180, 140, 90))
        elif surface_type == "door":
            return self._wood_texture(size, rng, base_color=(140, 100, 60))
        elif surface_type == "window":
            return self._glass_texture(size, rng)
        elif surface_type == "ceiling":
            return self._plaster_texture(size, rng, base_color=(245, 245, 240))
        else:  # wall
            return self._plaster_texture(size, rng, base_color=(235, 230, 220))

    def _plaster_texture(self, size: int, rng, base_color: tuple) -> Image.Image:
        """Smooth plaster/paint wall texture."""
        r, g, b = base_color
        # Base color with subtle noise
        noise = rng.normal(0, 6, (size, size, 3))
        img = np.clip(
            np.array([r, g, b], dtype=np.float32) + noise,
            0, 255
        ).astype(np.uint8)
        # Add very subtle large-scale variation
        from PIL import ImageFilter
        pil = Image.fromarray(img)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
        return pil

    def _wood_texture(self, size: int, rng, base_color: tuple) -> Image.Image:
        """Wood grain texture for floors and doors."""
        r, g, b = base_color
        img = np.zeros((size, size, 3), dtype=np.float32)

        # Wood grain lines along X axis
        for y in range(size):
            # Grain variation
            grain = np.sin(np.linspace(0, 20 * np.pi, size) +
                           rng.uniform(0, 2 * np.pi)) * 15
            brightness = 1.0 + grain / 255.0
            img[y, :, 0] = np.clip(r * brightness, 0, 255)
            img[y, :, 1] = np.clip(g * brightness, 0, 255)
            img[y, :, 2] = np.clip(b * brightness, 0, 255)

        # Add fine noise
        noise = rng.normal(0, 4, (size, size, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _glass_texture(self, size: int, rng) -> Image.Image:
        """Frosted glass texture for windows."""
        # Light blue-white with subtle noise
        base = np.array([200, 220, 240], dtype=np.float32)
        noise = rng.normal(0, 10, (size, size, 3))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        # Add transparency hint via lighter color
        from PIL import ImageFilter
        pil = Image.fromarray(img)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=3))
        return pil
