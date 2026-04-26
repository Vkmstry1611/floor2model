"""
room_renderer.py
----------------
Unified interface for room rendering, supporting multiple backends.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Literal

from .prompt_engine import PromptEngine
from .controlnet_pipeline import ControlNetPipeline
from .gan_pipeline import GANPipeline

class RoomRenderer:
    """
    Unified manager for rendering room interiors.
    Can switch between 'gan' (trained weights) and 'controlnet' (zero-shot).
    """

    def __init__(
        self, 
        backend: Literal["gan", "controlnet"] = "controlnet", 
        gan_checkpoint: str | None = None,
        device: str = "cpu"
    ):
        self.backend = backend
        self.device = device
        self.prompt_engine = PromptEngine()
        
        self.gan_pipe = None
        self.controlnet_pipe = None
        
        if backend == "gan":
            if not gan_checkpoint:
                raise ValueError("gan_checkpoint required for 'gan' backend")
            self.gan_pipe = GANPipeline(gan_checkpoint, device=device)
        else:
            self.controlnet_pipe = ControlNetPipeline(device=device)

    def render(
        self, 
        seg_map: np.ndarray, 
        room_label: str, 
        user_prompt: str | None = None,
        **kwargs
    ) -> Image.Image:
        """
        Render a single room.
        """
        if self.backend == "controlnet":
            prompt = self.prompt_engine.build(room_label, user_prompt)
            negative = self.prompt_engine.get_negative()
            return self.controlnet_pipe.generate(
                seg_map, 
                prompt, 
                negative_prompt=negative,
                **kwargs
            )
        else:
            # GAN doesn't naturally handle text prompts (unless we add conditioning)
            # but for pix2pix it's just image-to-image
            return self.gan_pipe.generate(seg_map)
