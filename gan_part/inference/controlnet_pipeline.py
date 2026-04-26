"""
controlnet_pipeline.py
----------------------
Inference using pretrained ControlNet (Stable Diffusion 1.5).
"""

from __future__ import annotations

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np

class ControlNetPipeline:
    """
    Zero-shot room generation using ControlNet conditioned on segmentation maps.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            print(f"Loading ControlNet Pipeline on {self.device}...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-seg", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
            
            self._pipe = pipe

    def generate(
        self, 
        seg_map: Image.Image | np.ndarray, 
        prompt: str, 
        negative_prompt: str | None = None,
        num_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Generate a room image from a segmentation map.
        """
        self._load()
        
        if isinstance(seg_map, np.ndarray):
            seg_map = Image.fromarray(seg_map)

        image = self._pipe(
            prompt,
            num_inference_steps=num_steps,
            image=seg_map,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale
        ).images[0]
        
        return image
