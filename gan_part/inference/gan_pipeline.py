"""
gan_pipeline.py
---------------
Inference using local trained Pix2Pix GAN weights.
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from ..models.generator import build_generator
from ..data.preprocessor import prepare_seg_map, denormalize

class GANPipeline:
    """
    Inference using trained home-grown Pix2Pix weights.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _load(self):
        if self._model is None:
            print(f"Loading GAN Generator from {self.checkpoint_path}...")
            self._model = build_generator().to(self.device)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            if 'G' in state_dict:
                self._model.load_state_dict(state_dict['G'])
            else:
                self._model.load_state_dict(state_dict)
            self._model.eval()

    def generate(self, seg_map: np.ndarray) -> Image.Image:
        """
        Generate a room image from a segmentation map using the trained GAN.
        """
        self._load()
        
        # Preprocess
        seg_tensor = prepare_seg_map(seg_map).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            fake_tensor = self._model(seg_tensor)
        
        # Postprocess
        fake_img_array = denormalize(fake_tensor[0])
        return Image.fromarray(fake_img_array)
