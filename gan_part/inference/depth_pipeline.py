"""
depth_pipeline.py
-----------------
MiDaS depth estimation from a rendered interior image.

Takes a PIL Image (from ControlNet or Pix2Pix GAN) and returns a
depth map as a numpy array (H x W), values in [0, 1] where
1 = closest to camera, 0 = furthest.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class DepthEstimator:
    """
    Wraps MiDaS (DPT_Large) for monocular depth estimation.
    Model is lazy-loaded on first call and cached.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        print("  Loading MiDaS depth model...")
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Large",
            trust_repo=True,
        )
        self._model.to(self.device)
        self._model.eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            trust_repo=True,
        )
        self._transform = transforms.dpt_transform
        print("  MiDaS loaded.")

    def estimate(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth from a PIL RGB image.

        Returns
        -------
        np.ndarray  shape (H, W), dtype float32, values in [0, 1]
                    1 = closest, 0 = furthest
        """
        import torch

        self._load()

        # MiDaS expects RGB numpy
        img_np = np.array(image.convert("RGB"))

        input_tensor = self._transform(img_np).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]  (invert so near=1, far=0)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth
