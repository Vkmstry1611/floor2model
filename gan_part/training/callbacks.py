"""
callbacks.py
------------
Training callbacks for saving checkpoints and sample images.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from ..data.preprocessor import denormalize, make_comparison_grid


class CheckpointCallback:
    """Saves model checkpoints during training."""
    def __init__(self, checkpoint_dir: Path, save_every: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

    def __call__(self, epoch, G, D, opt_G, opt_D, name=None):
        if name or (epoch + 1) % self.save_every == 0:
            self.save(epoch, G, D, opt_G, opt_D, name=name)

    def save(self, epoch, G, D, opt_G, opt_D, name=None):
        name = name or f"epoch_{epoch+1:03d}"
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            'epoch': epoch,
            'G': G.state_dict(),
            'D': D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")


class SampleCallback:
    """Saves visual comparison grids during training."""
    def __init__(self, samples_dir: Path, save_every: int = 200):
        self.samples_dir = samples_dir
        self.save_every = save_every

    def __call__(self, step, G, seg, photo, fake, device):
        # We only save on the specified step frequency
        with torch.no_grad():
            G.eval()
            # Pick first sample from batch
            s_seg = denormalize(seg[0])
            s_real = denormalize(photo[0])
            s_fake = denormalize(fake[0])
            G.train()

        grid = make_comparison_grid(s_seg, s_real, s_fake)
        path = self.samples_dir / f"step_{step:06d}.png"
        Image.fromarray(grid).save(path)
        print(f"  Sample saved: {path}")
