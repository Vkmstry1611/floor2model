"""
discriminator.py
----------------
Pix2Pix PatchGAN Discriminator.

Instead of classifying the whole image as real/fake, PatchGAN classifies
overlapping N×N patches (default 70×70). Each patch score is averaged to
give the final discriminator output.

Input:  6-channel image = concatenated (seg_map, photo_or_generated)
Output: Scalar-per-patch tensor — averaged to get real/fake score.

Reference:
    "Image-to-Image Translation with Conditional Adversarial Networks"
    Isola et al., CVPR 2017
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Takes concatenated (condition, image) and predicts patch-level
    realness scores.

    Input:  (B, 6, 256, 256)  — 3 ch seg_map + 3 ch photo
    Output: (B, 1, 30, 30)    — patch realness map (averaged during loss)
    """

    def __init__(self, in_channels: int = 6, ndf: int = 64):
        """
        Args:
            in_channels: 6 = seg_map(3) + image(3).
            ndf:         Base number of discriminator filters.
        """
        super().__init__()

        # 4 conv blocks → 30×30 receptive field on 256×256 input
        self.model = nn.Sequential(
            # Layer 1: no BN on first layer
            nn.Conv2d(in_channels, ndf,      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ndf,        ndf * 2,   kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ndf * 2,    ndf * 4,   kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 (stride=1)
            nn.Conv2d(ndf * 4,    ndf * 8,   kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer — 1 channel per patch
            nn.Conv2d(ndf * 8,    1,          kernel_size=4, stride=1, padding=1),
            # No Sigmoid here — we use BCEWithLogitsLoss or MSELoss (LSGAN)
        )

    def forward(self, seg_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_map: (B, 3, H, W) condition (segmentation map).
            image:   (B, 3, H, W) real or generated image.

        Returns:
            (B, 1, h, w) patch prediction map.
        """
        x = torch.cat([seg_map, image], dim=1)   # (B, 6, H, W)
        return self.model(x)

    def num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_discriminator(in_channels: int = 6, ndf: int = 64) -> PatchGANDiscriminator:
    """
    Factory: create and initialise a PatchGAN discriminator.

    Args:
        in_channels: Input channels (default 6 = seg+image).
        ndf:         Base filter multiplier.

    Returns:
        Initialised PatchGANDiscriminator.
    """
    from .generator import _init_weights
    disc = PatchGANDiscriminator(in_channels, ndf)
    _init_weights(disc)
    print(f"Discriminator: {disc.num_params() / 1e6:.2f}M parameters")
    return disc
