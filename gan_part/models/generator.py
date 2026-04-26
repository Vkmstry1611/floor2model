"""
generator.py
------------
Pix2Pix UNet Generator for room interior synthesis.

Architecture:
  Encoder-decoder with skip connections. Semantic segmentation map (3-channel,
  ADE20K-colorized) goes in → realistic room photo (3-channel RGB) comes out.

  Input:  (B, 3, 256, 256)  — normalized seg map
  Output: (B, 3, 256, 256)  — normalized room photo

Based on the original Pix2Pix paper:
  "Image-to-Image Translation with Conditional Adversarial Networks"
  Isola et al., CVPR 2017
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Pix2Pix encoder block: Conv → BatchNorm → LeakyReLU
    First block has no batchnorm (input layer convention).
    """
    def __init__(self, in_ch: int, out_ch: int, batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not batchnorm)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Pix2Pix decoder block: ConvTranspose → BatchNorm → Dropout → ReLU
    Dropout is applied in first 3 decoder blocks for stochasticity.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)   # Skip connection


# ── UNet Generator ─────────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    8-level U-Net generator for pix2pix.

    Input:  3-channel RGB segmentation map
    Output: 3-channel RGB photo

    Channel progression:
        Encoder: 3 → 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512 (bottleneck)
        Decoder: (skip) 512 → 512 → 512 → 512 → 256 → 128 → 64 → 3
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64):
        """
        Args:
            in_channels:  Input channels (3 for RGB seg map).
            out_channels: Output channels (3 for RGB photo).
            ngf:          Base number of generator filters.
        """
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # ── Encoder ─────────────────────────────────────────────────────────
        # E1: 256 → 128  (no batchnorm on first layer)
        self.e1 = EncoderBlock(in_channels,    ngf,      batchnorm=False)   # 64
        # E2: 128 → 64
        self.e2 = EncoderBlock(ngf,            ngf * 2)                     # 128
        # E3: 64 → 32
        self.e3 = EncoderBlock(ngf * 2,        ngf * 4)                     # 256
        # E4: 32 → 16
        self.e4 = EncoderBlock(ngf * 4,        ngf * 8)                     # 512
        # E5: 16 → 8
        self.e5 = EncoderBlock(ngf * 8,        ngf * 8)                     # 512
        # E6: 8 → 4
        self.e6 = EncoderBlock(ngf * 8,        ngf * 8)                     # 512
        # E7: 4 → 2
        self.e7 = EncoderBlock(ngf * 8,        ngf * 8)                     # 512
        # E8 (bottleneck): 2 → 1
        self.e8 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ─────────────────────────────────────────────────────────
        # D1: 1 → 2  (no skip from bottleneck, dropout)
        self.d1 = DecoderBlock(ngf * 8,        ngf * 8, dropout=True)   # out: 512, cat→1024
        # D2: 2 → 4
        self.d2 = DecoderBlock(ngf * 8 * 2,   ngf * 8, dropout=True)   # out: 512, cat→1024
        # D3: 4 → 8
        self.d3 = DecoderBlock(ngf * 8 * 2,   ngf * 8, dropout=True)   # out: 512, cat→1024
        # D4: 8 → 16
        self.d4 = DecoderBlock(ngf * 8 * 2,   ngf * 8)                  # out: 512, cat→1024
        # D5: 16 → 32
        self.d5 = DecoderBlock(ngf * 8 * 2,   ngf * 4)                  # out: 256, cat→512
        # D6: 32 → 64
        self.d6 = DecoderBlock(ngf * 4 * 2,   ngf * 2)                  # out: 128, cat→256
        # D7: 64 → 128
        self.d7 = DecoderBlock(ngf * 2 * 2,   ngf)                      # out: 64,  cat→128

        # D8 (output): 128 → 256
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 256) normalized seg map.

        Returns:
            (B, 3, 256, 256) normalized generated image in [-1, 1].
        """
        # Encode
        e1 = self.e1(x)     # (B, 64,  128, 128)
        e2 = self.e2(e1)    # (B, 128,  64,  64)
        e3 = self.e3(e2)    # (B, 256,  32,  32)
        e4 = self.e4(e3)    # (B, 512,  16,  16)
        e5 = self.e5(e4)    # (B, 512,   8,   8)
        e6 = self.e6(e5)    # (B, 512,   4,   4)
        e7 = self.e7(e6)    # (B, 512,   2,   2)
        e8 = self.e8(e7)    # (B, 512,   1,   1)  ← bottleneck

        # Decode with skip connections
        d1 = self.d1(e8, e7)    # (B, 1024,  2,   2)
        d2 = self.d2(d1, e6)    # (B, 1024,  4,   4)
        d3 = self.d3(d2, e5)    # (B, 1024,  8,   8)
        d4 = self.d4(d3, e4)    # (B, 1024, 16,  16)
        d5 = self.d5(d4, e3)    # (B,  512, 32,  32)
        d6 = self.d6(d5, e2)    # (B,  256, 64,  64)
        d7 = self.d7(d6, e1)    # (B,  128,128, 128)
        out = self.d8(d7)        # (B,    3,256, 256) ← final output

        return out

    def num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_generator(in_channels: int = 3, out_channels: int = 3, ngf: int = 64) -> UNetGenerator:
    """
    Factory function to create and optionally initialise the generator.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
        ngf:          Base filter count.

    Returns:
        Initialised UNetGenerator.
    """
    gen = UNetGenerator(in_channels, out_channels, ngf)
    _init_weights(gen)
    print(f"Generator: {gen.num_params() / 1e6:.2f}M parameters")
    return gen


def _init_weights(model: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """
    Initialise Conv and BatchNorm weights using a Gaussian distribution.
    Used by both generator and discriminator.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, std)
            nn.init.constant_(m.bias.data, 0)
