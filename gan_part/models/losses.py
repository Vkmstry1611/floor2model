"""
losses.py
---------
Loss functions for the pix2pix GAN.

Three losses are combined:
  1. GAN Loss (adversarial) — fool the discriminator
  2. L1 Loss (pixel-wise)  — keep output close to ground truth
  3. Perceptual Loss (optional) — match VGG feature space for sharper results

Training objective:
    G* = argmin_{G} max_{D} L_cGAN(G, D) + λ * L_L1(G)

Where λ = 100 (default, as per original paper).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── GAN Loss ──────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Flexible GAN adversarial loss.

    Supports:
      - 'lsgan'   — Mean Squared Error (MSE), more stable training
      - 'vanilla' — Binary Cross Entropy (BCE), original pix2pix
      - 'wgan'    — Wasserstein distance for discriminator

    Args:
        mode:       Loss mode ('lsgan', 'vanilla', 'wgan').
        real_label: Target value for real samples (default 1.0).
        fake_label: Target value for fake samples (default 0.0).
    """

    def __init__(
        self,
        mode: Literal["lsgan", "vanilla", "wgan"] = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.real_label = real_label
        self.fake_label = fake_label

        if mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif mode == "wgan":
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown GAN mode: {mode}")

    def _get_target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if is_real:
            return torch.ones_like(pred) * self.real_label
        return torch.ones_like(pred) * self.fake_label

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Compute the GAN loss.

        Args:
            pred:    Discriminator output patch tensor (B, 1, h, w).
            is_real: True for real samples, False for fake.

        Returns:
            Scalar loss tensor.
        """
        if self.mode == "wgan":
            # Wasserstein: real → maximize, fake → minimize
            return -pred.mean() if is_real else pred.mean()

        target = self._get_target(pred, is_real)
        return self.loss_fn(pred, target)


# ── L1 Loss ───────────────────────────────────────────────────────────────────

class L1Loss(nn.Module):
    """
    Plain pixel-wise L1 reconstruction loss.
    Encourages the generator to be faithful to the ground truth.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: Generator output (B, 3, H, W) in [-1, 1].
            target:    Ground truth photo (B, 3, H, W) in [-1, 1].

        Returns:
            Scalar L1 loss.
        """
        return self.loss_fn(generated, target)


# ── Perceptual (VGG) Loss ─────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16 feature matching loss (perceptual loss).

    Computes the L2 distance between VGG16 intermediate feature maps of the
    generated and real images. Produces sharper, more photorealistic textures
    compared to L1-only training.

    Uses layers: relu1_2, relu2_2, relu3_3, relu4_3 (standard choice).
    VGG model is frozen — only used for feature extraction.

    Args:
        layers:  List of VGG16 layer indices to extract features from.
        weights: Per-layer contribution weights.
    """

    VGG_LAYERS = [3, 8, 15, 22]    # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(
        self,
        layers: list[int] = None,
        weights: list[float] = None,
    ):
        super().__init__()
        self.layers  = layers  or self.VGG_LAYERS
        self.weights = weights or [1.0 / len(self.VGG_LAYERS)] * len(self.VGG_LAYERS)

        # Load VGG16 (pretrained) and freeze it
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.ModuleList(
                [nn.Sequential(*list(vgg.features.children())[:l+1]) for l in self.layers]
            )
            for p in self.parameters():
                p.requires_grad_(False)
        except ImportError:
            raise ImportError("torchvision required for PerceptualLoss.")

        # ImageNet normalization (VGG expects this)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1,1] range to ImageNet-normalized [0,1]."""
        x = (x + 1.0) / 2.0              # [-1,1] → [0,1]
        return (x - self.mean) / self.std  # ImageNet normalize

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: Generator output (B, 3, H, W) in [-1, 1].
            target:    Ground truth photo (B, 3, H, W) in [-1, 1].

        Returns:
            Scalar perceptual loss.
        """
        gen_prep = self._preprocess(generated)
        tgt_prep = self._preprocess(target.detach())

        total = 0.0
        for i, extractor in enumerate(self.feature_extractor):
            gen_feat = extractor(gen_prep)
            tgt_feat = extractor(tgt_prep)
            total += self.weights[i] * F.mse_loss(gen_feat, tgt_feat)

        return total


# ── Combined Generator Loss ───────────────────────────────────────────────────

class GeneratorLoss(nn.Module):
    """
    Combined generator training loss:
        L_G = L_GAN(fake) + λ_l1 * L_L1 + λ_percep * L_Perceptual

    Args:
        lambda_l1:      Weight for L1 loss (default 100, per paper).
        lambda_percep:  Weight for perceptual loss (default 10, 0 = disabled).
        gan_mode:       GAN loss type ('lsgan', 'vanilla', 'wgan').
    """

    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_percep: float = 10.0,
        gan_mode: str = "lsgan",
    ):
        super().__init__()
        self.lambda_l1     = lambda_l1
        self.lambda_percep = lambda_percep

        self.gan_loss   = GANLoss(mode=gan_mode)
        self.l1_loss    = L1Loss()
        self.perc_loss  = PerceptualLoss() if lambda_percep > 0 else None

    def forward(
        self,
        fake_pred: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            fake_pred:  Discriminator output on generated image (B,1,h,w).
            generated:  Generator output image (B,3,H,W).
            target:     Ground truth photo (B,3,H,W).

        Returns:
            (total_loss, loss_dict): scalar total and dict of components.
        """
        g_adv  = self.gan_loss(fake_pred, is_real=True)
        g_l1   = self.l1_loss(generated, target) * self.lambda_l1
        total  = g_adv + g_l1

        loss_dict = {
            "G_adv": g_adv.item(),
            "G_L1":  g_l1.item(),
        }

        if self.perc_loss is not None and self.lambda_percep > 0:
            g_perc = self.perc_loss(generated, target) * self.lambda_percep
            total  += g_perc
            loss_dict["G_perc"] = g_perc.item()

        loss_dict["G_total"] = total.item()
        return total, loss_dict
