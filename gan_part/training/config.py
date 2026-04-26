"""
config.py
---------
Training configuration dataclass for the pix2pix GAN.

All hyperparameters are centralised here. Pass to Trainer and Callbacks.
Works both locally (for testing) and on Kaggle (production training).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainingConfig:
    """
    Pix2Pix training hyperparameters.

    Reasonable defaults are set for ADE20K indoor scene training on
    a Kaggle T4/P100 GPU.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root:    str  = "data/ade20k"         # Root path to ADE20K dataset
    lsun_root:    Optional[str] = None          # Optional LSUN for fine-tuning
    indoor_only:  bool = True                   # Use indoor scenes only
    image_size:   int  = 256                    # Training resolution
    num_workers:  int  = 4                      # DataLoader workers

    # ── Model ─────────────────────────────────────────────────────────────────
    ngf:          int  = 64                     # Generator base filters
    ndf:          int  = 64                     # Discriminator base filters
    in_channels:  int  = 3                      # Seg map channels
    out_channels: int  = 3                      # Output image channels

    # ── Losses ────────────────────────────────────────────────────────────────
    gan_mode:      Literal["lsgan", "vanilla", "wgan"] = "lsgan"
    lambda_l1:     float = 100.0               # Weight for L1 loss
    lambda_percep: float = 10.0                # Weight for perceptual loss (0=disabled)

    # ── Training schedule ─────────────────────────────────────────────────────
    num_epochs:   int   = 100                   # Total training epochs
    batch_size:   int   = 4                     # Samples per batch
    lr_g:         float = 2e-4                  # Generator learning rate
    lr_d:         float = 2e-4                  # Discriminator learning rate
    beta1:        float = 0.5                   # Adam β1 (standard GAN value)
    beta2:        float = 0.999                 # Adam β2

    # LR decay: keep lr for first `decay_start` epochs, then linearly decay to 0
    decay_start:  int   = 50                    # Epoch to start LR decay
    n_disc_steps: int   = 1                     # Discriminator steps per generator step

    # ── Checkpointing & logging ────────────────────────────────────────────────
    output_dir:       str  = "outputs/training"   # Where to save everything
    checkpoint_every: int  = 5                    # Save checkpoint every N epochs
    sample_every:     int  = 200                  # Save sample images every N steps
    log_every:        int  = 50                   # Print loss every N steps
    resume_from:      Optional[str] = None        # Path to checkpoint to resume from

    # ── Hardware ──────────────────────────────────────────────────────────────
    device:     str  = "auto"           # "auto", "cuda", "mps", "cpu"
    mixed_precision: bool = True        # Use fp16 for faster GPU training

    # ── Kaggle-specific ───────────────────────────────────────────────────────
    is_kaggle:  bool = False            # Set True when running on Kaggle

    def __post_init__(self):
        """Resolve device and adjust for Kaggle environment."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                self.mixed_precision = False  # MPS doesn't support fp16 GAN well
            else:
                self.device = "cpu"
                self.mixed_precision = False

        # Kaggle adjustments
        if self.is_kaggle:
            self.data_root  = "/kaggle/input/ade20k-mit/ADEChallengeData2016"
            self.output_dir = "/kaggle/working/outputs"
            self.num_workers = 2

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    @property
    def samples_dir(self) -> Path:
        return Path(self.output_dir) / "samples"

    @property
    def logs_dir(self) -> Path:
        return Path(self.output_dir) / "logs"

    def make_dirs(self) -> None:
        """Create all output directories."""
        for d in (self.checkpoint_dir, self.samples_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Human-readable config summary."""
        return (
            f"Training Config\n"
            f"  Device:       {self.device} ({'fp16' if self.mixed_precision else 'fp32'})\n"
            f"  Epochs:       {self.num_epochs} (decay from {self.decay_start})\n"
            f"  Batch size:   {self.batch_size}\n"
            f"  Image size:   {self.image_size}×{self.image_size}\n"
            f"  LR:           G={self.lr_g}, D={self.lr_d}\n"
            f"  Loss weights: L1={self.lambda_l1}, Perc={self.lambda_percep}\n"
            f"  GAN mode:     {self.gan_mode}\n"
            f"  Output dir:   {self.output_dir}\n"
        )


# ── Preset configs ─────────────────────────────────────────────────────────────

def kaggle_config() -> TrainingConfig:
    """Optimised config for Kaggle T4 (16 GB VRAM)."""
    return TrainingConfig(
        is_kaggle=True,
        batch_size=8,
        num_epochs=100,
        decay_start=50,
        num_workers=2,
        mixed_precision=True,
        lambda_percep=10.0,
        sample_every=100,
        checkpoint_every=10,
    )


def local_test_config() -> TrainingConfig:
    """Minimal config for local testing (CPU or small GPU)."""
    return TrainingConfig(
        batch_size=1,
        num_epochs=2,
        decay_start=1,
        num_workers=0,
        mixed_precision=False,
        lambda_percep=0.0,    # Disable perceptual loss for speed
        log_every=10,
        sample_every=20,
        checkpoint_every=1,
        output_dir="outputs/test_run",
    )
