"""
trainer.py
----------
Pix2Pix GAN training loop.

Orchestrates:
  - Generator + Discriminator forward passes
  - Alternating G/D optimization
  - Learning rate scheduling
  - Logging and checkpoint saving

Usage:
    from gan_part.training.config import kaggle_config
    from gan_part.training.trainer import Pix2PixTrainer

    config = kaggle_config()
    trainer = Pix2PixTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .config import TrainingConfig
from .callbacks import CheckpointCallback, SampleCallback
from ..models.generator import build_generator
from ..models.discriminator import build_discriminator
from ..models.losses import GANLoss, GeneratorLoss
from ..data.ade20k_loader import get_dataloaders


class Pix2PixTrainer:
    """
    Full pix2pix training pipeline.

    Implements the standard pix2pix training algorithm:
      1. Train Discriminator: max log(D(x,y)) + log(1 - D(x, G(x)))
      2. Train Generator:     min log(1 - D(x, G(x))) + λ * |y - G(x)|₁

    Args:
        config: TrainingConfig instance.
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.cfg.make_dirs()

        print(self.cfg.summary())

        # ── Device ────────────────────────────────────────────────────────────
        self.device = torch.device(config.device)

        # ── Models ────────────────────────────────────────────────────────────
        self.G = build_generator(
            config.in_channels, config.out_channels, config.ngf
        ).to(self.device)

        self.D = build_discriminator(
            in_channels=config.in_channels + config.out_channels,
            ndf=config.ndf,
        ).to(self.device)

        # ── Losses ────────────────────────────────────────────────────────────
        self.loss_G = GeneratorLoss(
            lambda_l1=config.lambda_l1,
            lambda_percep=config.lambda_percep,
            gan_mode=config.gan_mode,
        ).to(self.device)

        self.loss_D = GANLoss(mode=config.gan_mode).to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
        )

        # ── LR Schedulers (linear decay) ──────────────────────────────────────
        def lr_lambda(epoch: int) -> float:
            if epoch < config.decay_start:
                return 1.0
            remaining = config.num_epochs - config.decay_start
            return max(0.0, 1.0 - (epoch - config.decay_start) / remaining)

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda)

        # ── Mixed precision ───────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=config.mixed_precision and config.device == "cuda")

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = get_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            indoor_only=config.indoor_only,
            size=config.image_size,
            num_workers=config.num_workers,
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        self.checkpoint_cb = CheckpointCallback(
            config.checkpoint_dir, save_every=config.checkpoint_every
        )
        self.sample_cb = SampleCallback(
            config.samples_dir, save_every=config.sample_every
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch = 0
        self.global_step = 0

        # Resume if requested
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")

        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            self._train_epoch(epoch)
            self.sched_G.step()
            self.sched_D.step()
            self.checkpoint_cb(epoch, self.G, self.D, self.opt_G, self.opt_D)

        print("\nTraining complete!")
        self.checkpoint_cb.save(
            self.cfg.num_epochs,
            self.G, self.D, self.opt_G, self.opt_D,
            name="final",
        )

    def _train_epoch(self, epoch: int) -> None:
        """Single epoch of training."""
        self.G.train()
        self.D.train()

        epoch_start = time.time()
        total_loss_g = 0.0
        total_loss_d = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            seg   = batch["seg"].to(self.device)    # (B, 3, H, W)
            photo = batch["photo"].to(self.device)  # (B, 3, H, W)

            # ── Train Discriminator ──────────────────────────────────────────
            self.opt_D.zero_grad()
            with autocast(enabled=self.cfg.mixed_precision and self.cfg.device == "cuda"):
                with torch.no_grad():
                    fake = self.G(seg)

                real_pred = self.D(seg, photo)
                fake_pred = self.D(seg, fake.detach())

                d_real = self.loss_D(real_pred, is_real=True)
                d_fake = self.loss_D(fake_pred, is_real=False)
                d_loss = (d_real + d_fake) * 0.5

            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_D)
            total_loss_d += d_loss.item()

            # ── Train Generator ──────────────────────────────────────────────
            self.opt_G.zero_grad()
            with autocast(enabled=self.cfg.mixed_precision and self.cfg.device == "cuda"):
                fake      = self.G(seg)
                fake_pred = self.D(seg, fake)
                g_loss, loss_dict = self.loss_G(fake_pred, fake, photo)

            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.opt_G)
            self.scaler.update()

            total_loss_g += g_loss.item()
            self.global_step += 1

            # ── Logging ──────────────────────────────────────────────────────
            if self.global_step % self.cfg.log_every == 0:
                print(
                    f"Epoch {epoch+1:03d} | Step {self.global_step:06d} | "
                    f"D: {d_loss.item():.4f} | "
                    f"G: {loss_dict.get('G_total', 0):.4f} | "
                    f"L1: {loss_dict.get('G_L1', 0):.4f}"
                )

            # ── Sample images ─────────────────────────────────────────────────
            if self.global_step % self.cfg.sample_every == 0:
                self.sample_cb(
                    self.global_step, self.G, seg, photo, fake, self.device
                )

        elapsed = time.time() - epoch_start
        avg_g = total_loss_g / len(self.train_loader)
        avg_d = total_loss_d / len(self.train_loader)
        lr    = self.opt_G.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch+1:03d} done in {elapsed:.0f}s | "
            f"avg G={avg_g:.4f}, D={avg_d:.4f} | lr={lr:.2e}"
        )

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from disk to resume training."""
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.start_epoch  = ckpt.get("epoch", 0) + 1
        self.global_step  = ckpt.get("global_step", 0)
        print(f"Resumed from checkpoint: {path} (epoch {self.start_epoch})")
