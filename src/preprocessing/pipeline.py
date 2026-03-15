"""
pipeline.py
-----------
Orchestrates the complete Phase 1 preprocessing pipeline:

  load → enhance contrast → binarize → deskew → remove noise → save

Usage:
    from src.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    result = pipeline.run("samples/floor_plan.png")
    result.save("outputs/")
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .loader import load_image, save_image
from .binarizer import binarize, enhance_contrast, remove_small_components
from .skew_corrector import deskew


@dataclass
class PreprocessingConfig:
    """All tunable parameters for the preprocessing pipeline."""
    target_size: int = 1024          # Resize longer dimension to this (pixels)
    enhance_contrast: bool = True    # Apply CLAHE before thresholding
    blur_kernel: int = 5             # Gaussian blur kernel size
    block_size: int = 25             # Adaptive threshold block size
    c_offset: int = 10               # Adaptive threshold C constant
    morph_kernel: int = 3            # Morphological ops kernel size
    min_component_area: int = 150    # Remove components smaller than this (px²)
    correct_skew: bool = True        # Enable skew correction


@dataclass
class PreprocessingResult:
    """Holds all intermediate and final outputs from the pipeline."""
    original: np.ndarray
    grayscale: np.ndarray
    contrast_enhanced: Optional[np.ndarray]
    binary: np.ndarray
    deskewed: np.ndarray
    cleaned: np.ndarray
    skew_angle: float
    source_path: str
    config: PreprocessingConfig = field(repr=False)

    def save(self, output_dir: str, prefix: str = "") -> dict[str, str]:
        """
        Save all pipeline stages to output_dir.

        Returns:
            Dict mapping stage name → saved file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem = Path(self.source_path).stem
        p = f"{prefix}{stem}" if prefix else stem

        paths = {
            "grayscale":  str(out / f"{p}_1_grayscale.png"),
            "binary":     str(out / f"{p}_2_binary.png"),
            "deskewed":   str(out / f"{p}_3_deskewed.png"),
            "cleaned":    str(out / f"{p}_4_cleaned.png"),
        }

        save_image(self.grayscale,  paths["grayscale"])
        save_image(self.binary,     paths["binary"])
        save_image(self.deskewed,   paths["deskewed"])
        save_image(self.cleaned,    paths["cleaned"])

        print(f"\nPreprocessing complete for: {self.source_path}")
        print(f"  Skew angle detected: {self.skew_angle:.2f}°")
        print(f"  Output size: {self.cleaned.shape[1]}×{self.cleaned.shape[0]} px")
        print(f"  Files saved to: {output_dir}/")

        return paths


class PreprocessingPipeline:
    """
    Full Phase 1 preprocessing pipeline for architectural floor plans.

    Example:
        pipeline = PreprocessingPipeline()
        result = pipeline.run("samples/plan.png")
        result.save("outputs/")
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def run(self, image_path: str) -> PreprocessingResult:
        """
        Execute the full pipeline on a single floor plan image.

        Args:
            image_path: Path to the input image (PNG, JPG, PDF, etc.)

        Returns:
            PreprocessingResult with all pipeline stages.
        """
        cfg = self.config

        # ── Step 1: Load & resize ─────────────────────────────────────────
        print(f"[1/5] Loading: {image_path}")
        gray = load_image(image_path, target_size=cfg.target_size)

        # Keep a copy of the original grayscale before any processing
        original = gray.copy()

        # ── Step 2: Contrast enhancement ─────────────────────────────────
        if cfg.enhance_contrast:
            print("[2/5] Enhancing contrast (CLAHE)...")
            enhanced = enhance_contrast(gray)
        else:
            enhanced = None

        source = enhanced if enhanced is not None else gray

        # ── Step 3: Binarization ──────────────────────────────────────────
        print("[3/5] Binarizing (adaptive threshold)...")
        binary = binarize(
            source,
            blur_kernel=cfg.blur_kernel,
            block_size=cfg.block_size,
            c_offset=cfg.c_offset,
            morph_kernel=cfg.morph_kernel,
        )

        # ── Step 4: Skew correction ───────────────────────────────────────
        if cfg.correct_skew:
            print("[4/5] Correcting skew...")
            deskewed_gray, angle = deskew(gray, binary)
            # Re-binarize after deskewing for a clean result
            deskewed_binary, _ = deskew(binary, binary)
        else:
            deskewed_gray = gray.copy()
            deskewed_binary = binary.copy()
            angle = 0.0

        # ── Step 5: Remove small noise components ─────────────────────────
        print("[5/5] Removing noise components...")
        cleaned = remove_small_components(
            deskewed_binary, min_area=cfg.min_component_area
        )

        return PreprocessingResult(
            original=original,
            grayscale=gray,
            contrast_enhanced=enhanced,
            binary=binary,
            deskewed=cleaned,
            cleaned=cleaned,
            skew_angle=angle,
            source_path=image_path,
            config=cfg,
        )

    def run_batch(
        self, image_paths: list[str], output_dir: str
    ) -> list[PreprocessingResult]:
        """
        Run the pipeline on multiple images.

        Args:
            image_paths: List of input image paths.
            output_dir:  Directory to save all outputs.

        Returns:
            List of PreprocessingResult objects.
        """
        results = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n── Image {i}/{len(image_paths)} ──")
            try:
                result = self.run(path)
                result.save(output_dir)
                results.append(result)
            except Exception as e:
                print(f"  ERROR processing {path}: {e}")
        return results
