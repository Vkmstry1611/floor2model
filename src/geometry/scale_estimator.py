"""
scale_estimator.py
------------------
Estimates the pixel-to-metre scale of a floor plan image.

Strategies (in priority order):
  1. OCR — detect dimension annotations in the image (e.g. "4.5m", "450cm")
  2. Standard room size — use detected room polygons and compare to known
     average room dimensions to infer scale
  3. Fallback — assume a standard A4/A3 drawing at 1:100 scale

Usage:
    from src.geometry.scale_estimator import ScaleEstimator

    estimator = ScaleEstimator()
    scale = estimator.estimate(image, vectorization_result)
    print(f"1 pixel = {scale.pixels_per_metre:.2f} px/m")
    real_area = polygon.area / (scale.pixels_per_metre ** 2)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ScaleEstimate:
    """Pixel-to-real-world scale estimate."""
    pixels_per_metre: float         # How many pixels = 1 metre
    confidence:       float         # 0-1, how confident we are
    method:           str           # 'ocr', 'room_size', 'fallback'
    notes:            str = ""

    @property
    def metres_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_metre

    def pixels_to_metres(self, pixels: float) -> float:
        return pixels / self.pixels_per_metre

    def metres_to_pixels(self, metres: float) -> float:
        return metres * self.pixels_per_metre

    def area_px_to_m2(self, area_px: float) -> float:
        return area_px / (self.pixels_per_metre ** 2)


# ── Known room size priors (metres²) ─────────────────────────────────────────

ROOM_SIZE_PRIORS = {
    "Bedroom":    (9.0,  20.0),   # typical: 9–20 m²
    "LivingRoom": (15.0, 40.0),   # typical: 15–40 m²
    "Kitchen":    (7.0,  20.0),   # typical: 7–20 m²
    "Bathroom":   (3.0,  8.0),    # typical: 3–8 m²
    "Corridor":   (2.0,  12.0),   # typical: 2–12 m²
    "Balcony":    (2.0,  10.0),   # typical: 2–10 m²
}


# ── Scale estimator ───────────────────────────────────────────────────────────

class ScaleEstimator:
    """
    Estimates pixel-to-metre conversion factor for a floor plan.

    Args:
        target_image_size: The longer dimension of the normalized image (px).
                           Should match Phase 1 target_size (default 1024).
        fallback_scale:    Metres per pixel to use when all else fails.
                           Default assumes a 10m × 10m plan at 1024px = 10m.
    """

    def __init__(
        self,
        target_image_size: int = 1024,
        fallback_scale: float = 0.015,  # ~15mm per pixel = typical floor plan
    ):
        self.target_image_size = target_image_size
        self.fallback_scale = fallback_scale

    def estimate(
        self,
        image: np.ndarray,
        vectorization_result=None,
    ) -> ScaleEstimate:
        """
        Estimate scale using all available strategies.

        Args:
            image:                Grayscale or BGR floor plan image.
            vectorization_result: Optional VectorizationResult from Phase 3.

        Returns:
            ScaleEstimate with best available estimate.
        """
        # Strategy 1: OCR-based scale detection
        ocr_result = self._estimate_from_ocr(image)
        if ocr_result is not None:
            return ocr_result

        # Strategy 2: Room size heuristics
        if vectorization_result is not None and vectorization_result.rooms:
            room_result = self._estimate_from_rooms(vectorization_result)
            if room_result is not None:
                return room_result

        # Strategy 3: Fallback
        return self._fallback_estimate(image)

    def pixels_to_metres(self, pixels: float, scale: ScaleEstimate) -> float:
        return scale.pixels_to_metres(pixels)

    # ── Strategies ────────────────────────────────────────────────────────────

    def _estimate_from_ocr(
        self, image: np.ndarray
    ) -> Optional[ScaleEstimate]:
        """
        Try to detect dimension text (e.g. '4.5m', '3500mm', '450cm')
        in the image using pytesseract OCR.
        """
        try:
            import pytesseract
        except ImportError:
            return None

        try:
            # Preprocess for OCR — invert if needed, sharpen
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Upscale small images for better OCR
            h, w = gray.shape[:2]
            if max(h, w) < 1000:
                scale_up = 1000 / max(h, w)
                gray = cv2.resize(gray, None, fx=scale_up, fy=scale_up,
                                  interpolation=cv2.INTER_LINEAR)

            # Threshold for cleaner text
            _, thresh = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            text = pytesseract.image_to_string(
                thresh,
                config='--psm 11 -c tessedit_char_whitelist=0123456789.,m c'
            )

            scale = self._parse_dimension_text(text, image.shape)
            if scale is not None:
                return scale

        except Exception:
            pass

        return None

    def _parse_dimension_text(
        self, text: str, image_shape: tuple
    ) -> Optional[ScaleEstimate]:
        """Parse OCR text to find dimension annotations."""
        h, w = image_shape[:2]
        img_size = max(h, w)

        # Patterns: "4.5m", "4,500mm", "450cm", "4.5 m"
        patterns = [
            (r'(\d+\.?\d*)\s*m(?:etres?|eters?)?\b(?!m)', 1.0),      # metres
            (r'(\d+\.?\d*)\s*cm\b', 0.01),                             # cm → m
            (r'(\d{3,5})\s*mm\b', 0.001),                              # mm → m
        ]

        found_dims = []
        for pattern, multiplier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_m = float(match.group(1)) * multiplier
                if 0.5 <= value_m <= 50.0:  # sanity check: 0.5m to 50m
                    found_dims.append(value_m)

        if not found_dims:
            return None

        # Use median dimension as reference
        ref_dim_m = float(np.median(found_dims))

        # Assume the reference dimension spans ~40% of the image
        ref_dim_px = img_size * 0.4
        pixels_per_metre = ref_dim_px / ref_dim_m

        return ScaleEstimate(
            pixels_per_metre=pixels_per_metre,
            confidence=0.75,
            method="ocr",
            notes=f"Detected dimension: {ref_dim_m:.2f}m from OCR",
        )

    def _estimate_from_rooms(self, vectorization_result) -> Optional[ScaleEstimate]:
        """
        Use known room size priors to estimate scale.
        Compares detected room pixel areas to expected real-world areas.
        """
        scale_estimates = []

        for room in vectorization_result.rooms:
            class_name = room.class_name
            if class_name not in ROOM_SIZE_PRIORS:
                continue

            min_m2, max_m2 = ROOM_SIZE_PRIORS[class_name]
            mid_m2 = (min_m2 + max_m2) / 2.0
            area_px = room.area

            if area_px <= 0:
                continue

            # pixels_per_metre² = area_px / mid_m2
            # pixels_per_metre = sqrt(area_px / mid_m2)
            ppm = (area_px / mid_m2) ** 0.5
            scale_estimates.append(ppm)

        if not scale_estimates:
            return None

        # Use median to be robust against outliers
        ppm = float(np.median(scale_estimates))

        # Sanity check: 10–500 px/m is reasonable
        if not (10 <= ppm <= 500):
            return None

        return ScaleEstimate(
            pixels_per_metre=ppm,
            confidence=0.55,
            method="room_size",
            notes=f"Estimated from {len(scale_estimates)} room(s)",
        )

    def _fallback_estimate(self, image: np.ndarray) -> ScaleEstimate:
        """
        Fallback: assume standard floor plan proportions.
        A 1024px image typically represents a ~10-15m building footprint.
        """
        h, w = image.shape[:2]
        img_size = max(h, w)

        # Assume building footprint ≈ 12m on the longer axis
        assumed_building_size_m = 12.0
        ppm = img_size / assumed_building_size_m

        return ScaleEstimate(
            pixels_per_metre=ppm,
            confidence=0.30,
            method="fallback",
            notes=f"Fallback: assumed {assumed_building_size_m}m building at {img_size}px",
        )
