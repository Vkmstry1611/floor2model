"""
tests/test_preprocessing.py
----------------------------
Unit tests for Phase 1 preprocessing pipeline.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.binarizer import (
    binarize,
    enhance_contrast,
    remove_small_components,
)
from src.preprocessing.skew_corrector import (
    detect_skew_angle,
    correct_skew,
    deskew,
)
from src.preprocessing.loader import _resize_keep_aspect


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_gray(h=256, w=256, value=128) -> np.ndarray:
    """Create a uniform grayscale image."""
    return np.full((h, w), value, dtype=np.uint8)


def make_floor_plan_mock(h=512, w=512) -> np.ndarray:
    """
    Create a synthetic floor plan: white background with black wall lines.
    Mimics a simple room layout.
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    # Outer walls
    img[50:460, 50:52] = 0    # left wall
    img[50:460, 460:462] = 0  # right wall
    img[50:52, 50:462] = 0    # top wall
    img[458:460, 50:462] = 0  # bottom wall
    # Interior wall (room divider)
    img[50:260, 250:252] = 0
    return img


# ── Binarizer tests ───────────────────────────────────────────────────────────

class TestBinarizer:

    def test_binarize_returns_binary_image(self):
        img = make_floor_plan_mock()
        result = binarize(img)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255}), f"Expected only 0/255, got {unique}"

    def test_binarize_correct_shape(self):
        img = make_floor_plan_mock(300, 400)
        result = binarize(img)
        assert result.shape == (300, 400)

    def test_binarize_rejects_color_image(self):
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            binarize(color_img)

    def test_binarize_rejects_none(self):
        with pytest.raises(TypeError):
            binarize(None)

    def test_enhance_contrast_preserves_shape(self):
        img = make_gray(200, 300)
        result = enhance_contrast(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_remove_small_components_removes_specks(self):
        binary = np.zeros((200, 200), dtype=np.uint8)
        # Large component (should be kept)
        binary[10:50, 10:50] = 255
        # Small speck (should be removed)
        binary[100, 100] = 255
        binary[101, 101] = 255

        result = remove_small_components(binary, min_area=50)
        # Large component should still be there
        assert result[30, 30] == 255
        # Speck should be gone
        assert result[100, 100] == 0

    def test_remove_small_components_keeps_large(self):
        binary = np.zeros((200, 200), dtype=np.uint8)
        binary[10:100, 10:100] = 255  # large region
        result = remove_small_components(binary, min_area=50)
        assert np.sum(result > 0) > 0


# ── Skew corrector tests ──────────────────────────────────────────────────────

class TestSkewCorrector:

    def test_detect_skew_zero_on_clean_image(self):
        """A synthetic floor plan with axis-aligned walls should have near-zero skew."""
        plan = make_floor_plan_mock()
        binary = binarize(plan)
        angle = detect_skew_angle(binary)
        assert abs(angle) < 5.0, f"Expected ~0° skew, got {angle:.2f}°"

    def test_detect_skew_returns_float(self):
        binary = binarize(make_floor_plan_mock())
        angle = detect_skew_angle(binary)
        assert isinstance(angle, float)

    def test_correct_skew_returns_same_shape(self):
        img = make_floor_plan_mock()
        corrected = correct_skew(img, angle=3.0)
        assert corrected.shape == img.shape

    def test_correct_skew_no_rotation_on_small_angle(self):
        """Angles < 0.3° should be skipped to avoid resampling artifacts."""
        img = make_floor_plan_mock()
        result = correct_skew(img, angle=0.2)
        np.testing.assert_array_equal(result, img)

    def test_correct_skew_raises_without_angle_or_binary(self):
        img = make_gray()
        with pytest.raises(ValueError):
            correct_skew(img)

    def test_deskew_returns_tuple(self):
        img = make_floor_plan_mock()
        binary = binarize(img)
        result = deskew(img, binary)
        assert isinstance(result, tuple)
        assert len(result) == 2
        corrected, angle = result
        assert corrected.shape == img.shape
        assert isinstance(angle, float)


# ── Loader helper tests ───────────────────────────────────────────────────────

class TestLoaderHelpers:

    def test_resize_keep_aspect_downscale(self):
        img = make_gray(2000, 1000)
        result = _resize_keep_aspect(img, 1024)
        h, w = result.shape[:2]
        assert max(h, w) == 1024
        assert abs(w / h - 1000 / 2000) < 0.05  # aspect ratio preserved

    def test_resize_keep_aspect_already_correct(self):
        img = make_gray(1024, 768)
        result = _resize_keep_aspect(img, 1024)
        assert result.shape[0] == 1024

    def test_resize_keep_aspect_upscale(self):
        img = make_gray(200, 100)
        result = _resize_keep_aspect(img, 500)
        assert max(result.shape) == 500
