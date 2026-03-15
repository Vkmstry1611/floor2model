"""
binarizer.py
------------
Converts a grayscale floor plan image into a clean binary (black/white) image.

Pipeline:
  1. Gaussian blur  →  reduce sensor/scan noise
  2. Adaptive threshold  →  handle uneven lighting across the page
  3. Morphological close  →  fill tiny gaps in wall lines
  4. Morphological open   →  remove isolated specks
"""

import cv2
import numpy as np


def binarize(
    img: np.ndarray,
    blur_kernel: int = 5,
    block_size: int = 25,
    c_offset: int = 10,
    morph_kernel: int = 3,
) -> np.ndarray:
    """
    Convert a grayscale image to a clean binary image.

    Args:
        img:          Grayscale uint8 numpy array.
        blur_kernel:  Gaussian blur kernel size (must be odd).
        block_size:   Neighbourhood size for adaptive threshold (must be odd, ≥3).
        c_offset:     Constant subtracted from the mean in adaptive threshold.
                      Higher = more aggressive (removes faint lines too).
        morph_kernel: Kernel size for morphological cleanup.

    Returns:
        Binary image (0 = background, 255 = foreground/walls), uint8.
    """
    _validate_grayscale(img)

    # 1. Gaussian blur to suppress scan noise
    blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

    # 2. Adaptive threshold — handles uneven illumination better than Otsu
    #    THRESH_BINARY_INV: walls/lines become white (255) on black background
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_offset,
    )

    # 3. Morphological closing: fills small breaks in wall lines
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel, morph_kernel)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # 4. Morphological opening: removes isolated noise specks
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel, morph_kernel)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    return binary


def remove_small_components(
    binary: np.ndarray, min_area: int = 100
) -> np.ndarray:
    """
    Remove connected components smaller than min_area pixels.
    Useful for eliminating text fragments and scan artifacts.

    Args:
        binary:   Binary image (uint8, values 0 or 255).
        min_area: Components with fewer pixels than this are removed.

    Returns:
        Cleaned binary image.
    """
    _validate_grayscale(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Background is label 0 — skip it
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves visibility of faint lines before thresholding.

    Args:
        img: Grayscale uint8 numpy array.

    Returns:
        Contrast-enhanced grayscale image.
    """
    _validate_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _validate_grayscale(img: np.ndarray) -> None:
    if img is None or not isinstance(img, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if len(img.shape) != 2:
        raise ValueError(
            f"Expected a grayscale (2D) image, got shape {img.shape}. "
            "Convert to grayscale first."
        )
