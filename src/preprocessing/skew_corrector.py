"""
skew_corrector.py
-----------------
Detects and corrects rotational skew in scanned/photographed floor plans.

Strategy:
  1. Detect dominant lines using Probabilistic Hough Transform.
  2. Compute the median angle of near-horizontal and near-vertical lines.
  3. Rotate the image to align the dominant axis.

Floor plans are axis-aligned by design, so the dominant line angle
should be close to 0° or 90°.
"""

import cv2
import numpy as np
from typing import Optional


def detect_skew_angle(binary: np.ndarray, angle_threshold: float = 45.0) -> float:
    """
    Estimate the skew angle of the image using the Hough Line Transform.

    Args:
        binary:           Binary image (uint8, 0/255).
        angle_threshold:  Only consider lines within this many degrees of
                          horizontal (0°) or vertical (90°).

    Returns:
        Estimated skew angle in degrees. Positive = clockwise skew.
        Returns 0.0 if no dominant angle is found.
    """
    # Hough works on edges — use Canny on the binary image
    edges = cv2.Canny(binary, threshold1=50, threshold2=150, apertureSize=3)

    # Probabilistic Hough: faster and more robust for long wall lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=binary.shape[1] // 8,  # at least 1/8 of image width
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # vertical line → 90°, handle separately
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalise to [-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # Keep only near-horizontal lines (close to 0°)
        if abs(angle) <= angle_threshold:
            angles.append(angle)

    if not angles:
        return 0.0

    # Use the median to be robust against outliers
    return float(np.median(angles))


def correct_skew(
    img: np.ndarray,
    angle: Optional[float] = None,
    binary: Optional[np.ndarray] = None,
    background_color: int = 255,
) -> np.ndarray:
    """
    Rotate an image to correct for skew.

    Args:
        img:              Grayscale image to correct.
        angle:            Skew angle in degrees (provide this OR binary).
        binary:           Binary image used to auto-detect angle if angle=None.
        background_color: Fill value for borders created by rotation (0=black, 255=white).

    Returns:
        Rotated (deskewed) grayscale image, same size as input.
    """
    if img is None:
        raise ValueError("img must not be None.")

    if angle is None:
        if binary is None:
            raise ValueError("Provide either 'angle' or 'binary' for auto-detection.")
        angle = detect_skew_angle(binary)

    # If skew is negligible, skip rotation to avoid resampling artifacts
    if abs(angle) < 0.3:
        return img.copy()

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Rotation matrix — negate angle because OpenCV y-axis is flipped
    M = cv2.getRotationMatrix2D(center, -angle, scale=1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,
    )

    return rotated


def deskew(img: np.ndarray, binary: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convenience wrapper: detect angle and correct in one call.

    Args:
        img:    Original grayscale image.
        binary: Binary version used for angle detection.

    Returns:
        (corrected_image, detected_angle_degrees)
    """
    angle = detect_skew_angle(binary)
    corrected = correct_skew(img, angle=angle)
    return corrected, angle
