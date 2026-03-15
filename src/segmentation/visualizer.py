"""
visualizer.py
-------------
Draws segmentation results (masks, bounding boxes, labels) on floor plan images.
Useful for debugging and reviewing model predictions.

Usage:
    from src.segmentation.visualizer import SegmentationVisualizer

    viz = SegmentationVisualizer()
    annotated = viz.draw(image, result)
    viz.save(annotated, "outputs/annotated.png")
    viz.show(annotated)
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .predictor import SegmentationResult, DetectedElement


# ── Colour palette (BGR for OpenCV) — one per class ───────────────────────────

CLASS_COLORS = {
    "OuterWall":  (50,  50,  180),    # dark blue
    "InnerWall":  (100, 100, 220),    # medium blue
    "Window":     (255, 200,  50),    # amber
    "Door":       (50,  200, 200),    # teal
    "Stairs":     (160,  80, 200),    # purple
    "Railing":    (200, 140,  80),    # tan
    "Kitchen":    (50,  180,  80),    # green
    "LivingRoom": (80,  200, 255),    # sky blue
    "Bedroom":    (200, 100, 150),    # pink
    "Bathroom":   (100, 220, 200),    # mint
    "Corridor":   (180, 180,  80),    # olive
    "Balcony":    (255, 140,  50),    # orange
    "Garage":     (140, 140, 140),    # gray
}

DEFAULT_COLOR = (200, 200, 200)
MASK_ALPHA = 0.40   # Mask transparency
TEXT_SCALE = 0.45
TEXT_THICKNESS = 1


class SegmentationVisualizer:
    """
    Renders segmentation predictions over floor plan images.

    Args:
        mask_alpha:  Opacity for filled masks (0=transparent, 1=solid).
        show_boxes:  Draw bounding boxes.
        show_labels: Draw class name + confidence labels.
        show_masks:  Fill detected regions with transparent colour.
    """

    def __init__(
        self,
        mask_alpha: float = MASK_ALPHA,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_masks: bool = True,
    ):
        self.mask_alpha = mask_alpha
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_masks = show_masks

    def draw(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        min_confidence: float = 0.0,
    ) -> np.ndarray:
        """
        Draw segmentation results on the image.

        Args:
            image:          BGR image (H, W, 3) as numpy array.
            result:         SegmentationResult from FloorPlanPredictor.
            min_confidence: Only draw elements above this confidence.

        Returns:
            Annotated BGR image.
        """
        if len(image.shape) == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        # Draw masks first (behind boxes and labels)
        if self.show_masks:
            canvas = self._draw_masks(
                canvas,
                [e for e in result.elements if e.confidence >= min_confidence],
            )

        # Draw boxes and labels on top
        for elem in result.elements:
            if elem.confidence < min_confidence:
                continue
            color = CLASS_COLORS.get(elem.class_name, DEFAULT_COLOR)

            if self.show_boxes:
                self._draw_box(canvas, elem, color)

            if self.show_labels:
                self._draw_label(canvas, elem, color)

        # Draw legend
        canvas = self._draw_legend(canvas, result)

        return canvas

    def draw_from_path(
        self, image_path: str, result: SegmentationResult
    ) -> np.ndarray:
        """Load image from path and draw segmentation results."""
        img = cv2.imread(image_path)
        if img is None:
            raise IOError(f"Could not load image: {image_path}")
        return self.draw(img, result)

    def save(self, image: np.ndarray, output_path: str) -> None:
        """Save annotated image to disk."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"Saved annotation: {output_path}")

    def show(self, image: np.ndarray, title: str = "Segmentation Result") -> None:
        """Display image in an OpenCV window. Press any key to close."""
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_masks(
        self, canvas: np.ndarray, elements: list[DetectedElement]
    ) -> np.ndarray:
        """Overlay semi-transparent filled masks."""
        overlay = canvas.copy()
        for elem in elements:
            if elem.mask is None:
                continue
            color = CLASS_COLORS.get(elem.class_name, DEFAULT_COLOR)
            overlay[elem.mask > 0] = color
        return cv2.addWeighted(overlay, self.mask_alpha, canvas, 1 - self.mask_alpha, 0)

    def _draw_box(
        self, canvas: np.ndarray, elem: DetectedElement, color: tuple
    ) -> None:
        """Draw bounding box rectangle."""
        x1, y1, x2, y2 = elem.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

    def _draw_label(
        self, canvas: np.ndarray, elem: DetectedElement, color: tuple
    ) -> None:
        """Draw class name + confidence label above bounding box."""
        x1, y1 = elem.bbox[0], elem.bbox[1]
        label = f"{elem.class_name} {elem.confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)

        # Background pill
        pad = 3
        cv2.rectangle(
            canvas,
            (x1, max(0, y1 - th - 2 * pad)),
            (x1 + tw + 2 * pad, y1),
            color,
            thickness=-1,
        )
        # Text (white on coloured background)
        cv2.putText(
            canvas,
            label,
            (x1 + pad, max(th, y1 - pad)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    def _draw_legend(
        self, canvas: np.ndarray, result: SegmentationResult
    ) -> np.ndarray:
        """Draw a class legend in the bottom-left corner."""
        seen_classes = sorted(set(e.class_name for e in result.elements))
        if not seen_classes:
            return canvas

        box_size = 14
        padding = 6
        line_height = box_size + padding
        legend_w = 160
        legend_h = len(seen_classes) * line_height + padding * 2

        h, w = canvas.shape[:2]
        x0 = padding
        y0 = h - legend_h - padding

        # Semi-transparent legend background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, h - padding), (30, 30, 30), -1)
        canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

        for i, cls_name in enumerate(seen_classes):
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            y = y0 + padding + i * line_height
            cv2.rectangle(canvas, (x0 + padding, y), (x0 + padding + box_size, y + box_size), color, -1)
            cv2.putText(
                canvas,
                cls_name,
                (x0 + padding + box_size + 6, y + box_size - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return canvas
