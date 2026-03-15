"""
predictor.py
------------
Runs YOLOv8 segmentation inference on new floor plan images.
Outputs structured predictions consumed by Phase 3 (geometry reconstruction).

Usage:
    from src.segmentation.predictor import FloorPlanPredictor

    predictor = FloorPlanPredictor("models/segmentation/best.pt")
    result = predictor.predict("outputs/plan_4_cleaned.png")
    print(result.walls)      # list of wall polygons
    print(result.rooms)      # list of room regions with class labels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Result data structures ────────────────────────────────────────────────────

@dataclass
class DetectedElement:
    """A single detected floor plan element."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]          # (x1, y1, x2, y2) in pixels
    mask: Optional[np.ndarray] = None        # Binary mask, same size as image
    polygon: Optional[list[tuple]] = None    # Contour polygon points


@dataclass
class SegmentationResult:
    """Full segmentation result for one floor plan image."""
    image_path: str
    image_shape: tuple[int, int]             # (H, W)
    elements: list[DetectedElement] = field(default_factory=list)

    # Convenience accessors by type
    @property
    def walls(self) -> list[DetectedElement]:
        return [e for e in self.elements if e.class_id in (0, 1)]  # outer + inner

    @property
    def doors(self) -> list[DetectedElement]:
        return [e for e in self.elements if e.class_id == 3]

    @property
    def windows(self) -> list[DetectedElement]:
        return [e for e in self.elements if e.class_id == 2]

    @property
    def rooms(self) -> list[DetectedElement]:
        room_ids = {6, 7, 8, 9, 10, 11}
        return [e for e in self.elements if e.class_id in room_ids]

    @property
    def summary(self) -> dict:
        from collections import Counter
        counts = Counter(e.class_name for e in self.elements)
        return {
            "total_elements": len(self.elements),
            "by_class": dict(counts),
            "image": self.image_path,
        }


# ── Predictor ─────────────────────────────────────────────────────────────────

class FloorPlanPredictor:
    """
    Runs YOLOv8 segmentation on cleaned floor plan images.

    Args:
        model_path:  Path to trained best.pt weights.
        confidence:  Minimum confidence threshold (0-1).
        iou:         NMS IoU threshold (0-1).
        device:      Inference device ('mps', 'cpu', '0'). Auto-detected if None.
    """

    # Class names matching the training dataset (0-indexed, background excluded)
    CLASS_NAMES = [
        "OuterWall", "InnerWall", "Window", "Door",
        "Stairs", "Railing", "Kitchen", "LivingRoom",
        "Bedroom", "Bathroom", "Corridor", "Balcony", "Garage",
    ]

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou: float = 0.45,
        device: Optional[str] = None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._model = None  # Lazy load

    def _load_model(self):
        """Lazy-load the YOLO model on first inference call."""
        if self._model is None:
            from ultralytics import YOLO
            from src.segmentation.trainer import get_best_device

            device = self.device or get_best_device()
            self._model = YOLO(str(self.model_path))
            print(f"Model loaded: {self.model_path.name} on {device}")
            self._device = device

    def predict(self, image_path: str) -> SegmentationResult:
        """
        Run segmentation on a single floor plan image.

        Args:
            image_path: Path to a preprocessed floor plan (Phase 1 output).

        Returns:
            SegmentationResult with all detected elements.
        """
        self._load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        raw = self._model.predict(
            source=str(image_path),
            conf=self.confidence,
            iou=self.iou,
            device=self._device,
            verbose=False,
            retina_masks=True,   # Higher quality masks
        )

        result = SegmentationResult(
            image_path=str(image_path),
            image_shape=(h, w),
        )

        if raw and raw[0].masks is not None:
            result.elements = self._parse_detections(raw[0], h, w)

        print(f"Detected {len(result.elements)} elements in {image_path.name}")
        for cls, count in result.summary["by_class"].items():
            print(f"  {cls}: {count}")

        return result

    def predict_batch(
        self, image_paths: list[str]
    ) -> list[SegmentationResult]:
        """Run prediction on multiple images."""
        self._load_model()
        results = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n── [{i}/{len(image_paths)}] {Path(path).name}")
            try:
                results.append(self.predict(path))
            except Exception as e:
                print(f"  ERROR: {e}")
        return results

    def _parse_detections(self, raw_result, img_h: int, img_w: int) -> list[DetectedElement]:
        """Convert raw YOLO output to DetectedElement list."""
        elements = []

        boxes = raw_result.boxes
        masks = raw_result.masks

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]

            # Get binary mask
            if masks is not None:
                mask_data = masks.data[i].cpu().numpy()
                # Resize mask to original image dimensions
                mask = cv2.resize(
                    (mask_data * 255).astype(np.uint8),
                    (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask = None

            # Extract polygon from mask
            polygon = self._mask_to_polygon(mask) if mask is not None else None

            class_name = (
                self.CLASS_NAMES[class_id]
                if class_id < len(self.CLASS_NAMES)
                else f"class_{class_id}"
            )

            elements.append(DetectedElement(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                mask=mask,
                polygon=polygon,
            ))

        return elements

    def _mask_to_polygon(
        self, mask: np.ndarray, epsilon_factor: float = 0.005
    ) -> Optional[list[tuple]]:
        """
        Convert binary mask to simplified polygon.

        Args:
            mask:           Binary mask (0/255).
            epsilon_factor: Contour approximation accuracy (fraction of perimeter).

        Returns:
            List of (x, y) pixel coordinates or None if no contour found.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        epsilon = epsilon_factor * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
