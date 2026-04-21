"""
wall_vectorizer.py
------------------
Converts YOLOv8 segmentation masks into clean 2D wall polygons.

Pipeline per mask:
  1. Binarize mask
  2. Morphological cleanup (close gaps, remove noise)
  3. Find contours
  4. Approximate contours to simplified polygons (Douglas-Peucker)
  5. Filter by area and aspect ratio
  6. Return list of WallPolygon objects

Usage:
    from src.geometry.wall_vectorizer import WallVectorizer

    vectorizer = WallVectorizer()
    walls = vectorizer.extract(segmentation_result, image_shape)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WallPolygon:
    """A single vectorized wall or room boundary polygon."""
    class_id:   int
    class_name: str
    points:     list[tuple[int, int]]    # pixel coordinates (x, y)
    area:       float                    # pixel area
    bbox:       tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 1.0

    @property
    def is_wall(self) -> bool:
        return self.class_id in (0, 1)  # OuterWall, InnerWall

    @property
    def is_room(self) -> bool:
        return self.class_id in (6, 7, 8, 9, 10, 11, 12)

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.points:
            return (0.0, 0.0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def to_numpy(self) -> np.ndarray:
        """Return points as (N, 2) numpy array."""
        return np.array(self.points, dtype=np.int32)


@dataclass
class VectorizationResult:
    """All vectorized elements from one floor plan."""
    walls:    list[WallPolygon] = field(default_factory=list)
    rooms:    list[WallPolygon] = field(default_factory=list)
    doors:    list[WallPolygon] = field(default_factory=list)
    windows:  list[WallPolygon] = field(default_factory=list)
    other:    list[WallPolygon] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)

    @property
    def all_polygons(self) -> list[WallPolygon]:
        return self.walls + self.rooms + self.doors + self.windows + self.other

    @property
    def summary(self) -> dict:
        return {
            "walls":   len(self.walls),
            "rooms":   len(self.rooms),
            "doors":   len(self.doors),
            "windows": len(self.windows),
            "other":   len(self.other),
            "total":   len(self.all_polygons),
        }


# ── Vectorizer ────────────────────────────────────────────────────────────────

class WallVectorizer:
    """
    Converts segmentation masks into clean 2D vector polygons.

    Args:
        epsilon_factor:   Douglas-Peucker approximation factor
                          (fraction of arc length). Lower = more detail.
        min_area:         Discard polygons smaller than this (px²).
        morph_kernel:     Kernel size for morphological cleanup.
        simplify_walls:   Extra simplification pass for wall polygons.
    """

    # Which class_ids map to which category (0-indexed, background excluded)
    WALL_IDS    = {0, 1}          # OuterWall, InnerWall
    DOOR_IDS    = {3}             # Door
    WINDOW_IDS  = {2}             # Window
    ROOM_IDS    = {6, 7, 8, 9, 10, 11, 12}  # room types

    CLASS_NAMES = [
        "OuterWall", "InnerWall", "Window", "Door", "Stairs",
        "Railing", "Kitchen", "LivingRoom", "Bedroom", "Bathroom",
        "Corridor", "Balcony", "Garage",
    ]

    def __init__(
        self,
        epsilon_factor: float = 0.008,
        min_area: int = 200,
        morph_kernel: int = 3,
        simplify_walls: bool = True,
    ):
        self.epsilon_factor = epsilon_factor
        self.min_area = min_area
        self.morph_kernel = morph_kernel
        self.simplify_walls = simplify_walls

    def extract(
        self,
        segmentation_result,
        image_shape: Optional[tuple] = None,
    ) -> VectorizationResult:
        """
        Extract vector polygons from a SegmentationResult (Phase 2 output).

        Args:
            segmentation_result: FloorPlanPredictor result object.
            image_shape:         (H, W) of the source image.

        Returns:
            VectorizationResult with categorized polygons.
        """
        if image_shape is None:
            image_shape = segmentation_result.image_shape

        h, w = image_shape[:2]
        result = VectorizationResult(image_shape=(h, w))

        for element in segmentation_result.elements:
            if element.mask is None:
                continue

            polygons = self._mask_to_polygons(
                mask=element.mask,
                class_id=element.class_id,
                class_name=element.class_name,
                confidence=element.confidence,
                is_wall=(element.class_id in self.WALL_IDS),
            )

            for poly in polygons:
                if poly.class_id in self.WALL_IDS:
                    result.walls.append(poly)
                elif poly.class_id in self.DOOR_IDS:
                    result.doors.append(poly)
                elif poly.class_id in self.WINDOW_IDS:
                    result.windows.append(poly)
                elif poly.class_id in self.ROOM_IDS:
                    result.rooms.append(poly)
                else:
                    result.other.append(poly)

        return result

    def extract_from_mask(
        self,
        mask: np.ndarray,
        class_id: int,
        class_name: str,
        confidence: float = 1.0,
    ) -> list[WallPolygon]:
        """
        Extract polygons directly from a binary mask array.
        Useful for testing without a full SegmentationResult.
        """
        return self._mask_to_polygons(
            mask=mask,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            is_wall=(class_id in self.WALL_IDS),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mask_to_polygons(
        self,
        mask: np.ndarray,
        class_id: int,
        class_name: str,
        confidence: float,
        is_wall: bool,
    ) -> list[WallPolygon]:
        """Convert a binary mask to a list of simplified polygons."""

        # Ensure binary uint8
        binary = (mask > 127).astype(np.uint8) * 255

        # Morphological cleanup
        k = self.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find external contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Douglas-Peucker simplification
            epsilon = self.epsilon_factor * cv2.arcLength(contour, closed=True)

            # Walls get extra simplification to remove noise
            if is_wall and self.simplify_walls:
                epsilon *= 1.5

            approx = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Need at least 3 points for a valid polygon
            if len(approx) < 3:
                continue

            points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
            x, y, w, h = cv2.boundingRect(contour)

            polygons.append(WallPolygon(
                class_id=class_id,
                class_name=class_name,
                points=points,
                area=float(area),
                bbox=(x, y, w, h),
                confidence=confidence,
            ))

        # Sort by area descending (largest first)
        polygons.sort(key=lambda p: p.area, reverse=True)
        return polygons

    def draw(
        self,
        image: np.ndarray,
        result: VectorizationResult,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Draw vectorized polygons on an image for visualization.

        Returns annotated BGR image.
        """
        if len(image.shape) == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        colors = {
            "wall":    (50,  50,  200),
            "door":    (50,  200, 200),
            "window":  (200, 180,  50),
            "room":    (50,  180,  80),
            "other":   (150, 150, 150),
        }

        def draw_poly(polys, color, label_prefix=""):
            for poly in polys:
                pts = np.array(poly.points, dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=True,
                              color=color, thickness=2)
                if draw_labels:
                    cx, cy = int(poly.centroid[0]), int(poly.centroid[1])
                    cv2.putText(canvas, poly.class_name,
                                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, color, 1, cv2.LINE_AA)

        draw_poly(result.walls,   colors["wall"])
        draw_poly(result.doors,   colors["door"])
        draw_poly(result.windows, colors["window"])
        draw_poly(result.rooms,   colors["room"])
        draw_poly(result.other,   colors["other"])

        return canvas
