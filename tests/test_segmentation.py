"""
tests/test_segmentation.py
---------------------------
Unit tests for Phase 2 segmentation modules.

Tests that don't require model weights or the full dataset.

Run with:
    pytest tests/test_segmentation.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.segmentation.dataset import (
    parse_svg_annotations,
    write_yolo_labels,
    _parse_polygon_points,
    _rect_to_polygon,
    _path_to_polygon,
    CLASS_NAMES,
    SVG_CLASS_MAP,
    NUM_CLASSES,
)
from src.segmentation.predictor import (
    DetectedElement,
    SegmentationResult,
    FloorPlanPredictor,
)
from src.segmentation.visualizer import SegmentationVisualizer


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestDataset:

    def test_class_names_count(self):
        assert NUM_CLASSES == 14

    def test_class_map_has_valid_ids(self):
        for name, cid in SVG_CLASS_MAP.items():
            assert 1 <= cid <= NUM_CLASSES - 1, \
                f"Class '{name}' has out-of-range ID {cid}"

    def test_parse_polygon_points_basic(self):
        pts = _parse_polygon_points("10,20 30,40 50,60")
        assert pts == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]

    def test_parse_polygon_points_space_separated(self):
        pts = _parse_polygon_points("10 20 30 40 50 60")
        assert len(pts) == 3

    def test_parse_polygon_points_empty(self):
        pts = _parse_polygon_points("")
        assert pts == []

    def test_rect_to_polygon_basic(self):
        class FakeElem:
            def get(self, key, default=None):
                return {"x": "10", "y": "20", "width": "100", "height": "50"}.get(key, default)
        pts = _rect_to_polygon(FakeElem())
        assert len(pts) == 4
        assert (10.0, 20.0) in pts
        assert (110.0, 70.0) in pts

    def test_path_to_polygon_basic(self):
        d = "M 10 20 L 30 40 L 50 60 Z"
        pts = _path_to_polygon(d)
        assert len(pts) == 3
        assert (10.0, 20.0) in pts

    def test_write_yolo_labels(self, tmp_path):
        polygons = [
            {"class_id": 1, "polygon": [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]},
            {"class_id": 3, "polygon": [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6)]},
        ]
        out = tmp_path / "label.txt"
        write_yolo_labels(polygons, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        # Class IDs should be 0-indexed (subtract 1)
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("2 ")


# ── Predictor tests ───────────────────────────────────────────────────────────

class TestSegmentationResult:

    def _make_result(self) -> SegmentationResult:
        elements = [
            DetectedElement(0, "OuterWall", 0.9, (0, 0, 100, 100)),
            DetectedElement(1, "InnerWall", 0.8, (10, 10, 50, 50)),
            DetectedElement(3, "Door",      0.7, (20, 20, 40, 40)),
            DetectedElement(2, "Window",    0.6, (60, 60, 90, 90)),
            DetectedElement(8, "Bedroom",   0.85,(5,  5,  80, 80)),
        ]
        result = SegmentationResult(
            image_path="test.png",
            image_shape=(512, 512),
            elements=elements,
        )
        return result

    def test_walls_filter(self):
        r = self._make_result()
        assert len(r.walls) == 2
        assert all(e.class_name in ("OuterWall", "InnerWall") for e in r.walls)

    def test_doors_filter(self):
        r = self._make_result()
        assert len(r.doors) == 1

    def test_windows_filter(self):
        r = self._make_result()
        assert len(r.windows) == 1

    def test_rooms_filter(self):
        r = self._make_result()
        assert len(r.rooms) == 1
        assert r.rooms[0].class_name == "Bedroom"

    def test_summary_counts(self):
        r = self._make_result()
        s = r.summary
        assert s["total_elements"] == 5
        assert s["by_class"]["OuterWall"] == 1
        assert s["by_class"]["Door"] == 1

    def test_predictor_raises_if_model_missing(self):
        with pytest.raises(FileNotFoundError):
            FloorPlanPredictor("nonexistent/model.pt")


# ── Visualizer tests ──────────────────────────────────────────────────────────

class TestVisualizer:

    def test_draw_returns_bgr_image(self):
        viz = SegmentationVisualizer()
        img = np.full((256, 256, 3), 200, dtype=np.uint8)
        result = SegmentationResult("test.png", (256, 256), elements=[])
        out = viz.draw(img, result)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8

    def test_draw_from_grayscale(self):
        viz = SegmentationVisualizer()
        img = np.full((256, 256), 200, dtype=np.uint8)
        result = SegmentationResult("test.png", (256, 256), elements=[])
        out = viz.draw(img, result)
        assert len(out.shape) == 3  # should be converted to BGR

    def test_draw_with_elements(self):
        viz = SegmentationVisualizer(show_masks=False)
        img = np.full((256, 256, 3), 200, dtype=np.uint8)
        elem = DetectedElement(0, "OuterWall", 0.9, (10, 10, 100, 100))
        result = SegmentationResult("test.png", (256, 256), elements=[elem])
        out = viz.draw(img, result)
        # Should not crash, and image should be modified
        assert out is not None
        assert not np.array_equal(out, img)
