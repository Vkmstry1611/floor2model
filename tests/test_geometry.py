"""
tests/test_geometry.py
----------------------
Unit tests for Phase 3 geometry reconstruction modules.

Run with:
    pytest tests/test_geometry.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry.wall_vectorizer import WallVectorizer, WallPolygon, VectorizationResult
from src.geometry.scale_estimator import ScaleEstimator, ScaleEstimate
from src.geometry.room_graph import RoomGraphBuilder, FloorPlanGraph, RoomNode


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_square_mask(size=256, rect=(50, 50, 150, 150)) -> np.ndarray:
    """Create a binary mask with a filled rectangle."""
    mask = np.zeros((size, size), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = 255
    return mask

def make_mock_vectorization(
    n_rooms=2, n_doors=1, n_walls=2
) -> VectorizationResult:
    """Create a mock VectorizationResult for testing."""
    from src.geometry.wall_vectorizer import WallPolygon, VectorizationResult

    def make_poly(class_id, class_name, pts, area):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
        return WallPolygon(
            class_id=class_id,
            class_name=class_name,
            points=pts,
            area=area,
            bbox=bbox,
        )

    rooms = [
        make_poly(7, "LivingRoom", [(10,10),(10,110),(110,110),(110,10)], 10000.0),
        make_poly(9, "Bedroom",    [(150,10),(150,110),(250,110),(250,10)], 10000.0),
    ]
    doors = [
        make_poly(3, "Door", [(105,50),(105,70),(120,70),(120,50)], 300.0),
    ]
    walls = [
        make_poly(0, "OuterWall", [(5,5),(5,255),(255,255),(255,5)], 5000.0),
        make_poly(1, "InnerWall", [(120,10),(120,110),(125,110),(125,10)], 1000.0),
    ]

    result = VectorizationResult(image_shape=(256, 256))
    result.rooms = rooms
    result.doors = doors
    result.walls = walls
    return result


# ── WallVectorizer tests ──────────────────────────────────────────────────────

class TestWallVectorizer:

    def test_extract_from_mask_returns_polygons(self):
        mask = make_square_mask()
        vectorizer = WallVectorizer(min_area=100)
        polys = vectorizer.extract_from_mask(mask, class_id=0,
                                             class_name="OuterWall")
        assert len(polys) >= 1
        assert polys[0].class_name == "OuterWall"
        assert polys[0].area > 100

    def test_polygon_has_minimum_points(self):
        mask = make_square_mask()
        vectorizer = WallVectorizer()
        polys = vectorizer.extract_from_mask(mask, 0, "OuterWall")
        for poly in polys:
            assert len(poly.points) >= 3

    def test_small_mask_filtered_out(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:105, 100:105] = 255  # tiny 5×5 blob
        vectorizer = WallVectorizer(min_area=500)
        polys = vectorizer.extract_from_mask(mask, 0, "OuterWall")
        assert len(polys) == 0

    def test_wall_polygon_is_wall(self):
        mask = make_square_mask()
        vectorizer = WallVectorizer()
        polys = vectorizer.extract_from_mask(mask, 0, "OuterWall")
        assert polys[0].is_wall is True

    def test_room_polygon_is_room(self):
        mask = make_square_mask()
        vectorizer = WallVectorizer()
        polys = vectorizer.extract_from_mask(mask, 7, "LivingRoom")
        assert polys[0].is_room is True

    def test_centroid_inside_bbox(self):
        mask = make_square_mask(rect=(50, 50, 200, 200))
        vectorizer = WallVectorizer()
        polys = vectorizer.extract_from_mask(mask, 0, "OuterWall")
        cx, cy = polys[0].centroid
        x, y, w, h = polys[0].bbox
        assert x <= cx <= x + w
        assert y <= cy <= y + h

    def test_draw_returns_bgr_image(self):
        img = np.full((256, 256, 3), 200, dtype=np.uint8)
        vec = make_mock_vectorization()
        vectorizer = WallVectorizer()
        out = vectorizer.draw(img, vec)
        assert out.shape == (256, 256, 3)


# ── ScaleEstimator tests ──────────────────────────────────────────────────────

class TestScaleEstimator:

    def test_fallback_returns_estimate(self):
        img = np.zeros((1024, 1024), dtype=np.uint8)
        estimator = ScaleEstimator(target_image_size=1024)
        scale = estimator.estimate(img)
        assert isinstance(scale, ScaleEstimate)
        assert scale.pixels_per_metre > 0
        assert scale.method == "fallback"

    def test_pixels_to_metres_conversion(self):
        scale = ScaleEstimate(
            pixels_per_metre=100.0,
            confidence=1.0,
            method="test",
        )
        assert scale.pixels_to_metres(100) == pytest.approx(1.0)
        assert scale.pixels_to_metres(250) == pytest.approx(2.5)

    def test_area_conversion(self):
        scale = ScaleEstimate(
            pixels_per_metre=100.0,
            confidence=1.0,
            method="test",
        )
        # 100×100 px = 1m × 1m = 1m²
        assert scale.area_px_to_m2(10000) == pytest.approx(1.0)

    def test_room_size_estimate(self):
        img = np.zeros((1024, 1024), dtype=np.uint8)
        estimator = ScaleEstimator()
        vec = make_mock_vectorization()
        # room area is 10000 px², LivingRoom prior is 15-40m² (mid 27.5m²)
        scale = estimator.estimate(img, vec)
        assert scale.pixels_per_metre > 0
        # Should use room_size or fallback
        assert scale.method in ("room_size", "fallback")

    def test_metres_per_pixel_is_reciprocal(self):
        scale = ScaleEstimate(50.0, 1.0, "test")
        assert scale.metres_per_pixel == pytest.approx(1 / 50.0)


# ── RoomGraphBuilder tests ────────────────────────────────────────────────────

class TestRoomGraphBuilder:

    def test_builds_nodes_from_rooms(self):
        vec = make_mock_vectorization(n_rooms=2)
        scale = ScaleEstimate(100.0, 1.0, "test")
        builder = RoomGraphBuilder(proximity_threshold=300)
        graph = builder.build(vec, scale)
        assert len(graph.nodes) == 2

    def test_node_has_area_m2(self):
        vec = make_mock_vectorization()
        scale = ScaleEstimate(100.0, 1.0, "test")
        builder = RoomGraphBuilder()
        graph = builder.build(vec, scale)
        for node in graph.nodes:
            assert node.area_m2 > 0

    def test_door_creates_edge(self):
        vec = make_mock_vectorization(n_rooms=2, n_doors=1)
        scale = ScaleEstimate(100.0, 1.0, "test")
        # Rooms at centroids ~(60,60) and ~(200,60), door at ~(112,60)
        # proximity_threshold=200 should connect them
        builder = RoomGraphBuilder(proximity_threshold=300, door_proximity=200)
        graph = builder.build(vec, scale)
        assert len(graph.edges) >= 1

    def test_get_node_returns_correct_node(self):
        vec = make_mock_vectorization()
        scale = ScaleEstimate(100.0, 1.0, "test")
        builder = RoomGraphBuilder()
        graph = builder.build(vec, scale)
        for node in graph.nodes:
            found = graph.get_node(node.node_id)
            assert found is not None
            assert found.node_id == node.node_id

    def test_to_dict_serializable(self):
        import json
        vec = make_mock_vectorization()
        scale = ScaleEstimate(100.0, 1.0, "test")
        builder = RoomGraphBuilder()
        graph = builder.build(vec, scale)
        d = graph.to_dict()
        json_str = json.dumps(d)
        assert "nodes" in d
        assert "edges" in d
        assert len(json_str) > 0

    def test_summary_has_expected_keys(self):
        vec = make_mock_vectorization()
        scale = ScaleEstimate(100.0, 1.0, "test")
        builder = RoomGraphBuilder()
        graph = builder.build(vec, scale)
        summary = graph.summary
        assert "rooms" in summary
        assert "connections" in summary
        assert "total_area_m2" in summary

    def test_no_duplicate_edges(self):
        vec = make_mock_vectorization()
        scale = ScaleEstimate(100.0, 1.0, "test")
        builder = RoomGraphBuilder(proximity_threshold=500)
        graph = builder.build(vec, scale)
        edge_pairs = set()
        for edge in graph.edges:
            pair = (min(edge.node_a, edge.node_b),
                    max(edge.node_a, edge.node_b))
            assert pair not in edge_pairs, "Duplicate edge found!"
            edge_pairs.add(pair)
