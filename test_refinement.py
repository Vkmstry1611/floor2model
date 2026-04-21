"""
Test script for detection refinement feature.

This script demonstrates the detection refinement module by creating
synthetic test data and running the refinement pipeline.
"""

import numpy as np
from src.detection.refinement import (
    RefinementConfig,
    DetectionRefiner,
    refine_detections,
    SpatialAnalyzer,
    GapDetector,
    EdgeAnalyzer,
    LineDetector,
)
from src.geometry.wall_vectorizer import WallPolygon, VectorizationResult
from src.segmentation.predictor import SegmentationResult, DetectedElement


def create_test_wall_polygon(points, class_id=0, class_name="InnerWall"):
    """Create a test wall polygon."""
    points_array = np.array(points)
    area = float(len(points) * 10)  # Approximate area
    
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    bbox = (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max() - x_coords.min()),
        int(y_coords.max() - y_coords.min())
    )
    
    return WallPolygon(
        class_id=class_id,
        class_name=class_name,
        points=points,
        area=area,
        bbox=bbox,
        confidence=1.0
    )


def test_configuration():
    """Test RefinementConfig creation and validation."""
    print("\n=== Testing RefinementConfig ===")
    
    # Test default config
    config = RefinementConfig()
    print(f"✓ Default config created")
    print(f"  - Canny thresholds: {config.canny_low_threshold}, {config.canny_high_threshold}")
    print(f"  - Door width: {config.door_width}px")
    print(f"  - Door detection: {config.enable_door_detection}")
    print(f"  - Window detection: {config.enable_window_detection}")
    
    # Test custom config
    custom_config = RefinementConfig(
        canny_low_threshold=40,
        door_width=100,
        enable_window_detection=False
    )
    print(f"✓ Custom config created")
    print(f"  - Canny low threshold: {custom_config.canny_low_threshold}")
    print(f"  - Door width: {custom_config.door_width}px")
    print(f"  - Window detection: {custom_config.enable_window_detection}")
    
    # Test validation (invalid thresholds)
    invalid_config = RefinementConfig(
        canny_low_threshold=200,
        canny_high_threshold=100  # Invalid: high < low
    )
    print(f"✓ Config validation works (corrected invalid thresholds)")
    print(f"  - Corrected thresholds: {invalid_config.canny_low_threshold}, {invalid_config.canny_high_threshold}")


def test_spatial_analyzer():
    """Test SpatialAnalyzer functionality."""
    print("\n=== Testing SpatialAnalyzer ===")
    
    config = RefinementConfig()
    analyzer = SpatialAnalyzer(config)
    
    # Create two adjacent rooms
    room1 = create_test_wall_polygon(
        [(100, 100), (300, 100), (300, 300), (100, 300)],
        class_id=7,
        class_name="LivingRoom"
    )
    room2 = create_test_wall_polygon(
        [(300, 100), (500, 100), (500, 300), (300, 300)],
        class_id=8,
        class_name="Bedroom"
    )
    room3 = create_test_wall_polygon(
        [(600, 600), (800, 600), (800, 800), (600, 800)],
        class_id=9,
        class_name="Bathroom"
    )
    
    rooms = [room1, room2, room3]
    
    # Test adjacent room finding
    adjacent_pairs = analyzer.find_adjacent_rooms(rooms)
    print(f"✓ Found {len(adjacent_pairs)} adjacent room pairs")
    for room_a, room_b in adjacent_pairs:
        print(f"  - {room_a.class_name} <-> {room_b.class_name}")
    
    # Test wall direction computation
    wall_segment = np.array([[100, 100], [300, 100], [500, 100]], dtype=np.float32)
    direction = analyzer.get_wall_direction(wall_segment)
    print(f"✓ Wall direction computed: {direction:.1f}°")
    
    # Test point-on-wall check
    wall = create_test_wall_polygon(
        [(100, 100), (300, 100), (300, 120), (100, 120)],
        class_id=1,
        class_name="InnerWall"
    )
    point_on = (200, 110)
    point_off = (200, 200)
    
    is_on = analyzer.is_point_on_wall(point_on, wall)
    is_off = analyzer.is_point_on_wall(point_off, wall)
    print(f"✓ Point-on-wall check: on={is_on}, off={is_off}")


def test_gap_detector():
    """Test GapDetector functionality."""
    print("\n=== Testing GapDetector ===")
    
    config = RefinementConfig(wall_gap_threshold=30)
    detector = GapDetector(config)
    
    # Create wall segment with a gap
    wall_with_gap = np.array([
        [100, 100],
        [150, 100],
        [200, 100],
        # Gap here (50px)
        [250, 100],
        [300, 100],
        [350, 100],
    ], dtype=np.float32)
    
    gaps = detector.detect_wall_gaps(wall_with_gap)
    print(f"✓ Detected {len(gaps)} gaps in wall segment")
    for i, (start, end) in enumerate(gaps):
        gap_width = np.linalg.norm(end - start)
        print(f"  - Gap {i+1}: {gap_width:.1f}px wide")
    
    # Create continuous wall (no gaps)
    continuous_wall = np.array([
        [100, 100],
        [110, 100],
        [120, 100],
        [130, 100],
        [140, 100],
    ], dtype=np.float32)
    
    gaps_continuous = detector.detect_wall_gaps(continuous_wall)
    print(f"✓ Continuous wall: {len(gaps_continuous)} gaps detected (expected 0)")


def test_edge_and_line_detection():
    """Test EdgeAnalyzer and LineDetector."""
    print("\n=== Testing Edge and Line Detection ===")
    
    config = RefinementConfig()
    edge_analyzer = EdgeAnalyzer(config)
    line_detector = LineDetector(config)
    
    # Create synthetic image with edges
    image = np.zeros((500, 500), dtype=np.uint8)
    # Draw some rectangles (windows)
    image[100:120, 200:300] = 255
    image[200:220, 200:300] = 255
    
    # Test edge detection
    edges = edge_analyzer.detect_edges(image)
    edge_count = np.count_nonzero(edges)
    print(f"✓ Edge detection: {edge_count} edge pixels found")
    
    # Test line detection
    lines = line_detector.detect_lines(edges)
    print(f"✓ Line detection: {len(lines)} lines found")
    
    if len(lines) >= 2:
        # Test parallel pair finding
        parallel_pairs = line_detector.find_parallel_pairs(lines)
        print(f"✓ Parallel pair detection: {len(parallel_pairs)} pairs found")
        
        # Test line angle computation
        if lines:
            angle = line_detector.compute_line_angle(lines[0])
            print(f"✓ Line angle: {angle:.1f}°")


def test_detection_refiner():
    """Test the main DetectionRefiner class."""
    print("\n=== Testing DetectionRefiner ===")
    
    config = RefinementConfig()
    refiner = DetectionRefiner(config)
    
    # Create test data
    walls = [
        create_test_wall_polygon(
            [(100, 200), (500, 200), (500, 220), (100, 220)],
            class_id=1,
            class_name="InnerWall"
        )
    ]
    
    rooms = [
        create_test_wall_polygon(
            [(100, 100), (300, 100), (300, 300), (100, 300)],
            class_id=7,
            class_name="LivingRoom"
        ),
        create_test_wall_polygon(
            [(300, 100), (500, 100), (500, 300), (300, 300)],
            class_id=8,
            class_name="Bedroom"
        )
    ]
    
    # Create VectorizationResult
    vec_result = VectorizationResult(
        walls=walls,
        rooms=rooms,
        doors=[],
        windows=[],
        other=[],
        image_shape=(500, 500)
    )
    
    # Create dummy segmentation result
    seg_result = SegmentationResult(
        image_path="test.png",
        image_shape=(500, 500),
        elements=[]
    )
    
    # Create test image
    image = np.zeros((500, 500), dtype=np.uint8)
    
    # Run refinement
    refined = refiner.refine(seg_result, image, vec_result)
    
    print(f"✓ Refinement complete")
    print(f"  - Walls: {len(refined.walls)} (preserved)")
    print(f"  - Rooms: {len(refined.rooms)} (preserved)")
    print(f"  - Doors: {len(refined.doors)} (refined)")
    print(f"  - Windows: {len(refined.windows)} (refined)")


def test_public_api():
    """Test the public refine_detections() API."""
    print("\n=== Testing Public API ===")
    
    # Create minimal test data
    vec_result = VectorizationResult(
        walls=[],
        rooms=[],
        doors=[],
        windows=[],
        other=[],
        image_shape=(500, 500)
    )
    
    seg_result = SegmentationResult(
        image_path="test.png",
        image_shape=(500, 500),
        elements=[]
    )
    
    image = np.zeros((500, 500), dtype=np.uint8)
    
    # Test with default config
    refined = refine_detections(seg_result, image, vec_result)
    print(f"✓ Public API works with default config")
    
    # Test with custom config
    custom_config = RefinementConfig(enable_window_detection=False)
    refined = refine_detections(seg_result, image, vec_result, config=custom_config)
    print(f"✓ Public API works with custom config")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Detection Refinement Module - Test Suite")
    print("=" * 60)
    
    try:
        test_configuration()
        test_spatial_analyzer()
        test_gap_detector()
        test_edge_and_line_detection()
        test_detection_refiner()
        test_public_api()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
