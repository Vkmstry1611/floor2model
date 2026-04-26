#!/usr/bin/env python3
"""
Simple test of detection refinement without requiring YOLO.

This creates synthetic test data to demonstrate the refinement module.
"""

import numpy as np
import cv2
from pathlib import Path

from src.geometry.wall_vectorizer import WallPolygon, VectorizationResult
from src.segmentation.predictor import SegmentationResult
from src.detection.refinement import refine_detections, RefinementConfig
from src.utils.visualization import visualize_vectorization_result, visualize_comparison


def create_synthetic_floor_plan():
    """Create a synthetic floor plan image."""
    # Create blank image
    image = np.ones((600, 800), dtype=np.uint8) * 255
    
    # Draw walls (black lines)
    cv2.rectangle(image, (100, 100), (700, 500), 0, 3)  # Outer walls
    cv2.line(image, (400, 100), (400, 500), 0, 3)  # Middle wall
    
    # Draw some window-like features (parallel lines)
    cv2.line(image, (200, 98), (250, 98), 0, 2)
    cv2.line(image, (200, 102), (250, 102), 0, 2)
    
    cv2.line(image, (550, 98), (600, 98), 0, 2)
    cv2.line(image, (550, 102), (600, 102), 0, 2)
    
    return image


def create_test_vectorization():
    """Create test vectorization result."""
    # Walls
    walls = [
        WallPolygon(
            class_id=1,
            class_name="InnerWall",
            points=[(100, 100), (700, 100), (700, 110), (100, 110)],
            area=6000.0,
            bbox=(100, 100, 600, 10),
            confidence=1.0
        ),
        WallPolygon(
            class_id=1,
            class_name="InnerWall",
            points=[(100, 490), (700, 490), (700, 500), (100, 500)],
            area=6000.0,
            bbox=(100, 490, 600, 10),
            confidence=1.0
        ),
        WallPolygon(
            class_id=1,
            class_name="InnerWall",
            points=[(100, 100), (110, 100), (110, 500), (100, 500)],
            area=4000.0,
            bbox=(100, 100, 10, 400),
            confidence=1.0
        ),
        WallPolygon(
            class_id=1,
            class_name="InnerWall",
            points=[(690, 100), (700, 100), (700, 500), (690, 500)],
            area=4000.0,
            bbox=(690, 100, 10, 400),
            confidence=1.0
        ),
        WallPolygon(
            class_id=1,
            class_name="InnerWall",
            points=[(395, 100), (405, 100), (405, 500), (395, 500)],
            area=4000.0,
            bbox=(395, 100, 10, 400),
            confidence=1.0
        ),
    ]
    
    # Rooms
    rooms = [
        WallPolygon(
            class_id=7,
            class_name="LivingRoom",
            points=[(110, 110), (390, 110), (390, 490), (110, 490)],
            area=106400.0,
            bbox=(110, 110, 280, 380),
            confidence=1.0
        ),
        WallPolygon(
            class_id=8,
            class_name="Bedroom",
            points=[(410, 110), (690, 110), (690, 490), (410, 490)],
            area=106400.0,
            bbox=(410, 110, 280, 380),
            confidence=1.0
        ),
    ]
    
    # No doors or windows initially (will be detected by refinement)
    doors = []
    windows = []
    
    return VectorizationResult(
        walls=walls,
        rooms=rooms,
        doors=doors,
        windows=windows,
        other=[],
        image_shape=(600, 800)
    )


def main():
    """Run the test."""
    print("=" * 70)
    print("Detection Refinement - Simple Test")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("outputs/test_refinement")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create synthetic data
    print("\n[1/4] Creating synthetic floor plan...")
    image = create_synthetic_floor_plan()
    cv2.imwrite(str(output_dir / "synthetic_floorplan.png"), image)
    print(f"  ✓ Synthetic floor plan created")
    print(f"    Image shape: {image.shape}")
    
    # Step 2: Create test vectorization
    print("\n[2/4] Creating test vectorization...")
    vec_before = create_test_vectorization()
    print(f"  ✓ Test vectorization created")
    print(f"    Walls: {len(vec_before.walls)}")
    print(f"    Rooms: {len(vec_before.rooms)}")
    print(f"    Doors: {len(vec_before.doors)} (before refinement)")
    print(f"    Windows: {len(vec_before.windows)} (before refinement)")
    
    # Step 3: Apply refinement
    print("\n[3/4] Applying detection refinement...")
    
    # Create dummy segmentation result
    seg_result = SegmentationResult(
        image_path="synthetic",
        image_shape=(600, 800),
        elements=[]
    )
    
    # Configure refinement
    refinement_config = RefinementConfig(
        door_width=90,
        wall_gap_threshold=20,
        room_adjacency_threshold=50,
        canny_low_threshold=50,
        canny_high_threshold=150,
        enable_door_detection=True,
        enable_window_detection=True
    )
    
    # Run refinement
    vec_after = refine_detections(
        yolo_output=seg_result,
        image=image,
        geometry_output=vec_before,
        config=refinement_config
    )
    
    print(f"  ✓ Refinement complete")
    print(f"    Doors: {len(vec_after.doors)} (after refinement)")
    print(f"    Windows: {len(vec_after.windows)} (after refinement)")
    
    # Step 4: Create visualizations
    print("\n[4/4] Creating visualizations...")
    
    # Before refinement
    viz_before = visualize_vectorization_result(
        image,
        vec_before,
        output_path=str(output_dir / "before_refinement.png"),
        show_labels=False
    )
    print(f"  ✓ Before visualization saved")
    
    # After refinement
    viz_after = visualize_vectorization_result(
        image,
        vec_after,
        output_path=str(output_dir / "after_refinement.png"),
        show_labels=True
    )
    print(f"  ✓ After visualization saved")
    
    # Comparison
    detections_before = {
        "walls": [np.array(w.points) for w in vec_before.walls],
        "rooms": [np.array(r.points) for r in vec_before.rooms],
        "doors": [np.array(d.points) for d in vec_before.doors],
        "windows": [np.array(w.points) for w in vec_before.windows]
    }
    detections_after = {
        "walls": [np.array(w.points) for w in vec_after.walls],
        "rooms": [np.array(r.points) for r in vec_after.rooms],
        "doors": [np.array(d.points) for d in vec_after.doors],
        "windows": [np.array(w.points) for w in vec_after.windows]
    }
    
    from src.utils.visualization import visualize_comparison
    comparison = visualize_comparison(
        image,
        detections_before,
        detections_after,
        output_path=str(output_dir / "comparison.png")
    )
    print(f"  ✓ Comparison saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}/")
    print(f"\n📊 Results:")
    print(f"  Walls: {len(vec_after.walls)} (preserved)")
    print(f"  Rooms: {len(vec_after.rooms)} (preserved)")
    print(f"  Doors: {len(vec_before.doors)} → {len(vec_after.doors)} (refined)")
    print(f"  Windows: {len(vec_before.windows)} → {len(vec_after.windows)} (refined)")
    
    print(f"\n🎨 Generated files:")
    print(f"  - synthetic_floorplan.png (input)")
    print(f"  - before_refinement.png (YOLO detections)")
    print(f"  - after_refinement.png (refined detections)")
    print(f"  - comparison.png (side-by-side)")
    
    print(f"\n✅ Open comparison.png to see the results!")
    print(f"   {output_dir / 'comparison.png'}")
    
    return 0


if __name__ == "__main__":
    exit(main())
