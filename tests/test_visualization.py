"""
Test script for visualization utilities.

This script demonstrates the visualization functions with synthetic data.
"""

import numpy as np
import cv2
from src.utils.visualization import (
    visualize_detections,
    visualize_comparison,
    create_detection_report
)


def create_synthetic_floor_plan():
    """Create a synthetic floor plan image."""
    # Create blank image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw some walls (black lines)
    cv2.rectangle(image, (100, 100), (700, 500), (0, 0, 0), 3)
    cv2.line(image, (400, 100), (400, 500), (0, 0, 0), 3)
    
    return image


def create_test_detections():
    """Create test detection data."""
    # Walls
    walls = [
        np.array([[100, 100], [700, 100], [700, 110], [100, 110]]),  # Top wall
        np.array([[100, 490], [700, 490], [700, 500], [100, 500]]),  # Bottom wall
        np.array([[100, 100], [110, 100], [110, 500], [100, 500]]),  # Left wall
        np.array([[690, 100], [700, 100], [700, 500], [690, 500]]),  # Right wall
        np.array([[395, 100], [405, 100], [405, 500], [395, 500]]),  # Middle wall
    ]
    
    # Rooms
    rooms = [
        np.array([[110, 110], [390, 110], [390, 490], [110, 490]]),  # Left room
        np.array([[410, 110], [690, 110], [690, 490], [410, 490]]),  # Right room
    ]
    
    # Doors
    doors = [
        np.array([[395, 280], [405, 280], [405, 320], [395, 320]]),  # Door in middle wall
    ]
    
    # Windows
    windows = [
        np.array([[200, 100], [250, 100], [250, 110], [200, 110]]),  # Window in top wall
        np.array([[550, 100], [600, 100], [600, 110], [550, 110]]),  # Window in top wall
    ]
    
    return {
        "walls": walls,
        "rooms": rooms,
        "doors": doors,
        "windows": windows
    }


def test_basic_visualization():
    """Test basic visualization function."""
    print("\n=== Testing Basic Visualization ===")
    
    image = create_synthetic_floor_plan()
    detections = create_test_detections()
    
    # Create visualization
    viz = visualize_detections(
        image,
        detections,
        output_path="outputs/test_viz_basic.png",
        show_labels=True
    )
    
    print(f"✓ Basic visualization created")
    print(f"  Output: outputs/test_viz_basic.png")
    print(f"  Size: {viz.shape}")


def test_comparison_visualization():
    """Test comparison visualization."""
    print("\n=== Testing Comparison Visualization ===")
    
    image = create_synthetic_floor_plan()
    
    # Before: fewer detections
    detections_before = create_test_detections()
    detections_before["doors"] = []  # No doors detected
    detections_before["windows"] = detections_before["windows"][:1]  # Only 1 window
    
    # After: more detections
    detections_after = create_test_detections()
    
    # Create comparison
    comparison = visualize_comparison(
        image,
        detections_before,
        detections_after,
        output_path="outputs/test_viz_comparison.png"
    )
    
    print(f"✓ Comparison visualization created")
    print(f"  Output: outputs/test_viz_comparison.png")
    print(f"  Size: {comparison.shape}")


def test_detection_report():
    """Test detection report creation."""
    print("\n=== Testing Detection Report ===")
    
    image = create_synthetic_floor_plan()
    detections = create_test_detections()
    
    # Create report
    create_detection_report(
        image,
        detections,
        output_path="outputs/test_viz_report.png",
        title="Test Detection Report"
    )
    
    print(f"✓ Detection report created")
    print(f"  Output: outputs/test_viz_report.png")


def test_no_labels():
    """Test visualization without labels."""
    print("\n=== Testing Visualization Without Labels ===")
    
    image = create_synthetic_floor_plan()
    detections = create_test_detections()
    
    # Create visualization without labels
    viz = visualize_detections(
        image,
        detections,
        output_path="outputs/test_viz_no_labels.png",
        show_labels=False
    )
    
    print(f"✓ Visualization without labels created")
    print(f"  Output: outputs/test_viz_no_labels.png")


def test_custom_transparency():
    """Test visualization with custom room transparency."""
    print("\n=== Testing Custom Transparency ===")
    
    image = create_synthetic_floor_plan()
    detections = create_test_detections()
    
    # Create visualization with high transparency
    viz = visualize_detections(
        image,
        detections,
        output_path="outputs/test_viz_transparent.png",
        show_labels=True,
        room_alpha=0.1  # Very transparent
    )
    
    print(f"✓ Visualization with custom transparency created")
    print(f"  Output: outputs/test_viz_transparent.png")


def test_empty_detections():
    """Test visualization with empty detections."""
    print("\n=== Testing Empty Detections ===")
    
    image = create_synthetic_floor_plan()
    detections = {
        "walls": [],
        "rooms": [],
        "doors": [],
        "windows": []
    }
    
    # Create visualization
    viz = visualize_detections(
        image,
        detections,
        output_path="outputs/test_viz_empty.png"
    )
    
    print(f"✓ Visualization with empty detections created")
    print(f"  Output: outputs/test_viz_empty.png")


def main():
    """Run all visualization tests."""
    print("=" * 60)
    print("Visualization Utilities - Test Suite")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs("outputs", exist_ok=True)
    
    try:
        test_basic_visualization()
        test_comparison_visualization()
        test_detection_report()
        test_no_labels()
        test_custom_transparency()
        test_empty_detections()
        
        print("\n" + "=" * 60)
        print("✓ All visualization tests passed!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - outputs/test_viz_basic.png")
        print("  - outputs/test_viz_comparison.png")
        print("  - outputs/test_viz_report.png")
        print("  - outputs/test_viz_no_labels.png")
        print("  - outputs/test_viz_transparent.png")
        print("  - outputs/test_viz_empty.png")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
