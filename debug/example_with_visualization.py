"""
Example: Using detection refinement with visualization.

This script demonstrates how to:
1. Run the geometry pipeline with refinement enabled
2. Enable debug visualizations
3. Create custom visualizations
"""

import numpy as np
from pathlib import Path

# Import pipeline components
from src.preprocessing.pipeline import PreprocessingPipeline
from src.segmentation.predictor import FloorPlanPredictor
from src.geometry.pipeline import GeometryPipeline, GeometryConfig
from src.detection.refinement import RefinementConfig
from src.utils.visualization import (
    visualize_vectorization_result,
    visualize_comparison,
    create_detection_report
)


def example_1_basic_with_debug():
    """
    Example 1: Run pipeline with debug visualization enabled.
    
    This will automatically create before/after comparison images.
    """
    print("\n" + "=" * 60)
    print("Example 1: Pipeline with Debug Visualization")
    print("=" * 60)
    
    # Configure geometry pipeline with debug visualization
    geo_config = GeometryConfig(
        use_refinement=True,
        debug_visualization=True,  # Enable debug visualizations
        visualization_dir="outputs/debug_viz"
    )
    
    # Create pipeline
    geo_pipeline = GeometryPipeline(config=geo_config)
    
    print("\nConfiguration:")
    print(f"  Refinement enabled: {geo_config.use_refinement}")
    print(f"  Debug visualization: {geo_config.debug_visualization}")
    print(f"  Output directory: {geo_config.visualization_dir}")
    
    print("\nTo use this configuration:")
    print("  1. Run preprocessing: prep_pipeline.run('input.png')")
    print("  2. Run segmentation: predictor.predict(cleaned_image)")
    print("  3. Run geometry: geo_pipeline.run(seg_result, image, 'input.png')")
    print("\nDebug visualizations will be automatically saved to:")
    print(f"  - {geo_config.visualization_dir}/input_before_refinement.png")
    print(f"  - {geo_config.visualization_dir}/input_after_refinement.png")
    print(f"  - {geo_config.visualization_dir}/input_comparison.png")


def example_2_custom_refinement_config():
    """
    Example 2: Custom refinement configuration with visualization.
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Refinement Configuration")
    print("=" * 60)
    
    # Create custom refinement config
    refinement_config = RefinementConfig(
        # Edge detection parameters
        canny_low_threshold=40,
        canny_high_threshold=160,
        
        # Door detection parameters
        door_width=100,
        wall_gap_threshold=25,
        room_adjacency_threshold=60,
        
        # Window detection parameters
        parallel_line_proximity=25,
        parallel_angle_tolerance=20.0,
        window_min_length=40,
        
        # Enable/disable specific detections
        enable_door_detection=True,
        enable_window_detection=True
    )
    
    # Create geometry config with custom refinement
    geo_config = GeometryConfig(
        use_refinement=True,
        refinement_config=refinement_config,
        debug_visualization=True
    )
    
    print("\nCustom Refinement Parameters:")
    print(f"  Canny thresholds: {refinement_config.canny_low_threshold}, {refinement_config.canny_high_threshold}")
    print(f"  Door width: {refinement_config.door_width}px")
    print(f"  Wall gap threshold: {refinement_config.wall_gap_threshold}px")
    print(f"  Room adjacency: {refinement_config.room_adjacency_threshold}px")
    print(f"  Window min length: {refinement_config.window_min_length}px")


def example_3_manual_visualization():
    """
    Example 3: Create custom visualizations manually.
    """
    print("\n" + "=" * 60)
    print("Example 3: Manual Visualization")
    print("=" * 60)
    
    print("\nAfter running the pipeline, you can create custom visualizations:")
    print("""
# Get the result
geo_result = geo_pipeline.run(seg_result, image, 'floorplan.png')

# Method 1: Visualize the vectorization result
from src.utils.visualization import visualize_vectorization_result

viz = visualize_vectorization_result(
    image,
    geo_result.vectorization,
    output_path='outputs/my_visualization.png',
    show_labels=True
)

# Method 2: Create a detection report with statistics
from src.utils.visualization import create_detection_report

detections = {
    "walls": [np.array(w.points) for w in geo_result.vectorization.walls],
    "rooms": [np.array(r.points) for r in geo_result.vectorization.rooms],
    "doors": [np.array(d.points) for d in geo_result.vectorization.doors],
    "windows": [np.array(w.points) for w in geo_result.vectorization.windows]
}

create_detection_report(
    image,
    detections,
    output_path='outputs/detection_report.png',
    title='Floor Plan Detection Report'
)

# Method 3: Compare before and after refinement
# (You need to save vec_result before refinement)
from src.utils.visualization import visualize_comparison

visualize_comparison(
    image,
    detections_before,
    detections_after,
    output_path='outputs/comparison.png'
)
    """)


def example_4_selective_detection():
    """
    Example 4: Enable only door detection or only window detection.
    """
    print("\n" + "=" * 60)
    print("Example 4: Selective Detection")
    print("=" * 60)
    
    # Doors only
    config_doors_only = RefinementConfig(
        enable_door_detection=True,
        enable_window_detection=False
    )
    
    # Windows only
    config_windows_only = RefinementConfig(
        enable_door_detection=False,
        enable_window_detection=True
    )
    
    print("\nOption 1 - Refine only doors (use YOLO for windows):")
    print(f"  enable_door_detection: {config_doors_only.enable_door_detection}")
    print(f"  enable_window_detection: {config_doors_only.enable_window_detection}")
    
    print("\nOption 2 - Refine only windows (use YOLO for doors):")
    print(f"  enable_door_detection: {config_windows_only.enable_door_detection}")
    print(f"  enable_window_detection: {config_windows_only.enable_window_detection}")


def example_5_disable_refinement():
    """
    Example 5: Disable refinement completely.
    """
    print("\n" + "=" * 60)
    print("Example 5: Disable Refinement")
    print("=" * 60)
    
    # Disable refinement
    geo_config = GeometryConfig(
        use_refinement=False,
        debug_visualization=True  # Can still visualize YOLO results
    )
    
    print("\nConfiguration:")
    print(f"  Refinement enabled: {geo_config.use_refinement}")
    print(f"  Debug visualization: {geo_config.debug_visualization}")
    print("\nThis will use original YOLO detections for doors and windows.")


def example_6_complete_workflow():
    """
    Example 6: Complete workflow with visualization.
    """
    print("\n" + "=" * 60)
    print("Example 6: Complete Workflow")
    print("=" * 60)
    
    print("\nComplete workflow with visualization:")
    print("""
# Step 1: Preprocessing
from src.preprocessing.pipeline import PreprocessingPipeline

prep_pipeline = PreprocessingPipeline()
cleaned_image = prep_pipeline.run("samples/18_png.rf.4956b6043e9f9f738808088cfe37243d.jpg")

# Step 2: Segmentation
from src.segmentation.predictor import FloorPlanPredictor

predictor = FloorPlanPredictor("models/best.pt")
seg_result = predictor.predict(cleaned_image)

# Step 3: Geometry with refinement and visualization
from src.geometry.pipeline import GeometryPipeline, GeometryConfig
from src.detection.refinement import RefinementConfig

# Configure refinement
refinement_config = RefinementConfig(
    door_width=90,
    wall_gap_threshold=20,
    enable_door_detection=True,
    enable_window_detection=True
)

# Configure geometry pipeline
geo_config = GeometryConfig(
    use_refinement=True,
    refinement_config=refinement_config,
    debug_visualization=True,
    visualization_dir="outputs/debug_viz"
)

geo_pipeline = GeometryPipeline(config=geo_config)
geo_result = geo_pipeline.run(seg_result, cleaned_image, "samples/18_png.rf.4956b6043e9f9f738808088cfe37243d.jpg")

# Visualizations are automatically saved to outputs/debug_viz/
# - floorplan1_before_refinement.png
# - floorplan1_after_refinement.png
# - floorplan1_comparison.png

# Step 4: Access results
print(f"Detected {len(geo_result.vectorization.doors)} doors")
print(f"Detected {len(geo_result.vectorization.windows)} windows")
print(f"Detected {len(geo_result.vectorization.rooms)} rooms")

# Step 5: Create custom visualization
from src.utils.visualization import create_detection_report

detections = {
    "walls": [np.array(w.points) for w in geo_result.vectorization.walls],
    "rooms": [np.array(r.points) for r in geo_result.vectorization.rooms],
    "doors": [np.array(d.points) for d in geo_result.vectorization.doors],
    "windows": [np.array(w.points) for w in geo_result.vectorization.windows]
}

create_detection_report(
    cleaned_image,
    detections,
    output_path="outputs/final_report.png",
    title="Floor Plan Analysis Report"
)
    """)


def main():
    """Run all examples."""
    print("=" * 60)
    print("Detection Refinement with Visualization - Examples")
    print("=" * 60)
    
    example_1_basic_with_debug()
    example_2_custom_refinement_config()
    example_3_manual_visualization()
    example_4_selective_detection()
    example_5_disable_refinement()
    example_6_complete_workflow()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nKey Points:")
    print("  1. Set debug_visualization=True to auto-generate comparison images")
    print("  2. Customize refinement parameters via RefinementConfig")
    print("  3. Use visualization utilities for manual inspection")
    print("  4. Enable/disable door or window detection independently")
    print("  5. Disable refinement completely to use YOLO detections")
    print("\nFor visual validation, check the generated images:")
    print("  - Green: Walls")
    print("  - Light green (transparent): Rooms")
    print("  - Blue: Doors")
    print("  - Red: Windows")


if __name__ == "__main__":
    main()
