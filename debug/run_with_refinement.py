#!/usr/bin/env python3
"""
Simple script to run the complete pipeline with detection refinement.

This script demonstrates the full workflow:
1. Preprocessing
2. Segmentation (YOLO)
3. Geometry with Detection Refinement
4. Visualization

Usage:
    python3 run_with_refinement.py
    python3 run_with_refinement.py samples/18_png.rf.4956b6043e9f9f738808088cfe37243d.jpg
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.segmentation.predictor import FloorPlanPredictor
from src.geometry.pipeline import GeometryPipeline, GeometryConfig
from src.detection.refinement import RefinementConfig


def main():
    """Run the complete pipeline with refinement and visualization."""
    
    print("=" * 70)
    print("Floor2Model Pipeline with Detection Refinement")
    print("=" * 70)
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "samples/18_png.rf.4956b6043e9f9f738808088cfe37243d.jpg"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"\n❌ Error: Input file not found: {input_file}")
        print("\nAvailable sample files:")
        if os.path.exists("samples"):
            for f in os.listdir("samples"):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    print(f"  - samples/{f}")
        print("\nUsage: python3 run_with_refinement.py <input_file>")
        return 1
    
    print(f"\n📁 Input file: {input_file}")
    
    # Check if YOLO model exists
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ Error: YOLO model not found: {model_path}")
        print("Please train or download the YOLOv8 model first.")
        return 1
    
    try:
        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Preprocessing
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "─" * 70)
        print("Phase 1: Preprocessing")
        print("─" * 70)
        
        prep_pipeline = PreprocessingPipeline()
        prep_result = prep_pipeline.run(input_file)
        
        # Extract the cleaned image from the result
        if hasattr(prep_result, 'cleaned'):
            cleaned_image = prep_result.cleaned
        else:
            # Fallback if prep_result is already the image
            cleaned_image = prep_result
        
        print(f"✓ Preprocessing complete")
        print(f"  Image shape: {cleaned_image.shape}")
        print(f"  Skew angle: {prep_result.skew_angle:.2f}°")
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Segmentation (YOLO)
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "─" * 70)
        print("Phase 2: Segmentation (YOLOv8)")
        print("─" * 70)
        
        predictor = FloorPlanPredictor(model_path)
        seg_result = predictor.predict(cleaned_image)
        
        print(f"✓ Segmentation complete")
        print(f"  Detected elements: {len(seg_result.elements)}")
        print(f"  Walls: {len(seg_result.walls)}")
        print(f"  Rooms: {len(seg_result.rooms)}")
        print(f"  Doors: {len(seg_result.doors)} (YOLO)")
        print(f"  Windows: {len(seg_result.windows)} (YOLO)")
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Geometry with Detection Refinement
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "─" * 70)
        print("Phase 3: Geometry with Detection Refinement")
        print("─" * 70)
        
        # Configure refinement
        refinement_config = RefinementConfig(
            # Door detection
            door_width=90,
            wall_gap_threshold=20,
            room_adjacency_threshold=50,
            
            # Window detection
            canny_low_threshold=50,
            canny_high_threshold=150,
            parallel_line_proximity=20,
            window_min_length=30,
            
            # Enable both
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
        geo_result = geo_pipeline.run(seg_result, cleaned_image, input_file)
        
        # ═══════════════════════════════════════════════════════════════
        # Results Summary
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)
        
        print(f"\n📊 Detection Counts:")
        print(f"  Walls:   {len(geo_result.vectorization.walls)}")
        print(f"  Rooms:   {len(geo_result.vectorization.rooms)}")
        print(f"  Doors:   {len(geo_result.vectorization.doors)} (refined)")
        print(f"  Windows: {len(geo_result.vectorization.windows)} (refined)")
        
        print(f"\n📏 Scale Estimate:")
        print(f"  {geo_result.scale.pixels_per_metre:.1f} px/m")
        print(f"  Method: {geo_result.scale.method}")
        print(f"  Confidence: {geo_result.scale.confidence:.0%}")
        
        print(f"\n🏠 Room Graph:")
        print(f"  Rooms: {len(geo_result.graph.nodes)}")
        print(f"  Connections: {len(geo_result.graph.edges)}")
        
        # ═══════════════════════════════════════════════════════════════
        # Visualization Files
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("Visualization Files")
        print("=" * 70)
        
        stem = Path(input_file).stem
        viz_dir = Path(geo_config.visualization_dir)
        
        viz_files = [
            viz_dir / f"{stem}_before_refinement.png",
            viz_dir / f"{stem}_after_refinement.png",
            viz_dir / f"{stem}_comparison.png"
        ]
        
        print(f"\n📁 Output directory: {viz_dir}/")
        for viz_file in viz_files:
            if viz_file.exists():
                print(f"  ✓ {viz_file.name}")
            else:
                print(f"  ✗ {viz_file.name} (not generated)")
        
        # ═══════════════════════════════════════════════════════════════
        # Validation Instructions
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("Visual Validation")
        print("=" * 70)
        
        print(f"\n🔍 Open the comparison image to validate:")
        print(f"  {viz_dir / f'{stem}_comparison.png'}")
        
        print(f"\n✅ Check that:")
        print(f"  • Doors (blue) are perpendicular to walls")
        print(f"  • Doors connect two rooms")
        print(f"  • Windows (red) are parallel to walls")
        print(f"  • Windows are inside wall boundaries")
        print(f"  • Walls (green) are unchanged")
        print(f"  • Rooms (light green) are unchanged")
        
        print("\n" + "=" * 70)
        print("✓ Pipeline Complete!")
        print("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
