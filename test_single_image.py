#!/usr/bin/env python3
"""
Test the complete pipeline on a single image with visualization.

This script shows you:
1. What YOLO detects (before refinement)
2. What refinement detects (after refinement)
3. Side-by-side comparison

Usage:
    python3 test_single_image.py samples/floorplan1.png
    python3 test_single_image.py samples/floorplan2.png
    python3 test_single_image.py path/to/your/image.png
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Force correct working directory
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.segmentation.predictor import FloorPlanPredictor
from src.geometry.pipeline import GeometryPipeline, GeometryConfig
from src.detection.refinement import RefinementConfig
from src.utils.visualization import (
    visualize_vectorization_result,
    visualize_comparison,
    create_detection_report
)


def main():
    """Test pipeline on a single image."""
    
    print("=" * 80)
    print("Test Single Image - Complete Pipeline with Refinement")
    print("=" * 80)
    
    # Get input image
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        # Default to floorplan2.png
        input_image = "samples/floorplan2.png"
    
    if not os.path.exists(input_image):
        print(f"\n❌ Error: Image not found: {input_image}")
        print("\nUsage: python3 test_single_image.py <image_path>")
        print("\nAvailable samples:")
        if os.path.exists("samples"):
            for f in sorted(os.listdir("samples")):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    print(f"  - samples/{f}")
        return 1
    
    print(f"\n📁 Input: {input_image}")
    
    # Check model
    model_path = "src/segmentation/best.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ Error: YOLO model not found: {model_path}")
        print("Please train or download the model first.")
        return 1
    
    # Create output directory
    output_dir = Path("outputs/single_image_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Preprocessing
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 80)
        print("PHASE 1: Preprocessing")
        print("─" * 80)
        
        prep_pipeline = PreprocessingPipeline()
        prep_result = prep_pipeline.run(input_image)
        cleaned_image = prep_result.cleaned
        
        print(f"✓ Preprocessing complete")
        print(f"  Image shape: {cleaned_image.shape}")
        print(f"  Skew angle: {prep_result.skew_angle:.2f}°")
        
        # Save cleaned image
        cv2.imwrite(str(output_dir / "1_cleaned.png"), cleaned_image)
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Segmentation (YOLO)
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 80)
        print("PHASE 2: Segmentation (YOLO)")
        print("─" * 80)
        
        # Try different confidence thresholds
        print("Finding best confidence threshold...")
        best_conf = 0.35
        best_result = None
        
        for conf in [0.35, 0.25, 0.15, 0.10, 0.05]:
            predictor = FloorPlanPredictor(model_path, confidence=conf)
            seg_result = predictor.predict(cleaned_image)
            total = seg_result.summary["total_elements"]
            print(f"  conf={conf:.2f}: {total} elements")
            
            if total > 0 and best_result is None:
                best_conf = conf
                best_result = seg_result
        
        if best_result is None:
            print("\n⚠️  WARNING: No elements detected at any confidence level")
            print("This image may not work well with the current YOLO model.")
            return 1
        
        seg_result = best_result
        print(f"\n✓ Using confidence={best_conf:.2f}")
        print(f"  Total elements: {seg_result.summary['total_elements']}")
        print(f"  Walls: {len(seg_result.walls)}")
        print(f"  Rooms: {len(seg_result.rooms)}")
        print(f"  Doors: {len(seg_result.doors)} (YOLO)")
        print(f"  Windows: {len(seg_result.windows)} (YOLO)")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3A: Geometry WITHOUT Refinement (to see YOLO results)
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 80)
        print("PHASE 3A: Geometry WITHOUT Refinement (YOLO only)")
        print("─" * 80)
        
        geo_config_no_refine = GeometryConfig(use_refinement=False)
        geo_pipeline_no_refine = GeometryPipeline(config=geo_config_no_refine)
        
        img_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
        geo_result_before = geo_pipeline_no_refine.run(
            seg_result, img_bgr, input_image
        )
        
        print(f"✓ Geometry (YOLO only) complete")
        print(f"  Walls: {len(geo_result_before.vectorization.walls)}")
        print(f"  Rooms: {len(geo_result_before.vectorization.rooms)}")
        print(f"  Doors: {len(geo_result_before.vectorization.doors)} (YOLO)")
        print(f"  Windows: {len(geo_result_before.vectorization.windows)} (YOLO)")
        
        # Visualize YOLO results
        viz_before = visualize_vectorization_result(
            img_bgr,
            geo_result_before.vectorization,
            output_path=str(output_dir / "2_yolo_detections.png"),
            show_labels=True
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3B: Geometry WITH Refinement
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 80)
        print("PHASE 3B: Geometry WITH Refinement")
        print("─" * 80)
        
        refinement_config = RefinementConfig(
            door_width=90,
            wall_gap_threshold=20,
            room_adjacency_threshold=50,
            canny_low_threshold=50,
            canny_high_threshold=150,
            enable_door_detection=True,
            enable_window_detection=True
        )
        
        geo_config_refine = GeometryConfig(
            use_refinement=True,
            refinement_config=refinement_config
        )
        
        geo_pipeline_refine = GeometryPipeline(config=geo_config_refine)
        geo_result_after = geo_pipeline_refine.run(
            seg_result, img_bgr, input_image
        )
        
        print(f"✓ Geometry (with refinement) complete")
        print(f"  Walls: {len(geo_result_after.vectorization.walls)}")
        print(f"  Rooms: {len(geo_result_after.vectorization.rooms)}")
        print(f"  Doors: {len(geo_result_after.vectorization.doors)} (refined)")
        print(f"  Windows: {len(geo_result_after.vectorization.windows)} (refined)")
        
        # Visualize refined results
        viz_after = visualize_vectorization_result(
            img_bgr,
            geo_result_after.vectorization,
            output_path=str(output_dir / "3_refined_detections.png"),
            show_labels=True
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPARISON
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 80)
        print("Creating Comparison Visualizations")
        print("─" * 80)
        
        # Create comparison
        detections_before = {
            "walls": [np.array(w.points) for w in geo_result_before.vectorization.walls],
            "rooms": [np.array(r.points) for r in geo_result_before.vectorization.rooms],
            "doors": [np.array(d.points) for d in geo_result_before.vectorization.doors],
            "windows": [np.array(w.points) for w in geo_result_before.vectorization.windows]
        }
        
        detections_after = {
            "walls": [np.array(w.points) for w in geo_result_after.vectorization.walls],
            "rooms": [np.array(r.points) for r in geo_result_after.vectorization.rooms],
            "doors": [np.array(d.points) for d in geo_result_after.vectorization.doors],
            "windows": [np.array(w.points) for w in geo_result_after.vectorization.windows]
        }
        
        comparison = visualize_comparison(
            img_bgr,
            detections_before,
            detections_after,
            output_path=str(output_dir / "4_comparison.png")
        )
        
        # Create detailed report
        create_detection_report(
            img_bgr,
            detections_after,
            output_path=str(output_dir / "5_final_report.png"),
            title="Detection Report (After Refinement)"
        )
        
        print(f"✓ Visualizations created")
        
        # ═══════════════════════════════════════════════════════════════════
        # SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Detection Comparison:")
        print(f"  {'Element':<12} {'YOLO':<10} {'Refined':<10} {'Change':<10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        walls_before = len(geo_result_before.vectorization.walls)
        walls_after = len(geo_result_after.vectorization.walls)
        print(f"  {'Walls':<12} {walls_before:<10} {walls_after:<10} {walls_after-walls_before:+d}")
        
        rooms_before = len(geo_result_before.vectorization.rooms)
        rooms_after = len(geo_result_after.vectorization.rooms)
        print(f"  {'Rooms':<12} {rooms_before:<10} {rooms_after:<10} {rooms_after-rooms_before:+d}")
        
        doors_before = len(geo_result_before.vectorization.doors)
        doors_after = len(geo_result_after.vectorization.doors)
        print(f"  {'Doors':<12} {doors_before:<10} {doors_after:<10} {doors_after-doors_before:+d}")
        
        windows_before = len(geo_result_before.vectorization.windows)
        windows_after = len(geo_result_after.vectorization.windows)
        print(f"  {'Windows':<12} {windows_before:<10} {windows_after:<10} {windows_after-windows_before:+d}")
        
        print(f"\n📏 Scale: {geo_result_after.scale.pixels_per_metre:.1f} px/m ({geo_result_after.scale.method})")
        print(f"🏠 Rooms: {len(geo_result_after.graph.nodes)} rooms, {len(geo_result_after.graph.edges)} connections")
        
        print(f"\n📁 Output Directory: {output_dir}/")
        print(f"\n🎨 Generated Files:")
        print(f"  1. 1_cleaned.png - Preprocessed image")
        print(f"  2. 2_yolo_detections.png - YOLO detections (before)")
        print(f"  3. 3_refined_detections.png - Refined detections (after)")
        print(f"  4. 4_comparison.png - Side-by-side comparison ⭐")
        print(f"  5. 5_final_report.png - Final report with stats")
        
        print(f"\n✅ OPEN THIS FILE TO SEE RESULTS:")
        print(f"   {output_dir / '4_comparison.png'}")
        
        print(f"\n🎨 Color Legend:")
        print(f"  🟢 Green = Walls")
        print(f"  🟢 Light Green = Rooms (transparent)")
        print(f"  🔵 Blue = Doors")
        print(f"  🔴 Red = Windows")
        
        print("\n" + "=" * 80)
        print("✓ Test Complete!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
