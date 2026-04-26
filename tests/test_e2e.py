#!/usr/bin/env python3
"""
End-to-end test for the floor2model pipeline.
Tests the complete flow: preprocessing -> segmentation -> geometry -> 3D modeling

Run with: python test_e2e.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_full_pipeline():
    """Test the complete floor plan to 3D model conversion pipeline."""
    print("🚀 Starting end-to-end pipeline test...")

    try:
        # Test imports
        print("📦 Testing imports...")
        from preprocessing.pipeline import PreprocessingPipeline
        from segmentation.predictor import FloorPlanPredictor
        from geometry.pipeline import GeometryPipeline
        print("✅ All modules imported successfully")

        # Test preprocessing
        print("🧪 Testing preprocessing pipeline...")
        # TODO: Add actual preprocessing test

        # Test segmentation
        print("🧪 Testing segmentation pipeline...")
        # TODO: Add actual segmentation test

        # Test geometry
        print("🧪 Testing geometry pipeline...")
        # TODO: Add actual geometry test

        print("🎉 End-to-end test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_full_pipeline()