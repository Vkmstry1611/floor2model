import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Force correct working directory to floor2model root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.predictor import FloorPlanPredictor
from src.geometry.pipeline import GeometryPipeline
from src.reconstruction.pipeline import ReconstructionPipeline

# ── Absolute paths ────────────────────────────────────────────────────
SAMPLE_IMAGE  = str(PROJECT_ROOT / "samples"    / "floorplan2.png")
CLEANED_IMAGE = str(PROJECT_ROOT / "outputs"    / "floorplan2_4_cleaned.png")
MODEL_PATH    = str(PROJECT_ROOT / "src" / "segmentation" / "best.pt")
MAIN_PY       = str(PROJECT_ROOT / "main.py")

print(f"Project root:  {PROJECT_ROOT}")
print(f"Model exists:  {os.path.exists(MODEL_PATH)}")
print(f"Sample exists: {os.path.exists(SAMPLE_IMAGE)}")

# ── Phase 1: Preprocessing ────────────────────────────────────────────
if not os.path.exists(CLEANED_IMAGE):
    print("\nRunning Phase 1 preprocessing...")
    os.system(f"python {MAIN_PY} --input {SAMPLE_IMAGE}")
else:
    print(f"Found cleaned image: {CLEANED_IMAGE}")

# ── Image sanity check ────────────────────────────────────────────────
img_check = cv2.imread(CLEANED_IMAGE, cv2.IMREAD_GRAYSCALE)
print(f"\nImage shape:      {img_check.shape}")
print(f"Pixel range:      min={img_check.min()}, max={img_check.max()}")
print(f"Non-zero pixels:  {np.count_nonzero(img_check)}")

# ── Phase 2: Find best confidence threshold ───────────────────────────
print("\n== PHASE 2: Finding best confidence threshold ==")
best_conf   = 0.35
best_result = None

for conf in [0.35, 0.25, 0.15, 0.10, 0.05]:
    predictor  = FloorPlanPredictor(MODEL_PATH, confidence=conf)
    seg_result = predictor.predict(CLEANED_IMAGE)
    total      = seg_result.summary["total_elements"]
    print(f"  conf={conf}: {total} elements detected")
    if total > 0 and best_result is None:
        best_conf   = conf
        best_result = seg_result

if best_result is None:
    print("\nWARNING: No elements detected at any confidence level.")
    print("Possible reasons:")
    print("  1. Floor plan style differs from CubiCasa training data")
    print("  2. Model needs more training epochs (currently 71/100)")
    print("  3. Try a different floor plan image")
    sys.exit(0)

print(f"\nUsing confidence={best_conf} — detected {best_result.summary['total_elements']} elements")
print(best_result.summary)

# ── Phase 3: Geometry reconstruction ─────────────────────────────────
print("\n== PHASE 3: Geometry Reconstruction ==")
img = cv2.imread(CLEANED_IMAGE)
geo_pipeline = GeometryPipeline()
geo_result, poly_viz, graph_viz = geo_pipeline.run_and_visualize(
    best_result, img,
    image_path=CLEANED_IMAGE,
    output_dir=str(PROJECT_ROOT / "outputs" / "geometry"),
)

# ── Results ───────────────────────────────────────────────────────────
print("\n== RESULTS ==")
print(f"Graph:   {geo_result.graph.summary}")
print(f"Walls:   {len(geo_result.vectorization.walls)}")
print(f"Rooms:   {len(geo_result.vectorization.rooms)}")
print(f"Doors:   {len(geo_result.vectorization.doors)}")
print(f"Windows: {len(geo_result.vectorization.windows)}")
print(f"Scale:   {geo_result.scale.pixels_per_metre:.1f} px/m ({geo_result.scale.method})")
print(f"\nOpen these to see results:")
print(f"  outputs/geometry/floorplan2_4_cleaned_polygons.png")
print(f"  outputs/geometry/floorplan2_4_cleaned_graph.png")

# ── Phase 4: 3D Reconstruction ───────────────────────────────────────
print("\n== PHASE 4: 3D Reconstruction ==")
reconstruction_pipeline = ReconstructionPipeline()
model_3d = reconstruction_pipeline.reconstruct(
    geo_result,
    output_dir=str(PROJECT_ROOT / "outputs" / "models"),
)

print(f"\n3D Model: {model_3d.summary}")
print(f"\nExported files:")
for fmt, path in model_3d.export_paths.items():
    print(f"  {fmt.upper()}: {path}")