"""
cv_pipeline.py
--------------
CV Pipeline — Phase 2 → 3 → 4

Processes all images in samples/ and outputs to generated_models/<stem>/:
  - <stem>_detections.png   YOLO detection overlay
  - <stem>_floorplan.png    clean floorplan texture
  - <stem>.gltf             3D building model (walls, doors, windows)
  - <stem>.obj + .mtl       same model in OBJ format

Usage:
    python cv_pipeline.py                    # all images in samples/
    python cv_pipeline.py samples/plan.png   # single image
"""

from __future__ import annotations

import cv2
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.predictor import FloorPlanPredictor
from src.segmentation.visualizer import SegmentationVisualizer
from src.geometry.pipeline import GeometryPipeline
from src.reconstruction.pipeline import ReconstructionPipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = PROJECT_ROOT / "models" / "best.pt"
SAMPLES_DIR   = PROJECT_ROOT / "samples"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
GENERATED_DIR = PROJECT_ROOT / "generated_models"

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")


def collect_samples(path: Optional[Path] = None) -> list[Path]:
    """
    Collect all sample images.
    If path is a file → return [path].
    If path is a directory (or None → samples/) → return all images in it.
    """
    if path is not None and path.is_file():
        return [path]
    search_dir = path if (path and path.is_dir()) else SAMPLES_DIR
    return sorted(p for ext in IMAGE_EXTS for p in search_dir.glob(ext))


def run_cv(image_path: Path) -> Optional[Path]:
    """
    Run the full CV pipeline on one image.
    Returns the output directory path, or None if processing failed.
    """
    stem = image_path.stem

    print(f"\n{'='*60}")
    print(f"CV Pipeline: {image_path.name}")
    print(f"{'='*60}")

    # ── Phase 2: Segmentation ─────────────────────────────────────────
    print("\n[Phase 2] Segmentation...")
    best_result = None
    best_conf   = None

    for conf in [0.35, 0.25, 0.15, 0.10, 0.05]:
        predictor  = FloorPlanPredictor(str(MODEL_PATH), confidence=conf, device="cpu")
        seg_result = predictor.predict(str(image_path))
        total      = seg_result.summary["total_elements"]
        print(f"  conf={conf}: {total} elements detected")
        if total > 0 and best_result is None:
            best_conf   = conf
            best_result = seg_result

    # Prepare output directory
    out_dir = GENERATED_DIR / stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    base_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if base_img is not None and len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    ann_path = out_dir / f"{stem}_detections.png"

    if best_result is None:
        print("  ⚠ No elements detected — skipping.")
        if base_img is not None:
            cv2.imwrite(str(ann_path), base_img)
        return None

    print(f"  ✓ conf={best_conf} → {best_result.summary['total_elements']} elements")
    print(f"  {best_result.summary['by_class']}")

    viz       = SegmentationVisualizer()
    annotated = viz.draw(base_img, best_result)
    cv2.imwrite(str(ann_path), annotated)
    print(f"  ✓ Detections → {ann_path.name}")

    # ── Phase 3: Geometry ─────────────────────────────────────────────
    print("\n[Phase 3] Geometry reconstruction...")
    img = cv2.imread(str(image_path))
    geo_result, _, _ = GeometryPipeline().run_and_visualize(
        best_result, img,
        image_path=str(image_path),
        output_dir=str(OUTPUTS_DIR / "geometry"),
    )
    print(f"  Walls: {len(geo_result.vectorization.walls)}  "
          f"Doors: {len(geo_result.vectorization.doors)}  "
          f"Windows: {len(geo_result.vectorization.windows)}")
    print(f"  Scale: {geo_result.scale.pixels_per_metre:.1f} px/m ({geo_result.scale.method})")

    # ── Phase 4: 3D Reconstruction ────────────────────────────────────
    print("\n[Phase 4] 3D reconstruction...")

    if not len(best_result.elements) > 0 and not len(geo_result.vectorization.all_polygons) > 0:
        print("  ⚠ No geometry — skipping 3D.")
        return None

    model_3d = ReconstructionPipeline().reconstruct(
        geo_result,
        output_dir=str(out_dir),
        stem=stem,
        floorplan_image_path=str(image_path),
        render_image_path=None,
    )
    print(f"  ✓ {model_3d.summary}")
    print(f"\n  Output → generated_models/{stem}/")
    print(f"  ├── {stem}_detections.png")
    print(f"  ├── {stem}.gltf")
    print(f"  └── {stem}.obj")

    return out_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if not input_path.exists():
            print(f"❌ Path not found: {input_path}")
            sys.exit(1)
        images = collect_samples(input_path)
    else:
        images = collect_samples()

    if not images:
        print(f"❌ No images found in {SAMPLES_DIR}")
        sys.exit(1)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model:        {MODEL_PATH.name}  (exists={MODEL_PATH.exists()})")
    print(f"Found {len(images)} image(s): {[i.name for i in images]}\n")

    success = 0
    for img in images:
        if run_cv(img):
            success += 1

    print(f"\n{'='*60}")
    print(f"✓ CV Pipeline complete — {success}/{len(images)} processed")
    print(f"  Output: generated_models/")
    print(f"{'='*60}")
