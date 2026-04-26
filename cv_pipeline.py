"""
cv_pipeline.py
--------------
CV Pipeline — processes all floor plan images in samples/ and produces
3D building models (.gltf / .obj) in generated_models/.

Phases:
  2 — YOLOv8 segmentation (wall / door / window detection)
  3 — Geometry reconstruction (vectorization, scale, room graph)
  4 — 3D reconstruction (extrusion, wall splitting, export)

Usage:
    python cv_pipeline.py

Output: generated_models/<stem>/ for each image in samples/
"""

import cv2
import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.predictor import FloorPlanPredictor
from src.segmentation.visualizer import SegmentationVisualizer
from src.geometry.pipeline import GeometryPipeline
from src.reconstruction.pipeline import ReconstructionPipeline

MODEL_PATH    = PROJECT_ROOT / "models" / "best.pt"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
GENERATED_DIR = PROJECT_ROOT / "generated_models"

print(f"Project root: {PROJECT_ROOT}")
print(f"Model exists: {MODEL_PATH.exists()}")


def run_cv_pipeline(sample_image: Path) -> dict | None:
    """
    Run the full CV pipeline on one floor plan image.

    Returns a dict with keys: stem, out_dir, seg_result, geo_result, model_3d
    so the GAN pipeline can consume the outputs directly.
    Returns None if processing failed.
    """
    stem = sample_image.stem

    print(f"\n{'='*60}")
    print(f"Processing: {sample_image.name}")
    print(f"{'='*60}")

    # ── Phase 2: Segmentation ─────────────────────────────────────────
    print("\n[Phase 2] Segmentation...")
    best_conf   = None
    best_result = None

    for conf in [0.35, 0.25, 0.15, 0.10, 0.05]:
        predictor  = FloorPlanPredictor(str(MODEL_PATH), confidence=conf, device="cpu")
        seg_result = predictor.predict(str(sample_image))
        total      = seg_result.summary["total_elements"]
        print(f"  conf={conf}: {total} elements detected")
        if total > 0 and best_result is None:
            best_conf   = conf
            best_result = seg_result

    base_img = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
    if base_img is not None and len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    out_dir = GENERATED_DIR / stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ann_path = out_dir / f"{stem}_detections.png"

    if best_result is None:
        print("  ⚠ No elements detected.")
        if base_img is not None:
            cv2.imwrite(str(ann_path), base_img)
        return None

    print(f"  ✓ conf={best_conf} → {best_result.summary['total_elements']} elements")
    print(f"  {best_result.summary['by_class']}")

    viz = SegmentationVisualizer()
    cv2.imwrite(str(ann_path), viz.draw(base_img, best_result))
    print(f"  ✓ Detection overlay → {ann_path.name}")

    # ── Phase 3: Geometry ─────────────────────────────────────────────
    print("\n[Phase 3] Geometry reconstruction...")
    img = cv2.imread(str(sample_image))
    geo_result, _, _ = GeometryPipeline().run_and_visualize(
        best_result, img,
        image_path=str(sample_image),
        output_dir=str(OUTPUTS_DIR / "geometry"),
    )
    print(f"  Scale: {geo_result.scale.pixels_per_metre:.1f} px/m ({geo_result.scale.method})")

    # ── Phase 4: 3D Reconstruction ────────────────────────────────────
    print("\n[Phase 4] 3D reconstruction...")

    if not geo_result.vectorization.all_polygons and not best_result.elements:
        print("  ⚠ No geometry — skipping 3D.")
        return None

    model_3d = ReconstructionPipeline().reconstruct(
        geo_result,
        output_dir=str(out_dir),
        stem=stem,
        floorplan_image_path=str(sample_image),
    )
    print(f"  ✓ {model_3d.summary}")
    print(f"\n  CV output → {out_dir}/")
    print(f"  ├── {stem}_detections.png")
    print(f"  ├── {stem}.gltf")
    print(f"  └── {stem}.obj")

    return {
        "stem":       stem,
        "out_dir":    out_dir,
        "image_path": sample_image,
        "seg_result": best_result,
        "geo_result": geo_result,
        "model_3d":   model_3d,
    }


def collect_samples() -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")
    return sorted(
        p for ext in exts for p in (PROJECT_ROOT / "samples").glob(ext)
    )


if __name__ == "__main__":
    samples = collect_samples()
    if not samples:
        print("❌ No image files in samples/")
        sys.exit(1)

    print(f"Found {len(samples)} sample(s): {[s.name for s in samples]}")
    for sample in samples:
        run_cv_pipeline(sample)

    print(f"\n{'='*60}")
    print("✓ CV pipeline complete → generated_models/")
    print(f"{'='*60}")
