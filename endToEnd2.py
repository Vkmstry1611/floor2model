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

# ── Phase 5: GAN/ControlNet rendering (optional) ──────────────────────────────
# Skips cleanly if gan_part dependencies are not installed.
try:
    from gan_part.data.preprocessor import build_seg_map_from_elements
    from gan_part.inference.room_renderer import RoomRenderer
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False

MODEL_PATH    = PROJECT_ROOT / "models" / "best.pt"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
GENERATED_DIR = PROJECT_ROOT / "generated_models"

# ── ControlNet renderer (lazy-loaded once, reused across all samples) ──────────
_renderer = None

def get_renderer():
    global _renderer
    if _renderer is None:
        print("\n[Phase 5] Loading ControlNet pipeline (first time only)...")
        _renderer = RoomRenderer(backend="controlnet", device="cpu")
    return _renderer


def run_gan_render(seg_result, out_dir: Path, stem: str):
    """
    Phase 5: Build ADE20K seg map from detections and render with ControlNet.
    Saves <stem>_render.png into out_dir.
    Returns the render path if successful, None otherwise.
    """
    if not GAN_AVAILABLE:
        print("\n[Phase 5] Skipped — install gan_part deps: pip install -r gan_part/requirements.txt")
        return None

    print("\n[Phase 5] ControlNet interior render...")
    try:
        renderer = get_renderer()

        seg_map = build_seg_map_from_elements(
            seg_result.elements,
            seg_result.image_shape,
        )

        room_counts = {}
        for e in seg_result.elements:
            room_counts[e.class_name] = room_counts.get(e.class_name, 0) + 1
        room_label = max(room_counts, key=room_counts.get) if room_counts else "default"

        rendered = renderer.render(seg_map, room_label=room_label)

        render_path = out_dir / f"{stem}_render.png"
        rendered.save(str(render_path))
        print(f"  ✓ Render saved → {render_path.name}")
        return render_path

    except Exception as e:
        print(f"  ⚠ Phase 5 failed: {e}")
        return None

print(f"Project root: {PROJECT_ROOT}")
print(f"Model exists: {MODEL_PATH.exists()}")


def run_pipeline(sample_image: Path):
    stem = sample_image.stem

    print(f"\n{'='*60}")
    print(f"Processing: {sample_image.name}")
    print(f"{'='*60}")

    # ── Phase 2: Segmentation (raw image, CPU) ────────────────────────
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

    # Load original color image for drawing
    base_img = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
    if base_img is not None and len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    out_dir  = GENERATED_DIR / stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ann_path = out_dir / f"{stem}_detections.png"

    if best_result is None:
        print("  ⚠ No elements detected.")
        if base_img is not None:
            cv2.imwrite(str(ann_path), base_img)
        return

    print(f"  ✓ conf={best_conf} → {best_result.summary['total_elements']} elements")
    print(f"  {best_result.summary['by_class']}")
    print(f"  Walls: {len(best_result.walls)}  Doors: {len(best_result.doors)}  Windows: {len(best_result.windows)}")

    # Save detection overlay into output folder
    viz       = SegmentationVisualizer()
    annotated = viz.draw(base_img, best_result)
    cv2.imwrite(str(ann_path), annotated)
    print(f"  ✓ Detection overlay → {ann_path.name}")

    # ── Phase 3: Geometry ─────────────────────────────────────────────
    print("\n[Phase 3] Geometry reconstruction...")
    img = cv2.imread(str(sample_image))
    geo_result, _, _ = GeometryPipeline().run_and_visualize(
        best_result, img,
        image_path=str(sample_image),
        output_dir=str(OUTPUTS_DIR / "geometry"),
    )
    print(f"  Walls: {len(geo_result.vectorization.walls)}  "
          f"Rooms: {len(geo_result.vectorization.rooms)}  "
          f"Doors: {len(geo_result.vectorization.doors)}  "
          f"Windows: {len(geo_result.vectorization.windows)}")
    print(f"  Scale: {geo_result.scale.pixels_per_metre:.1f} px/m ({geo_result.scale.method})")

    # ── Phase 5: ControlNet interior render ───────────────────────────
    # Run BEFORE Phase 4 so the render can be embedded in the 3D model
    render_path = None
    render_path = run_gan_render(best_result, out_dir, stem)

    # ── Phase 4: 3D Reconstruction ────────────────────────────────────
    print("\n[Phase 4] 3D reconstruction...")

    seg_has_elements = best_result is not None and len(best_result.elements) > 0
    has_polygons = len(geo_result.vectorization.all_polygons) > 0

    if not has_polygons and not seg_has_elements:
        print("  ⚠ No geometry to extrude — skipping 3D.")
        return

    model_3d = ReconstructionPipeline().reconstruct(
        geo_result,
        output_dir=str(out_dir),
        stem=stem,
        floorplan_image_path=str(sample_image),
        render_image_path=str(render_path) if render_path else None,
    )
    print(f"  ✓ {model_3d.summary}")

    print(f"\n  Output → {out_dir}/")
    print(f"  ├── {stem}_detections.png")
    print(f"  ├── {stem}.gltf")
    print(f"  ├── {stem}.obj")
    if render_path:
        print(f"  └── {stem}_render.png  (embedded in .gltf)")


if __name__ == "__main__":
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")
    samples = sorted(
        p for ext in exts for p in (PROJECT_ROOT / "samples").glob(ext)
    )
    if not samples:
        print("❌ No image files in samples/")
        sys.exit(1)

    print(f"Found {len(samples)} sample(s): {[s.name for s in samples]}")
    for sample in samples:
        run_pipeline(sample)

    print(f"\n{'='*60}")
    print("✓ All samples processed → generated_models/")
    print(f"{'='*60}")
