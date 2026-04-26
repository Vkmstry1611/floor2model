"""
gan_runner.py
-------------
GAN Pipeline — takes CV-generated .gltf files and produces
_interior.gltf files with realistic textures and furniture.

Pipeline:
  1. Read CV .gltf from generated_models/<stem>/
  2. Run OCR on floorplan image to detect room labels
  3. GAN (Pix2Pix) generates textures per surface (wall/floor/door/window)
  4. Apply textures + place furniture based on OCR room positions
  5. Write gan_output/<stem>/<stem>_interior.gltf

Swap to your trained GAN when ready:
  BACKEND        = "gan"
  GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"

Usage:
    python -m gan_part.gan_runner                    # all samples
    python -m gan_part.gan_runner floorplan1         # specific stem
"""

from __future__ import annotations

import sys
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Config — change BACKEND + GAN_CHECKPOINT when your model is ready ─────────
BACKEND        = "controlnet"   # "controlnet" | "gan"
GAN_CHECKPOINT = None           # "gan_part/checkpoints/final.pt"
DEVICE         = "cpu"

CV_OUTPUT_DIR  = PROJECT_ROOT / "generated_models"
GAN_OUTPUT_DIR = PROJECT_ROOT / "gan_output"
SAMPLES_DIR    = PROJECT_ROOT / "samples"

# ── GAN imports ───────────────────────────────────────────────────────────────
from gan_part.data.preprocessor import build_seg_map_from_elements
from gan_part.inference.room_renderer import RoomRenderer
from gan_part.inference.texture_generator import TextureGenerator
from gan_part.inference.gltf_texturer import apply_textures
from gan_part.inference.ocr_room_detector import detect_rooms_ocr, match_rooms_to_footprint

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_renderer = None
_tex_gen  = None


def get_renderer() -> RoomRenderer:
    """ControlNet / GAN renderer for interior image generation."""
    global _renderer
    if _renderer is None:
        print(f"  Loading {'GAN' if BACKEND == 'gan' else 'ControlNet'} renderer...")
        _renderer = RoomRenderer(
            backend=BACKEND,
            gan_checkpoint=GAN_CHECKPOINT,
            device=DEVICE,
        )
    return _renderer


def get_texture_generator() -> TextureGenerator:
    """Pix2Pix GAN texture generator for surface materials."""
    global _tex_gen
    if _tex_gen is None:
        _tex_gen = TextureGenerator(device=DEVICE)
    return _tex_gen


# ── Phase A: Interior render (ControlNet / GAN) ───────────────────────────────

def run_interior_render(seg_result, out_dir: Path, stem: str) -> Path | None:
    """
    Generate a photorealistic interior render from the segmentation result.
    Saves <stem>_render.png. Returns path or None on failure.
    """
    print("\n[GAN Phase A] Interior render...")
    try:
        renderer = get_renderer()
        seg_map  = build_seg_map_from_elements(
            seg_result.elements,
            seg_result.image_shape,
        )
        room_counts = {}
        for e in seg_result.elements:
            room_counts[e.class_name] = room_counts.get(e.class_name, 0) + 1
        room_label = max(room_counts, key=room_counts.get) if room_counts else "default"

        rendered   = renderer.render(seg_map, room_label=room_label)
        render_path = out_dir / f"{stem}_render.png"
        rendered.save(str(render_path))
        print(f"  ✓ Render saved → {render_path.name}")
        return render_path
    except Exception as e:
        print(f"  ⚠ Render failed: {e}")
        return None


# ── Phase B: Textured interior .gltf ─────────────────────────────────────────

def run_interior_gltf(stem: str, image_path: Path | None = None) -> Path | None:
    """
    Read CV .gltf, apply GAN textures + furniture, write _interior.gltf.
    Returns output path or None on failure.
    """
    cv_gltf = CV_OUTPUT_DIR / stem / f"{stem}.gltf"
    if not cv_gltf.exists():
        print(f"  ✗ CV gltf not found: {cv_gltf}")
        return None

    # Find the original floorplan image for OCR + floor texture
    fp_image = image_path
    if fp_image is None:
        # Try to find it in samples/
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
            candidate = SAMPLES_DIR / f"{stem}{ext}"
            if candidate.exists():
                fp_image = candidate
                break

    out_dir  = GAN_OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{stem}_interior.gltf")

    print(f"\n[GAN Phase B] Textured interior: {stem}")
    try:
        tex_gen = get_texture_generator()
        apply_textures(
            str(cv_gltf),
            tex_gen,
            out_path,
            image_path=str(fp_image) if fp_image else None,
        )
        print(f"  ✓ Interior gltf → gan_output/{stem}/{stem}_interior.gltf")
        print(f"  View: https://gltf-viewer.donmccurdy.com")
        return Path(out_path)
    except Exception as e:
        import traceback
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return None


# ── Full GAN pipeline for one stem ───────────────────────────────────────────

def process_floorplan(stem: str,
                      seg_result=None,
                      image_path: Path | None = None) -> Path | None:
    """
    Run the full GAN pipeline for one floorplan.

    Can be called:
      - Standalone: process_floorplan("floorplan1")
      - From main.py with CV results: process_floorplan(stem, seg_result, image_path)
    """
    print(f"\n{'='*60}")
    print(f"GAN Pipeline: {stem}")
    print(f"{'='*60}")

    # Phase A: render (only if seg_result provided — i.e. called from main.py)
    if seg_result is not None:
        cv_out_dir = CV_OUTPUT_DIR / stem
        cv_out_dir.mkdir(parents=True, exist_ok=True)
        run_interior_render(seg_result, cv_out_dir, stem)

    # Phase B: textured gltf
    return run_interior_gltf(stem, image_path)


# ── Entry point (standalone) ──────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        stems = sys.argv[1:]
    else:
        stems = sorted(
            d.name for d in CV_OUTPUT_DIR.iterdir()
            if d.is_dir() and (d / f"{d.name}.gltf").exists()
        )

    if not stems:
        print("❌ No CV outputs found in generated_models/")
        print("   Run cv_pipeline.py first.")
        sys.exit(1)

    print(f"Found {len(stems)} floorplan(s): {stems}")
    print(f"Backend: {BACKEND}\n")

    results = []
    for stem in stems:
        result = process_floorplan(stem)
        if result:
            results.append(result)

    print(f"\n{'='*60}")
    print(f"✓ GAN pipeline complete — {len(results)}/{len(stems)} processed")
    print(f"  Output: gan_output/")
    print(f"  View:   https://gltf-viewer.donmccurdy.com")
    print(f"{'='*60}")
