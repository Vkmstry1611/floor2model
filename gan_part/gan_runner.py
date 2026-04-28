"""
gan_runner.py
-------------
GAN pipeline — reads CV-generated .gltf and produces a new
_interior.gltf in gan_output/<stem>/ with realistic surface textures.

Pipeline:
  1. Find CV .gltf in generated_models/<stem>/
  2. Parse wall/door/window/floor face groups by per-vertex color
  3. GAN (ControlNet or procedural fallback) generates one texture per surface
  4. Apply textures to correct faces with UV mapping
  5. Write gan_output/<stem>/<stem>_interior.gltf

Swap backend when your GAN is ready:
  Set BACKEND = "gan"
  Set GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND        = "controlnet"   # "controlnet" | "gan"
GAN_CHECKPOINT = None           # "gan_part/checkpoints/final.pt"
DEVICE         = "cpu"
CV_OUTPUT_DIR  = PROJECT_ROOT / "generated_models"
GAN_OUTPUT_DIR = PROJECT_ROOT / "gan_output"

from gan_part.inference.texture_generator import TextureGenerator
from gan_part.inference.gltf_texturer import apply_textures

# Lazy-loaded singleton
_tex_gen = None

def get_texture_generator() -> TextureGenerator:
    global _tex_gen
    if _tex_gen is None:
        _tex_gen = TextureGenerator(device=DEVICE)
    return _tex_gen


def process_floorplan(stem: str) -> Path | None:
    """
    Apply GAN textures to the CV .gltf for one floorplan.

    Args:
        stem: Floorplan stem name (e.g. 'floorplan1')

    Returns:
        Path to output _interior.gltf or None on failure.
    """
    cv_gltf = CV_OUTPUT_DIR / stem / f"{stem}.gltf"
    if not cv_gltf.exists():
        print(f"  ✗ CV gltf not found: {cv_gltf}")
        return None

    out_dir  = GAN_OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{stem}_interior.gltf")

    print(f"\n{'='*60}")
    print(f"GAN Interior: {stem}")
    print(f"  Input:  generated_models/{stem}/{stem}.gltf")
    print(f"  Output: gan_output/{stem}/{stem}_interior.gltf")
    print(f"{'='*60}")

    try:
        tex_gen = get_texture_generator()
        apply_textures(str(cv_gltf), tex_gen, out_path)
        print(f"\n  View at: https://gltf-viewer.donmccurdy.com")
        return Path(out_path)
    except Exception as e:
        import traceback
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Specific stems passed as arguments
        stems = sys.argv[1:]
    else:
        # All folders in generated_models/
        stems = sorted(
            d.name for d in CV_OUTPUT_DIR.iterdir()
            if d.is_dir() and (d / f"{d.name}.gltf").exists()
        )

    if not stems:
        print("❌ No CV gltf files found in generated_models/")
        print("   Run endToEnd2.py first to generate CV outputs.")
        sys.exit(1)

    print(f"Found {len(stems)} floorplan(s): {stems}")
    print(f"Backend: {BACKEND}\n")

    results = []
    for stem in stems:
        result = process_floorplan(stem)
        if result:
            results.append(result)

    print(f"\n{'='*60}")
    print(f"✓ Done — {len(results)}/{len(stems)} processed")
    print(f"  Output: gan_output/")
    print(f"  View:   https://gltf-viewer.donmccurdy.com")
    print(f"{'='*60}")
