"""
gan_pipeline.py
---------------
GAN Pipeline — reads CV output from generated_models/ and produces
realistic textured 3D interiors in gan_output/<stem>/.

Input:  generated_models/<stem>/<stem>.gltf   (from cv_pipeline.py)
Output: gan_output/<stem>/<stem>_interior.gltf

Each output .gltf contains:
  - Original CV walls / doors / windows geometry
  - GAN-generated textures on all surfaces (wall, floor, door, window)
  - Procedural furniture placed inside the building footprint

Backend:
  "controlnet"  — Stable Diffusion + ControlNet (zero-shot, no training needed)
  "gan"         — Your trained Pix2Pix (set GAN_CHECKPOINT when ready)

Usage:
    python gan_pipeline.py                    # all stems in generated_models/
    python gan_pipeline.py floorplan1         # single stem
    python gan_pipeline.py floorplan1 sample3 # multiple stems
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Config — swap BACKEND + GAN_CHECKPOINT when your model is ready ───────────
BACKEND        = "controlnet"   # "controlnet" | "gan"
GAN_CHECKPOINT = None           # "gan_part/checkpoints/final.pt"
DEVICE         = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
CV_OUTPUT_DIR  = PROJECT_ROOT / "generated_models"
GAN_OUTPUT_DIR = PROJECT_ROOT / "gan_output"

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from gan_part.inference.texture_generator import TextureGenerator
    from gan_part.inference.gltf_texturer import apply_textures
    GAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ GAN dependencies not installed: {e}")
    print("  Run: pip install -r gan_part/requirements.txt")
    GAN_AVAILABLE = False

# Lazy-loaded texture generator (shared across all floorplans)
_tex_gen: Optional["TextureGenerator"] = None


def get_texture_generator() -> "TextureGenerator":
    global _tex_gen
    if _tex_gen is None:
        print(f"  Loading texture generator (backend={BACKEND})...")
        _tex_gen = TextureGenerator(device=DEVICE)
    return _tex_gen


def collect_stems(stems_arg: list[str]) -> list[str]:
    """
    Collect stems to process.
    If stems_arg is empty, find all valid CV output folders.
    """
    if stems_arg:
        return stems_arg

    return sorted(
        d.name for d in CV_OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.gltf").exists()
    )


def run_gan(stem: str) -> Optional[Path]:
    """
    Apply GAN textures + furniture to the CV .gltf for one floorplan.

    Args:
        stem: Floorplan stem (e.g. 'floorplan1') — must exist in generated_models/

    Returns:
        Path to output _interior.gltf, or None on failure.
    """
    if not GAN_AVAILABLE:
        print("❌ GAN dependencies not available.")
        return None

    cv_gltf = CV_OUTPUT_DIR / stem / f"{stem}.gltf"
    if not cv_gltf.exists():
        print(f"  ✗ CV output not found: {cv_gltf}")
        print(f"    Run cv_pipeline.py first.")
        return None

    out_dir  = GAN_OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{stem}_interior.gltf")

    print(f"\n{'='*60}")
    print(f"GAN Pipeline: {stem}")
    print(f"  Input:  generated_models/{stem}/{stem}.gltf")
    print(f"  Output: gan_output/{stem}/{stem}_interior.gltf")
    print(f"  Backend: {BACKEND}")
    print(f"{'='*60}")

    try:
        tex_gen = get_texture_generator()
        apply_textures(str(cv_gltf), tex_gen, out_path)
        print(f"\n  ✓ Done → gan_output/{stem}/{stem}_interior.gltf")
        print(f"  View:  https://gltf-viewer.donmccurdy.com")
        return Path(out_path)
    except Exception as e:
        import traceback
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Accept optional stem arguments
    stems = collect_stems(sys.argv[1:])

    if not stems:
        print("❌ No CV outputs found in generated_models/")
        print("   Run cv_pipeline.py first to generate CV outputs.")
        sys.exit(1)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Backend:      {BACKEND}")
    print(f"Found {len(stems)} CV output(s): {stems}\n")

    success, failed = 0, 0
    for stem in stems:
        result = run_gan(stem)
        if result:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"✓ GAN Pipeline complete — {success}/{len(stems)} processed")
    print(f"  Output: gan_output/")
    print(f"  View:   https://gltf-viewer.donmccurdy.com")
    print(f"{'='*60}")
