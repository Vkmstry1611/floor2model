"""
main.py
-------
Full pipeline — CV + GAN combined.

Runs cv_pipeline.py then gan_pipeline.py on all images in samples/.

Usage:
    python main.py                    # process all images in samples/
    python main.py samples/plan.png   # single image

Pipeline:
    samples/<image>
        ↓  cv_pipeline.py
    generated_models/<stem>/
        <stem>_detections.png   — YOLO detection overlay
        <stem>.gltf             — 3D building (walls/doors/windows)
        <stem>.obj
        ↓  gan_pipeline.py
    gan_output/<stem>/
        <stem>_interior.gltf    — 3D building + GAN textures + furniture
"""

from __future__ import annotations

import sys
import os
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def _load(name: str) -> object:
    """Load a pipeline module by filename from the project root."""
    spec = importlib.util.spec_from_file_location(
        name, str(PROJECT_ROOT / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cv  = _load("cv_pipeline")
_gan = _load("gan_pipeline")

collect_samples = _cv.collect_samples
run_cv          = _cv.run_cv
SAMPLES_DIR     = _cv.SAMPLES_DIR
IMAGE_EXTS      = _cv.IMAGE_EXTS
run_gan         = _gan.run_gan
GAN_AVAILABLE   = _gan.GAN_AVAILABLE


def main():
    # ── Collect input images ──────────────────────────────────────────────────
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

    print(f"{'='*60}")
    print(f"Floor2Model — Full Pipeline")
    print(f"{'='*60}")
    print(f"Found {len(images)} image(s): {[i.name for i in images]}")
    print(f"GAN available: {GAN_AVAILABLE}\n")

    cv_results  = []   # stems that CV succeeded on
    gan_results = []   # stems that GAN succeeded on

    # ── Phase 1: CV pipeline for all images ───────────────────────────────────
    print(f"\n{'─'*60}")
    print("STAGE 1 — CV Pipeline (detection + geometry + 3D model)")
    print(f"{'─'*60}")

    for img in images:
        out_dir = run_cv(img)
        if out_dir:
            cv_results.append(img.stem)

    print(f"\n✓ CV complete: {len(cv_results)}/{len(images)} succeeded")

    if not cv_results:
        print("❌ No CV outputs — cannot run GAN pipeline.")
        sys.exit(1)

    # ── Phase 2: GAN pipeline for all successful CV outputs ───────────────────
    if not GAN_AVAILABLE:
        print("\n⚠ GAN pipeline skipped — dependencies not installed.")
        print("  Run: pip install -r gan_part/requirements.txt")
    else:
        print(f"\n{'─'*60}")
        print("STAGE 2 — GAN Pipeline (textures + furniture)")
        print(f"{'─'*60}")

        for stem in cv_results:
            result = run_gan(stem)
            if result:
                gan_results.append(stem)

        print(f"\n✓ GAN complete: {len(gan_results)}/{len(cv_results)} succeeded")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  CV outputs  → generated_models/   ({len(cv_results)} floorplans)")
    if GAN_AVAILABLE:
        print(f"  GAN outputs → gan_output/          ({len(gan_results)} floorplans)")
    print(f"\n  View 3D models: https://gltf-viewer.donmccurdy.com")
    print(f"    Drag any .gltf file from generated_models/ or gan_output/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
