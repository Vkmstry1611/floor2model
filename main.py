"""
main.py
-------
Combined pipeline — runs CV then GAN on all images in samples/.

  Step 1 (CV):  Detect walls/doors/windows → 3D building model (.gltf)
  Step 2 (GAN): Apply textures + furniture → interior model (_interior.gltf)

Usage:
    python main.py              # run both CV + GAN
    python main.py --cv-only    # run CV pipeline only
    python main.py --gan-only   # run GAN on existing CV outputs

Outputs:
    generated_models/<stem>/    ← CV outputs (.gltf, .obj, detections)
    gan_output/<stem>/          ← GAN outputs (_interior.gltf)
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from cv_pipeline import run_cv_pipeline, collect_samples
from gan_part.gan_runner import process_floorplan as run_gan_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Floor2Model — CV + GAN pipeline"
    )
    parser.add_argument(
        "--cv-only",  action="store_true",
        help="Run CV pipeline only (no GAN textures/furniture)"
    )
    parser.add_argument(
        "--gan-only", action="store_true",
        help="Run GAN pipeline only on existing CV outputs"
    )
    args = parser.parse_args()

    run_cv  = not args.gan_only
    run_gan = not args.cv_only

    samples = collect_samples()
    if not samples:
        print("❌ No image files found in samples/")
        sys.exit(1)

    print(f"\nFloor2Model Pipeline")
    print(f"{'='*60}")
    print(f"Images:   {len(samples)} in samples/")
    print(f"CV:       {'✓' if run_cv  else '—'}")
    print(f"GAN:      {'✓' if run_gan else '—'}")
    print(f"{'='*60}\n")

    cv_results  = {}   # stem → cv result dict
    gan_results = []

    # ── Step 1: CV pipeline ───────────────────────────────────────────────────
    if run_cv:
        print("STEP 1 — CV Pipeline")
        print("─" * 60)
        for sample in samples:
            result = run_cv_pipeline(sample)
            if result:
                cv_results[result["stem"]] = result

        print(f"\n✓ CV complete — {len(cv_results)}/{len(samples)} processed")
        print(f"  Output: generated_models/\n")

    # ── Step 2: GAN pipeline ──────────────────────────────────────────────────
    if run_gan:
        print("STEP 2 — GAN Pipeline")
        print("─" * 60)

        if run_cv and cv_results:
            # Called right after CV — pass seg_result directly (no re-detection)
            for stem, cv_data in cv_results.items():
                result = run_gan_pipeline(
                    stem=stem,
                    seg_result=cv_data["seg_result"],
                    image_path=cv_data["image_path"],
                )
                if result:
                    gan_results.append(result)
        else:
            # Standalone GAN — read from generated_models/
            generated = PROJECT_ROOT / "generated_models"
            stems = sorted(
                d.name for d in generated.iterdir()
                if d.is_dir() and (d / f"{d.name}.gltf").exists()
            ) if generated.exists() else []

            if not stems:
                print("❌ No CV outputs found. Run without --gan-only first.")
                sys.exit(1)

            for stem in stems:
                # Find matching sample image
                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
                    candidate = PROJECT_ROOT / "samples" / f"{stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                result = run_gan_pipeline(stem=stem, image_path=img_path)
                if result:
                    gan_results.append(result)

        print(f"\n✓ GAN complete — {len(gan_results)} processed")
        print(f"  Output: gan_output/")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("✓ All done")
    if run_cv:
        print(f"  CV models:       generated_models/")
    if run_gan:
        print(f"  Interior models: gan_output/")
        print(f"  View:            https://gltf-viewer.donmccurdy.com")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
