"""
main.py
-------
Entry point for the Phase 1 preprocessing pipeline.

Usage:
    # Single image
    python main.py --input samples/floor_plan.png

    # All images in a folder
    python main.py --input samples/ --output outputs/

    # Custom settings
    python main.py --input samples/plan.png --size 2048 --no-skew
"""

import argparse
from pathlib import Path
from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Floor plan preprocessing pipeline (Phase 1)"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a floor plan image or a folder of images."
    )
    parser.add_argument(
        "--output", "-o", default="outputs/",
        help="Directory to save processed outputs. Default: outputs/"
    )
    parser.add_argument(
        "--size", type=int, default=1024,
        help="Target size for the longer dimension (pixels). Default: 1024"
    )
    parser.add_argument(
        "--no-skew", action="store_true",
        help="Disable skew correction."
    )
    parser.add_argument(
        "--no-contrast", action="store_true",
        help="Disable CLAHE contrast enhancement."
    )
    return parser.parse_args()


def collect_images(input_path: str) -> list[str]:
    """Return list of image file paths from a file or directory."""
    p = Path(input_path)
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        found = [str(f) for f in p.iterdir() if f.suffix.lower() in supported]
        if not found:
            print(f"No supported images found in: {input_path}")
        return sorted(found)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main():
    args = parse_args()

    config = PreprocessingConfig(
        target_size=args.size,
        correct_skew=not args.no_skew,
        enhance_contrast=not args.no_contrast,
    )

    pipeline = PreprocessingPipeline(config=config)
    image_paths = collect_images(args.input)

    print(f"Found {len(image_paths)} image(s) to process.")
    print(f"Config: size={config.target_size}, skew={config.correct_skew}, "
          f"contrast={config.enhance_contrast}\n")

    if len(image_paths) == 1:
        result = pipeline.run(image_paths[0])
        result.save(args.output)
    else:
        pipeline.run_batch(image_paths, args.output)

    print("\nAll done!")


if __name__ == "__main__":
    main()
