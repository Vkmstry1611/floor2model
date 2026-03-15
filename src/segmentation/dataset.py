"""
dataset.py
----------
Loads and converts the CubiCasa5k dataset into YOLOv8 segmentation format.

CubiCasa5k provides:
  - Floor plan images (PNG)
  - SVG annotations with labelled polygons per element class

We convert SVG → YOLO segmentation format:
  <class_id> <x1> <y1> <x2> <y2> ... (normalised 0-1 polygon coords)

Class map (14 classes):
  0  Background
  1  OuterWall
  2  InnerWall
  3  Window
  4  Door
  5  Stairs
  6  Railing
  7  Kitchen
  8  LivingRoom
  9  Bedroom
  10 Bathroom
  11 Corridor
  12 Balcony
  13 Garage

Usage:
    from src.segmentation.dataset import CubiCasaDataset
    ds = CubiCasaDataset("data/cubicasa5k")
    ds.prepare(output_dir="data/yolo_dataset")
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── Class definitions ─────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Background",   # 0
    "OuterWall",    # 1
    "InnerWall",    # 2
    "Window",       # 3
    "Door",         # 4
    "Stairs",       # 5
    "Railing",      # 6
    "Kitchen",      # 7
    "LivingRoom",   # 8
    "Bedroom",      # 9
    "Bathroom",     # 10
    "Corridor",     # 11
    "Balcony",      # 12
    "Garage",       # 13
]

# Map SVG class names → our integer IDs
SVG_CLASS_MAP = {
    "Wall":         1,
    "OuterWall":    1,
    "InnerWall":    2,
    "Window":       3,
    "Door":         4,
    "Stairs":       5,
    "Railing":      6,
    "Kitchen":      7,
    "LivingRoom":   8,
    "Living":       8,
    "Bedroom":      9,
    "Bathroom":     10,
    "Toilet":       10,
    "Corridor":     11,
    "Hallway":      11,
    "Balcony":      12,
    "Terrace":      12,
    "Garage":       13,
    "CarPort":      13,
}

NUM_CLASSES = len(CLASS_NAMES)


# ── Dataset class ─────────────────────────────────────────────────────────────

class CubiCasaDataset:
    """
    Converts CubiCasa5k dataset to YOLOv8 segmentation format.

    CubiCasa5k download:
        https://zenodo.org/record/2613548

    Args:
        root_dir:    Path to the extracted CubiCasa5k folder.
        val_split:   Fraction of data to use for validation.
        test_split:  Fraction of data to use for testing.
        seed:        Random seed for reproducible splits.
    """

    def __init__(
        self,
        root_dir: str,
        val_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
    ):
        self.root = Path(root_dir)
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {root_dir}\n"
                "Download CubiCasa5k from: https://zenodo.org/record/2613548"
            )

    def prepare(self, output_dir: str = "data/yolo_dataset") -> str:
        """
        Convert and split the dataset into train/val/test sets.

        Args:
            output_dir: Where to write the YOLO-formatted dataset.

        Returns:
            Path to the generated dataset.yaml file.
        """
        out = Path(output_dir)
        print(f"Preparing CubiCasa5k → YOLO format in: {out}")

        # Discover all floor plan samples
        samples = self._discover_samples()
        print(f"  Found {len(samples)} annotated floor plans")

        # Split into train / val / test
        splits = self._split(samples)
        for split_name, split_samples in splits.items():
            print(f"  {split_name}: {len(split_samples)} samples")

        # Convert and write each split
        for split_name, split_samples in splits.items():
            self._write_split(split_samples, out, split_name)

        # Write dataset.yaml
        yaml_path = self._write_yaml(out)
        print(f"\nDataset ready. Config: {yaml_path}")
        return str(yaml_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _discover_samples(self) -> list[dict]:
        """
        Find all (image, annotation) pairs in the dataset.
        CubiCasa5k stores each floor plan in its own subdirectory.
        """
        samples = []
        # CubiCasa5k structure: root/high_quality/<id>/F1_scaled.png + model.svg
        for subdir in sorted(self.root.rglob("F1_scaled.png")):
            img_path = subdir
            svg_path = subdir.parent / "model.svg"
            if svg_path.exists():
                samples.append({
                    "image": str(img_path),
                    "annotation": str(svg_path),
                })
        return samples

    def _split(self, samples: list[dict]) -> dict[str, list[dict]]:
        """Reproducible train/val/test split."""
        random.seed(self.seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)

        return {
            "test":  shuffled[:n_test],
            "val":   shuffled[n_test:n_test + n_val],
            "train": shuffled[n_test + n_val:],
        }

    def _write_split(
        self, samples: list[dict], out: Path, split: str
    ) -> None:
        """Convert samples and write images + labels to split directory."""
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        ok, skipped = 0, 0
        for sample in samples:
            try:
                stem = Path(sample["image"]).parent.name
                # Copy image
                dst_img = img_dir / f"{stem}.png"
                shutil.copy2(sample["image"], dst_img)

                # Parse SVG → YOLO label file
                img = Image.open(sample["image"])
                w, h = img.size
                polygons = parse_svg_annotations(sample["annotation"], w, h)

                if not polygons:
                    skipped += 1
                    continue

                dst_lbl = lbl_dir / f"{stem}.txt"
                write_yolo_labels(polygons, dst_lbl)
                ok += 1

            except Exception as e:
                print(f"  Warning: skipping {sample['image']}: {e}")
                skipped += 1

        print(f"    {split}: wrote {ok} labels, skipped {skipped}")

    def _write_yaml(self, out: Path) -> Path:
        """Write the YOLO dataset configuration YAML."""
        yaml_path = out / "dataset.yaml"
        content = f"""# CubiCasa5k — YOLOv8 segmentation dataset
path: {out.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {NUM_CLASSES - 1}  # exclude background (class 0)
names: {CLASS_NAMES[1:]}
"""
        yaml_path.write_text(content)
        return yaml_path


# ── SVG parsing ───────────────────────────────────────────────────────────────

def parse_svg_annotations(
    svg_path: str, img_w: int, img_h: int
) -> list[dict]:
    """
    Parse a CubiCasa5k SVG annotation file.

    Args:
        svg_path: Path to model.svg
        img_w:    Image width in pixels (for normalisation)
        img_h:    Image height in pixels (for normalisation)

    Returns:
        List of dicts: [{"class_id": int, "polygon": [(x, y), ...]}, ...]
        All coordinates normalised to [0, 1].
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid SVG: {svg_path}: {e}")

    ns = {"svg": "http://www.w3.org/2000/svg"}
    polygons = []

    # SVG viewBox gives us the coordinate system
    viewBox = root.get("viewBox", f"0 0 {img_w} {img_h}")
    vb = [float(v) for v in viewBox.split()]
    svg_w, svg_h = vb[2], vb[3]

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # strip namespace
        class_name = (
            elem.get("class", "") or
            elem.get("id", "").split("-")[0] or
            ""
        )

        class_id = SVG_CLASS_MAP.get(class_name)
        if class_id is None:
            continue

        pts = None
        if tag == "polygon":
            pts = _parse_polygon_points(elem.get("points", ""))
        elif tag == "polyline":
            pts = _parse_polygon_points(elem.get("points", ""))
        elif tag == "rect":
            pts = _rect_to_polygon(elem)
        elif tag == "path":
            pts = _path_to_polygon(elem.get("d", ""))

        if pts and len(pts) >= 3:
            # Normalise to [0, 1] relative to image size
            norm = [
                (
                    round(x / svg_w, 6),
                    round(y / svg_h, 6),
                )
                for x, y in pts
            ]
            polygons.append({"class_id": class_id, "polygon": norm})

    return polygons


def write_yolo_labels(polygons: list[dict], output_path: Path) -> None:
    """
    Write YOLO segmentation label file.
    Format per line: <class_id> <x1> <y1> <x2> <y2> ...
    """
    lines = []
    for ann in polygons:
        coords = " ".join(
            f"{x} {y}" for x, y in ann["polygon"]
        )
        lines.append(f"{ann['class_id'] - 1} {coords}")  # YOLO is 0-indexed

    output_path.write_text("\n".join(lines))


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _parse_polygon_points(points_str: str) -> list[tuple]:
    """Parse SVG 'points' attribute into list of (x, y) tuples."""
    try:
        vals = [float(v) for v in points_str.replace(",", " ").split()]
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
    except (ValueError, IndexError):
        return []


def _rect_to_polygon(elem) -> list[tuple]:
    """Convert SVG <rect> to 4-point polygon."""
    try:
        x = float(elem.get("x", 0))
        y = float(elem.get("y", 0))
        w = float(elem.get("width", 0))
        h = float(elem.get("height", 0))
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    except (ValueError, TypeError):
        return []


def _path_to_polygon(d: str) -> list[tuple]:
    """
    Naive SVG path → polygon converter.
    Only handles M/L/Z commands (absolute move/line/close).
    Sufficient for simple architectural polygons.
    """
    pts = []
    try:
        tokens = d.replace(",", " ").split()
        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            if cmd in ("M", "L"):
                x, y = float(tokens[i + 1]), float(tokens[i + 2])
                pts.append((x, y))
                i += 3
            elif cmd == "Z":
                i += 1
            else:
                i += 1
    except (IndexError, ValueError):
        pass
    return pts
