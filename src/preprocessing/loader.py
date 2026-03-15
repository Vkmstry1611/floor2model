"""
loader.py
---------
Handles loading floor plan images from disk.
Supports: PNG, JPG, JPEG, BMP, TIFF, PDF (first page via PIL).
Normalizes to a standard resolution while preserving aspect ratio.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"}


def load_image(image_path: str, target_size: int = 1024) -> np.ndarray:
    """
    Load a floor plan image and normalize it to a standard size.

    Args:
        image_path: Path to the input image file.
        target_size: The longer dimension will be resized to this value.
                     Aspect ratio is preserved.

    Returns:
        Grayscale numpy array of shape (H, W), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    # PDF: extract first page as image
    if path.suffix.lower() == ".pdf":
        img = _load_pdf_page(path)
    else:
        img = _load_raster(path)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to target_size along the longer dimension
    img = _resize_keep_aspect(img, target_size)

    return img


def _load_raster(path: Path) -> np.ndarray:
    """Load PNG/JPG/BMP/TIFF using OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"OpenCV could not read image: {path}")
    return img


def _load_pdf_page(path: Path, page: int = 0, dpi: int = 200) -> np.ndarray:
    """
    Load the first page of a PDF as a numpy array using PIL.
    Requires Pillow with PDF support.
    """
    try:
        pil_img = Image.open(str(path))
        pil_img.load()
        # For multi-page PDFs, seek to desired page
        if hasattr(pil_img, "n_frames") and pil_img.n_frames > page:
            pil_img.seek(page)
        pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise IOError(f"Failed to load PDF '{path}': {e}")


def _resize_keep_aspect(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize image so its longer dimension equals target_size.
    Uses INTER_AREA for downscaling (best quality for line drawings).
    """
    h, w = img.shape[:2]
    if max(h, w) == target_size:
        return img

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def save_image(img: np.ndarray, output_path: str) -> None:
    """Save a numpy array as an image file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
