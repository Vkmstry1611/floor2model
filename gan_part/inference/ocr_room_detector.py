"""
ocr_room_detector.py
--------------------
Uses EasyOCR (or pytesseract fallback) to detect room labels
printed inside a floorplan image.

Returns a list of detected rooms with their pixel positions,
which are then matched to the 3D building footprint for
furniture placement.

Room label matching:
  "bed", "bedroom", "master" → Bedroom
  "living", "lounge", "family" → LivingRoom
  "kitchen", "kit", "dining" → Kitchen
  "bath", "wc", "toilet", "shower" → Bathroom
  "hall", "corridor", "entry", "foyer" → Corridor
  "garage", "car" → Garage
  "balcony", "terrace", "patio" → Balcony
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional


# ── Room label keyword mapping ────────────────────────────────────────────────

ROOM_KEYWORDS = {
    "Bedroom":    ["bed", "bedroom", "master", "guest", "mbr", "br"],
    "LivingRoom": ["living", "lounge", "family", "great", "sitting", "lr"],
    "Kitchen":    ["kitchen", "kit", "dining", "breakfast", "pantry"],
    "Bathroom":   ["bath", "wc", "toilet", "shower", "ensuite", "lavatory"],
    "Corridor":   ["hall", "corridor", "entry", "foyer", "landing", "passage"],
    "Garage":     ["garage", "car", "parking"],
    "Balcony":    ["balcony", "terrace", "patio", "deck", "veranda"],
    "Stairs":     ["stair", "stairs", "staircase", "steps"],
}


def classify_text(text: str) -> Optional[str]:
    """Match a text string to a room type."""
    t = text.lower().strip()
    for room_type, keywords in ROOM_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return room_type
    return None


def detect_rooms_ocr(image_path: str) -> list[dict]:
    """
    Run OCR on a floorplan image and return detected room labels
    with their pixel positions.

    Returns:
        List of dicts: {"label": "Bedroom", "cx": 120, "cy": 340, "text": "Bedroom 1"}
    """
    rooms = []

    # Try EasyOCR first (better for small/rotated text)
    try:
        rooms = _detect_easyocr(image_path)
        if rooms:
            print(f"  OCR (EasyOCR): found {len(rooms)} room labels")
            return rooms
    except Exception as e:
        print(f"  EasyOCR unavailable ({type(e).__name__})")

    # Fallback: pytesseract
    try:
        rooms = _detect_tesseract(image_path)
        if rooms:
            print(f"  OCR (tesseract): found {len(rooms)} room labels")
            return rooms
    except Exception as e:
        print(f"  Tesseract unavailable ({type(e).__name__})")

    print("  OCR: no room labels detected")
    return []


def _detect_easyocr(image_path: str) -> list[dict]:
    import easyocr
    import cv2

    reader = easyocr.Reader(['en'], verbose=False)
    img = cv2.imread(image_path)
    results = reader.readtext(img)

    rooms = []
    for (bbox_pts, text, conf) in results:
        if conf < 0.3:
            continue
        room_type = classify_text(text)
        if room_type is None:
            continue
        # bbox_pts is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        pts = np.array(bbox_pts)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        rooms.append({
            "label": room_type,
            "cx": cx,
            "cy": cy,
            "text": text,
            "conf": conf,
        })

    return rooms


def _detect_tesseract(image_path: str) -> list[dict]:
    import pytesseract
    import cv2

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get word-level bounding boxes
    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT,
        config='--psm 11',
    )

    rooms = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text or len(text) < 2:
            continue
        try:
            conf = float(data['conf'][i])
        except (ValueError, TypeError):
            conf = 0
        if conf < 30:
            continue

        room_type = classify_text(text)
        if room_type is None:
            continue

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        cx = x + w // 2
        cy = y + h // 2

        rooms.append({
            "label": room_type,
            "cx": cx,
            "cy": cy,
            "text": text,
            "conf": conf,
        })

    return rooms


def match_rooms_to_footprint(
    ocr_rooms: list[dict],
    img_w: int,
    img_h: int,
    wall_x1_px: int,
    wall_y1_px: int,
    wall_x2_px: int,
    wall_y2_px: int,
    build_x_min: float,
    build_x_max: float,
    build_y_min: float,
    build_y_max: float,
) -> list[dict]:
    """
    Convert OCR room pixel positions to 3D metre positions.

    Maps pixel coordinates (relative to wall extents) to
    the 3D building footprint in metres.

    Returns:
        List of dicts: {"label": "Bedroom", "x": 1.2, "y": -0.5}
        (x, y in metres, same coordinate system as the 3D model)
    """
    wall_w_px = wall_x2_px - wall_x1_px
    wall_h_px = wall_y2_px - wall_y1_px
    build_w_m = build_x_max - build_x_min
    build_h_m = build_y_max - build_y_min

    if wall_w_px <= 0 or wall_h_px <= 0:
        return []

    matched = []
    for room in ocr_rooms:
        # Normalise pixel position relative to wall extents
        rel_x = (room["cx"] - wall_x1_px) / wall_w_px
        rel_y = (room["cy"] - wall_y1_px) / wall_h_px

        # Clamp to [0,1]
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # Map to 3D metres (Y is flipped — image top = 3D far)
        x_m = build_x_min + rel_x * build_w_m
        y_m = build_y_max - rel_y * build_h_m  # flip Y

        matched.append({
            "label": room["label"],
            "x": x_m,
            "y": y_m,
            "text": room["text"],
        })

    return matched
