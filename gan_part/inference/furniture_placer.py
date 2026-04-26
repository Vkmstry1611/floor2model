"""
furniture_placer.py
-------------------
Places furniture inside the building footprint with correct sizes
and no overlaps. Uses a simple occupied-zone system to prevent
furniture from overlapping each other or walls.

All dimensions in metres. Furniture is placed 0.3m from walls.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


# ── Box primitive ─────────────────────────────────────────────────────────────

def _box(cx, cy, cz, w, d, h, color) -> dict:
    hw, hd = w/2, d/2
    v = np.array([
        [cx-hw, cy-hd, cz],   [cx+hw, cy-hd, cz],
        [cx+hw, cy+hd, cz],   [cx-hw, cy+hd, cz],
        [cx-hw, cy-hd, cz+h], [cx+hw, cy-hd, cz+h],
        [cx+hw, cy+hd, cz+h], [cx-hw, cy+hd, cz+h],
    ], dtype=np.float32)
    f = np.array([
        [0,2,1],[0,3,2],  # bottom
        [4,5,6],[4,6,7],  # top
        [0,1,5],[0,5,4],  # front
        [2,3,7],[2,7,6],  # back
        [3,0,4],[3,4,7],  # left
        [1,2,6],[1,6,5],  # right
    ], dtype=np.int32)
    return {"vertices": v, "faces": f, "color": np.array(color, dtype=np.float32)}


def _merge(boxes):
    verts, faces, colors = [], [], []
    off = 0
    for b in boxes:
        verts.append(b["vertices"])
        faces.append(b["faces"] + off)
        colors.append(np.tile(b["color"], (len(b["vertices"]), 1)))
        off += len(b["vertices"])
    return {
        "vertices": np.vstack(verts).astype(np.float32),
        "faces":    np.vstack(faces).astype(np.int32),
        "colors":   np.vstack(colors).astype(np.float32),
    }


# ── Colors ────────────────────────────────────────────────────────────────────
GREY    = [0.52, 0.52, 0.55]
BEIGE   = [0.78, 0.70, 0.58]
W_LIGHT = [0.62, 0.44, 0.26]
W_DARK  = [0.32, 0.20, 0.10]
WHITE   = [0.94, 0.92, 0.90]
BLUE_G  = [0.86, 0.86, 0.90]
METAL   = [0.28, 0.28, 0.30]


# ── Furniture builders (realistic AEC sizes) ──────────────────────────────────

def sofa(cx, cy):
    """2-seat sofa: 1.6m wide × 0.75m deep × 0.85m tall"""
    return _merge([
        _box(cx,      cy,      0.00, 1.60, 0.75, 0.40, GREY),   # seat
        _box(cx,      cy+0.33, 0.40, 1.60, 0.12, 0.45, GREY),   # backrest
        _box(cx-0.74, cy,      0.00, 0.12, 0.75, 0.60, GREY),   # left arm
        _box(cx+0.74, cy,      0.00, 0.12, 0.75, 0.60, GREY),   # right arm
        _box(cx-0.45, cy-0.05, 0.40, 0.50, 0.62, 0.10, BEIGE),  # cushion L
        _box(cx+0.45, cy-0.05, 0.40, 0.50, 0.62, 0.10, BEIGE),  # cushion R
        _box(cx-0.70, cy-0.34, 0.00, 0.05, 0.05, 0.10, W_DARK), # leg FL
        _box(cx+0.70, cy-0.34, 0.00, 0.05, 0.05, 0.10, W_DARK), # leg FR
        _box(cx-0.70, cy+0.34, 0.00, 0.05, 0.05, 0.10, W_DARK), # leg BL
        _box(cx+0.70, cy+0.34, 0.00, 0.05, 0.05, 0.10, W_DARK), # leg BR
    ])


def coffee_table(cx, cy):
    """Coffee table: 0.9m × 0.45m × 0.42m"""
    return _merge([
        _box(cx, cy, 0.38, 0.90, 0.45, 0.04, W_LIGHT),
        _box(cx-0.41, cy-0.19, 0.00, 0.04, 0.04, 0.38, W_DARK),
        _box(cx+0.41, cy-0.19, 0.00, 0.04, 0.04, 0.38, W_DARK),
        _box(cx-0.41, cy+0.19, 0.00, 0.04, 0.04, 0.38, W_DARK),
        _box(cx+0.41, cy+0.19, 0.00, 0.04, 0.04, 0.38, W_DARK),
    ])


def dining_table(cx, cy):
    """Dining table: 1.4m × 0.8m × 0.75m"""
    return _merge([
        _box(cx, cy, 0.71, 1.40, 0.80, 0.04, W_LIGHT),
        _box(cx-0.64, cy-0.36, 0.00, 0.05, 0.05, 0.71, W_DARK),
        _box(cx+0.64, cy-0.36, 0.00, 0.05, 0.05, 0.71, W_DARK),
        _box(cx-0.64, cy+0.36, 0.00, 0.05, 0.05, 0.71, W_DARK),
        _box(cx+0.64, cy+0.36, 0.00, 0.05, 0.05, 0.71, W_DARK),
    ])


def chair(cx, cy):
    """Dining chair: 0.45m × 0.45m × 0.85m"""
    return _merge([
        _box(cx, cy, 0.44, 0.42, 0.42, 0.04, W_LIGHT),
        _box(cx, cy+0.18, 0.44, 0.42, 0.04, 0.38, W_LIGHT),
        _box(cx-0.17, cy-0.17, 0.00, 0.04, 0.04, 0.44, W_DARK),
        _box(cx+0.17, cy-0.17, 0.00, 0.04, 0.04, 0.44, W_DARK),
        _box(cx-0.17, cy+0.17, 0.00, 0.04, 0.04, 0.44, W_DARK),
        _box(cx+0.17, cy+0.17, 0.00, 0.04, 0.04, 0.44, W_DARK),
    ])


def bed(cx, cy):
    """Double bed: 1.5m × 2.0m × 0.55m"""
    return _merge([
        _box(cx, cy, 0.00, 1.50, 2.00, 0.22, W_LIGHT),   # base
        _box(cx, cy, 0.22, 1.46, 1.96, 0.20, WHITE),      # mattress
        _box(cx, cy+0.92, 0.00, 1.50, 0.08, 0.80, W_DARK),# headboard
        _box(cx-0.33, cy+0.55, 0.42, 0.52, 0.32, 0.10, WHITE),  # pillow L
        _box(cx+0.33, cy+0.55, 0.42, 0.52, 0.32, 0.10, WHITE),  # pillow R
        _box(cx, cy-0.15, 0.42, 1.42, 1.25, 0.07, BLUE_G),      # duvet
    ])


def wardrobe(cx, cy):
    """Wardrobe: 1.0m × 0.50m × 2.0m"""
    return _merge([
        _box(cx, cy, 0.00, 1.00, 0.50, 2.00, W_LIGHT),
        _box(cx-0.24, cy-0.24, 0.05, 0.48, 0.02, 1.88, W_DARK),
        _box(cx+0.24, cy-0.24, 0.05, 0.48, 0.02, 1.88, W_DARK),
        _box(cx-0.01, cy-0.25, 1.00, 0.04, 0.02, 0.10, METAL),
        _box(cx+0.49, cy-0.25, 1.00, 0.04, 0.02, 0.10, METAL),
    ])


def bookshelf(cx, cy):
    """Bookshelf: 0.7m × 0.25m × 1.6m"""
    boxes = [
        _box(cx, cy, 0.00, 0.70, 0.25, 1.60, W_DARK),
        _box(cx, cy, 0.35, 0.66, 0.23, 0.03, W_LIGHT),
        _box(cx, cy, 0.75, 0.66, 0.23, 0.03, W_LIGHT),
        _box(cx, cy, 1.15, 0.66, 0.23, 0.03, W_LIGHT),
        _box(cx, cy, 1.50, 0.66, 0.23, 0.03, W_LIGHT),
    ]
    book_colors = [[0.75,0.18,0.18],[0.18,0.45,0.75],[0.18,0.65,0.28],[0.75,0.55,0.10]]
    offsets = [-0.22, -0.08, 0.06, 0.20]
    for i, (ox, bc) in enumerate(zip(offsets, book_colors)):
        boxes.append(_box(cx+ox, cy, 0.38, 0.10, 0.20, 0.28+i*0.03, bc))
    return _merge(boxes)


# ── Overlap checker ───────────────────────────────────────────────────────────

class OccupancyGrid:
    """Simple 2D grid to prevent furniture overlaps."""

    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.1):
        self.x_min = x_min
        self.y_min = y_min
        self.res   = resolution
        self.nx    = int((x_max - x_min) / resolution) + 1
        self.ny    = int((y_max - y_min) / resolution) + 1
        self.grid  = np.zeros((self.nx, self.ny), dtype=bool)

    def _to_idx(self, x, y):
        ix = int((x - self.x_min) / self.res)
        iy = int((y - self.y_min) / self.res)
        return (
            max(0, min(ix, self.nx-1)),
            max(0, min(iy, self.ny-1)),
        )

    def is_free(self, cx, cy, w, d, margin=0.15) -> bool:
        """Check if a w×d box centred at (cx,cy) is free (with margin)."""
        x1, y1 = cx - w/2 - margin, cy - d/2 - margin
        x2, y2 = cx + w/2 + margin, cy + d/2 + margin
        ix1, iy1 = self._to_idx(x1, y1)
        ix2, iy2 = self._to_idx(x2, y2)
        return not self.grid[ix1:ix2+1, iy1:iy2+1].any()

    def occupy(self, cx, cy, w, d, margin=0.10):
        x1, y1 = cx - w/2 - margin, cy - d/2 - margin
        x2, y2 = cx + w/2 + margin, cy + d/2 + margin
        ix1, iy1 = self._to_idx(x1, y1)
        ix2, iy2 = self._to_idx(x2, y2)
        self.grid[ix1:ix2+1, iy1:iy2+1] = True

    def try_place(self, cx, cy, w, d) -> bool:
        """Try to place — returns True if placed, False if blocked."""
        if self.is_free(cx, cy, w, d):
            self.occupy(cx, cy, w, d)
            return True
        return False


# ── Main placement ────────────────────────────────────────────────────────────

def place_furniture(x_min, x_max, y_min, y_max, ocr_rooms=None) -> List[dict]:
    """
    Place furniture inside the building footprint.
    Uses OCR room positions if available, otherwise grid placement.
    Keeps 0.3m clearance from walls, no overlaps.
    """
    WALL_CLEAR = 0.30

    ix1 = x_min + WALL_CLEAR
    ix2 = x_max - WALL_CLEAR
    iy1 = y_min + WALL_CLEAR
    iy2 = y_max - WALL_CLEAR

    W = ix2 - ix1
    H = iy2 - iy1

    grid = OccupancyGrid(ix1, ix2, iy1, iy2)
    items = []

    def try_add(builder, cx, cy, w, d):
        cx = max(ix1 + w/2, min(ix2 - w/2, cx))
        cy = max(iy1 + d/2, min(iy2 - d/2, cy))
        if grid.try_place(cx, cy, w, d):
            items.append(builder(cx, cy))
            return True
        return False

    # ── OCR-guided placement ──────────────────────────────────────────────────
    if ocr_rooms:
        print(f"  OCR-guided furniture: {len(ocr_rooms)} rooms detected")
        for room in ocr_rooms:
            label = room["label"]
            rx = max(ix1, min(ix2, room["x"]))
            ry = max(iy1, min(iy2, room["y"]))

            if label == "Bedroom":
                try_add(bed,      rx,       ry + 0.5, 1.50, 2.00)
                try_add(wardrobe, rx + 1.0, ry - 0.8, 1.00, 0.50)

            elif label == "LivingRoom":
                try_add(sofa,         rx - 0.3, ry + 0.2, 1.60, 0.75)
                try_add(coffee_table, rx - 0.3, ry - 0.6, 0.90, 0.45)
                try_add(bookshelf,    rx + 1.2, ry + 0.5, 0.70, 0.25)

            elif label == "Kitchen":
                try_add(dining_table, rx,        ry,        1.40, 0.80)
                try_add(chair,        rx - 0.85, ry,        0.45, 0.45)
                try_add(chair,        rx + 0.85, ry,        0.45, 0.45)
                try_add(chair,        rx,        ry - 0.65, 0.45, 0.45)
                try_add(chair,        rx,        ry + 0.65, 0.45, 0.45)

        return items

    # ── Fallback: grid-based placement ───────────────────────────────────────
    print("  Grid-based furniture placement (no OCR room data)")

    sx = ix1 + 1.0
    sy = iy1 + 0.55
    try_add(sofa, sx, sy, 1.60, 0.75)
    try_add(coffee_table, sx, sy + 0.75, 0.90, 0.45)

    if W > 3.5:
        tx = ix2 - 1.0
        ty = iy1 + 0.65
        if try_add(dining_table, tx, ty, 1.40, 0.80):
            try_add(chair, tx - 0.85, ty, 0.45, 0.45)
            try_add(chair, tx + 0.85, ty, 0.45, 0.45)
            try_add(chair, tx, ty - 0.65, 0.45, 0.45)
            try_add(chair, tx, ty + 0.65, 0.45, 0.45)

    if H > 3.5:
        try_add(bed, ix1 + 1.0, iy2 - 1.2, 1.50, 2.00)

    if W > 2.5 and H > 3.0:
        try_add(wardrobe, ix2 - 0.6, iy2 - 0.35, 1.00, 0.50)

    if W > 2.0:
        try_add(bookshelf, ix2 - 0.45, iy1 + 0.5, 0.70, 0.25)

    return items
