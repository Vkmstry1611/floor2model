"""
furniture_placer.py
-------------------
Places furniture inside detected room spaces.

Uses room_detector.py to find actual open floor areas, then places
furniture only within those zones. Each room gets different furniture
based on its size and index — so different floorplans look different.
"""

from __future__ import annotations

import numpy as np
from typing import List


# ── Box primitive ─────────────────────────────────────────────────────────────

def _box(cx, cy, cz, w, d, h, color) -> dict:
    """Box mesh centred at (cx, cy) with bottom at cz."""
    hw, hd = w / 2, d / 2
    verts = np.array([
        [cx-hw, cy-hd, cz],     [cx+hw, cy-hd, cz],
        [cx+hw, cy+hd, cz],     [cx-hw, cy+hd, cz],
        [cx-hw, cy-hd, cz+h],   [cx+hw, cy-hd, cz+h],
        [cx+hw, cy+hd, cz+h],   [cx-hw, cy+hd, cz+h],
    ], dtype=np.float32)
    faces = np.array([
        [0,2,1],[0,3,2],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [2,3,7],[2,7,6],
        [3,0,4],[3,4,7],
        [1,2,6],[1,6,5],
    ], dtype=np.int32)
    return {"vertices": verts, "faces": faces,
            "color": np.array(color, dtype=np.float32)}


def _merge(boxes: List[dict]) -> dict:
    all_v, all_f, all_c = [], [], []
    offset = 0
    for b in boxes:
        all_v.append(b["vertices"])
        all_f.append(b["faces"] + offset)
        n = len(b["vertices"])
        all_c.append(np.tile(b["color"], (n, 1)))
        offset += n
    return {
        "vertices": np.vstack(all_v).astype(np.float32),
        "faces":    np.vstack(all_f).astype(np.int32),
        "colors":   np.vstack(all_c).astype(np.float32),
    }


# ── Colors ────────────────────────────────────────────────────────────────────
GREY_FABRIC   = [0.55, 0.55, 0.58]
WOOD_LIGHT    = [0.65, 0.45, 0.25]
WOOD_DARK     = [0.35, 0.22, 0.12]
WHITE_LINEN   = [0.95, 0.93, 0.90]
METAL_DARK    = [0.25, 0.25, 0.28]
CUSHION_BEIGE = [0.80, 0.72, 0.60]
BLUE_FABRIC   = [0.30, 0.40, 0.65]
GREEN_PLANT   = [0.20, 0.55, 0.20]


# ── Furniture builders ────────────────────────────────────────────────────────

def make_sofa(cx, cy, w=1.6, d=0.80) -> dict:
    hw, hd = w/2, d/2
    return _merge([
        _box(cx, cy, 0.0, w, d, 0.42, GREY_FABRIC),
        _box(cx, cy+hd-0.07, 0.42, w, 0.10, 0.48, GREY_FABRIC),
        _box(cx-hw+0.05, cy, 0.0, 0.09, d, 0.62, GREY_FABRIC),
        _box(cx+hw-0.05, cy, 0.0, 0.09, d, 0.62, GREY_FABRIC),
        _box(cx-w/3, cy-0.04, 0.42, w/3-0.03, d-0.18, 0.11, CUSHION_BEIGE),
        _box(cx,     cy-0.04, 0.42, w/3-0.03, d-0.18, 0.11, CUSHION_BEIGE),
        _box(cx+w/3, cy-0.04, 0.42, w/3-0.03, d-0.18, 0.11, CUSHION_BEIGE),
        _box(cx-hw+0.10, cy-hd+0.07, 0.0, 0.06, 0.06, 0.10, WOOD_DARK),
        _box(cx+hw-0.10, cy-hd+0.07, 0.0, 0.06, 0.06, 0.10, WOOD_DARK),
        _box(cx-hw+0.10, cy+hd-0.07, 0.0, 0.06, 0.06, 0.10, WOOD_DARK),
        _box(cx+hw-0.10, cy+hd-0.07, 0.0, 0.06, 0.06, 0.10, WOOD_DARK),
    ])


def make_coffee_table(cx, cy, w=0.80, d=0.40) -> dict:
    return _merge([
        _box(cx, cy, 0.36, w, d, 0.04, WOOD_LIGHT),
        _box(cx-w/2+0.06, cy-d/2+0.06, 0.0, 0.05, 0.05, 0.36, WOOD_DARK),
        _box(cx+w/2-0.06, cy-d/2+0.06, 0.0, 0.05, 0.05, 0.36, WOOD_DARK),
        _box(cx-w/2+0.06, cy+d/2-0.06, 0.0, 0.05, 0.05, 0.36, WOOD_DARK),
        _box(cx+w/2-0.06, cy+d/2-0.06, 0.0, 0.05, 0.05, 0.36, WOOD_DARK),
    ])


def make_tv_unit(cx, cy, w=1.1, d=0.32) -> dict:
    return _merge([
        _box(cx, cy, 0.0, w, d, 0.42, WOOD_DARK),
        _box(cx, cy-d/2+0.01, 0.04, w-0.04, 0.02, 0.34, [0.10,0.10,0.12]),
        _box(cx, cy-d/2-0.01, 0.42, w*0.50, 0.04, 0.50, [0.08,0.08,0.10]),
    ])


def make_dining_table(cx, cy, w=1.1, d=0.70) -> dict:
    return _merge([
        _box(cx, cy, 0.70, w, d, 0.04, WOOD_LIGHT),
        _box(cx-w/2+0.07, cy-d/2+0.07, 0.0, 0.06, 0.06, 0.70, WOOD_DARK),
        _box(cx+w/2-0.07, cy-d/2+0.07, 0.0, 0.06, 0.06, 0.70, WOOD_DARK),
        _box(cx-w/2+0.07, cy+d/2-0.07, 0.0, 0.06, 0.06, 0.70, WOOD_DARK),
        _box(cx+w/2-0.07, cy+d/2-0.07, 0.0, 0.06, 0.06, 0.70, WOOD_DARK),
    ])


def make_chair(cx, cy) -> dict:
    return _merge([
        _box(cx, cy, 0.44, 0.38, 0.38, 0.04, WOOD_LIGHT),
        _box(cx, cy+0.17, 0.44, 0.38, 0.04, 0.36, WOOD_LIGHT),
        _box(cx-0.15, cy-0.15, 0.0, 0.04, 0.04, 0.44, WOOD_DARK),
        _box(cx+0.15, cy-0.15, 0.0, 0.04, 0.04, 0.44, WOOD_DARK),
        _box(cx-0.15, cy+0.15, 0.0, 0.04, 0.04, 0.44, WOOD_DARK),
        _box(cx+0.15, cy+0.15, 0.0, 0.04, 0.04, 0.44, WOOD_DARK),
    ])


def make_bed(cx, cy, w=1.35, d=1.85) -> dict:
    return _merge([
        _box(cx, cy, 0.0, w, d, 0.22, WOOD_LIGHT),
        _box(cx, cy, 0.22, w-0.04, d-0.04, 0.20, WHITE_LINEN),
        _box(cx, cy+d/2-0.06, 0.0, w, 0.09, 0.75, WOOD_DARK),
        _box(cx-0.30, cy+d/2-0.38, 0.42, 0.48, 0.30, 0.11, WHITE_LINEN),
        _box(cx+0.30, cy+d/2-0.38, 0.42, 0.48, 0.30, 0.11, WHITE_LINEN),
        _box(cx, cy-d/2+0.42, 0.42, w-0.08, d*0.52, 0.07, [0.88,0.88,0.92]),
    ])


def make_wardrobe(cx, cy, w=0.90, d=0.45) -> dict:
    return _merge([
        _box(cx, cy, 0.0, w, d, 1.95, WOOD_LIGHT),
        _box(cx-w/4, cy-d/2+0.01, 0.10, w/2-0.02, 0.02, 1.75, WOOD_DARK),
        _box(cx+w/4, cy-d/2+0.01, 0.10, w/2-0.02, 0.02, 1.75, WOOD_DARK),
        _box(cx-0.02, cy-d/2+0.01, 0.95, 0.04, 0.02, 0.10, METAL_DARK),
        _box(cx+w/2-0.04, cy-d/2+0.01, 0.95, 0.04, 0.02, 0.10, METAL_DARK),
    ])


def make_bookshelf(cx, cy, w=0.65, d=0.22) -> dict:
    return _merge([
        _box(cx, cy, 0.0, w, d, 1.60, WOOD_DARK),
        _box(cx, cy, 0.36, w-0.04, d-0.02, 0.03, WOOD_LIGHT),
        _box(cx, cy, 0.72, w-0.04, d-0.02, 0.03, WOOD_LIGHT),
        _box(cx, cy, 1.08, w-0.04, d-0.02, 0.03, WOOD_LIGHT),
        _box(cx, cy, 1.44, w-0.04, d-0.02, 0.03, WOOD_LIGHT),
        _box(cx-0.20, cy, 0.39, 0.08, d-0.04, 0.26, [0.80,0.20,0.20]),
        _box(cx-0.09, cy, 0.39, 0.07, d-0.04, 0.31, [0.20,0.50,0.80]),
        _box(cx+0.03, cy, 0.39, 0.09, d-0.04, 0.24, [0.20,0.70,0.30]),
        _box(cx+0.14, cy, 0.39, 0.07, d-0.04, 0.28, [0.80,0.60,0.10]),
    ])


def make_desk(cx, cy, w=1.0, d=0.50) -> dict:
    return _merge([
        _box(cx, cy, 0.72, w, d, 0.03, WOOD_LIGHT),
        _box(cx-w/2+0.04, cy-d/2+0.04, 0.0, 0.05, 0.05, 0.72, WOOD_DARK),
        _box(cx+w/2-0.04, cy-d/2+0.04, 0.0, 0.05, 0.05, 0.72, WOOD_DARK),
        _box(cx-w/2+0.04, cy+d/2-0.04, 0.0, 0.05, 0.05, 0.72, WOOD_DARK),
        _box(cx+w/2-0.04, cy+d/2-0.04, 0.0, 0.05, 0.05, 0.72, WOOD_DARK),
        # Monitor
        _box(cx, cy+d/2-0.10, 0.72, 0.40, 0.04, 0.28, [0.10,0.10,0.12]),
    ])


def make_plant(cx, cy) -> dict:
    return _merge([
        _box(cx, cy, 0.0, 0.22, 0.22, 0.25, [0.55, 0.35, 0.20]),
        _box(cx, cy, 0.25, 0.18, 0.18, 0.05, [0.45, 0.28, 0.15]),
        _box(cx, cy, 0.30, 0.30, 0.30, 0.40, GREEN_PLANT),
        _box(cx-0.10, cy+0.05, 0.50, 0.20, 0.20, 0.25, GREEN_PLANT),
        _box(cx+0.08, cy-0.08, 0.45, 0.18, 0.18, 0.20, GREEN_PLANT),
    ])


# ── Room-based placement ──────────────────────────────────────────────────────

def _fits(cx, cy, w, d, zone) -> bool:
    """Check if a piece (cx,cy,w,d) fits inside the furniture zone."""
    fx0, fx1, fy0, fy1 = zone
    return (cx - w/2 >= fx0 and cx + w/2 <= fx1 and
            cy - d/2 >= fy0 and cy + d/2 <= fy1)


def _place_living_room(room: dict, seed: int) -> List[dict]:
    """Furniture for a living-room-sized space (>= 6 m²)."""
    rng = np.random.default_rng(seed)
    fx0, fx1, fy0, fy1 = room['furniture_zone']
    fw = fx1 - fx0
    fh = fy1 - fy0
    cx = (fx0 + fx1) / 2
    cy = (fy0 + fy1) / 2
    items = []

    # Sofa — against one wall, position varies by seed
    sofa_w = min(1.6, fw * 0.60)
    sofa_d = 0.80
    # Pick wall: 0=bottom, 1=top, 2=left, 3=right
    wall = int(rng.integers(0, 4))
    if wall == 0:   # bottom wall
        scx, scy = cx + rng.uniform(-fw*0.1, fw*0.1), fy0 + sofa_d/2
    elif wall == 1: # top wall
        scx, scy = cx + rng.uniform(-fw*0.1, fw*0.1), fy1 - sofa_d/2
    elif wall == 2: # left wall
        sofa_w, sofa_d = sofa_d, min(1.6, fh*0.60)
        scx, scy = fx0 + sofa_d/2, cy + rng.uniform(-fh*0.1, fh*0.1)
    else:           # right wall
        sofa_w, sofa_d = sofa_d, min(1.6, fh*0.60)
        scx, scy = fx1 - sofa_d/2, cy + rng.uniform(-fh*0.1, fh*0.1)

    if _fits(scx, scy, sofa_w, sofa_d, room['furniture_zone']):
        items.append(make_sofa(scx, scy, w=sofa_w, d=sofa_d))

        # Coffee table in front of sofa
        ct_w = min(0.80, sofa_w * 0.60)
        ct_d = 0.40
        if wall == 0:
            ctx, cty = scx, scy + sofa_d/2 + 0.30 + ct_d/2
        elif wall == 1:
            ctx, cty = scx, scy - sofa_d/2 - 0.30 - ct_d/2
        elif wall == 2:
            ctx, cty = scx + sofa_d/2 + 0.30 + ct_d/2, scy
            ct_w, ct_d = ct_d, ct_w
        else:
            ctx, cty = scx - sofa_d/2 - 0.30 - ct_d/2, scy
            ct_w, ct_d = ct_d, ct_w

        if _fits(ctx, cty, ct_w, ct_d, room['furniture_zone']):
            items.append(make_coffee_table(ctx, cty, w=ct_w, d=ct_d))

    # TV unit — opposite wall from sofa
    tv_w = min(1.1, fw * 0.45)
    tv_d = 0.32
    if wall == 0:
        tvx, tvy = cx, fy1 - tv_d/2
    elif wall == 1:
        tvx, tvy = cx, fy0 + tv_d/2
    elif wall == 2:
        tvx, tvy = fx1 - tv_d/2, cy
        tv_w, tv_d = tv_d, tv_w
    else:
        tvx, tvy = fx0 + tv_d/2, cy
        tv_w, tv_d = tv_d, tv_w

    if _fits(tvx, tvy, tv_w, tv_d, room['furniture_zone']):
        items.append(make_tv_unit(tvx, tvy, w=tv_w, d=tv_d))

    # Plant in a corner
    corners = [
        (fx0 + 0.20, fy0 + 0.20),
        (fx1 - 0.20, fy0 + 0.20),
        (fx0 + 0.20, fy1 - 0.20),
        (fx1 - 0.20, fy1 - 0.20),
    ]
    px, py = corners[int(rng.integers(0, 4))]
    if _fits(px, py, 0.35, 0.35, room['furniture_zone']):
        items.append(make_plant(px, py))

    return items


def _place_bedroom(room: dict, seed: int) -> List[dict]:
    """Furniture for a bedroom-sized space."""
    rng = np.random.default_rng(seed)
    fx0, fx1, fy0, fy1 = room['furniture_zone']
    fw = fx1 - fx0
    fh = fy1 - fy0
    items = []

    # Bed — against top wall, offset slightly
    bed_w = min(1.35, fw * 0.55)
    bed_d = min(1.85, fh * 0.50)
    offset = rng.uniform(-fw * 0.10, fw * 0.10)
    bcx = np.clip((fx0+fx1)/2 + offset, fx0 + bed_w/2, fx1 - bed_w/2)
    bcy = fy1 - bed_d/2

    if _fits(bcx, bcy, bed_w, bed_d, room['furniture_zone']):
        items.append(make_bed(bcx, bcy, w=bed_w, d=bed_d))

    # Wardrobe — left or right wall
    ward_w = min(0.90, fw * 0.30)
    ward_d = 0.45
    side = int(rng.integers(0, 2))
    if side == 0:
        wcx = fx0 + ward_w/2
    else:
        wcx = fx1 - ward_w/2
    wcy = (fy0 + fy1) / 2

    if _fits(wcx, wcy, ward_w, ward_d, room['furniture_zone']):
        items.append(make_wardrobe(wcx, wcy, w=ward_w, d=ward_d))

    # Desk — bottom wall
    desk_w = min(1.0, fw * 0.40)
    desk_d = 0.50
    dcx = (fx0 + fx1) / 2 + rng.uniform(-fw*0.10, fw*0.10)
    dcy = fy0 + desk_d/2

    if _fits(dcx, dcy, desk_w, desk_d, room['furniture_zone']):
        items.append(make_desk(dcx, dcy, w=desk_w, d=desk_d))

    return items


def _place_dining(room: dict, seed: int) -> List[dict]:
    """Furniture for a dining/kitchen-sized space."""
    rng = np.random.default_rng(seed)
    fx0, fx1, fy0, fy1 = room['furniture_zone']
    fw = fx1 - fx0
    fh = fy1 - fy0
    cx = (fx0 + fx1) / 2
    cy = (fy0 + fy1) / 2
    items = []

    # Dining table — centred with slight random offset
    dt_w = min(1.1, fw * 0.55)
    dt_d = min(0.70, fh * 0.45)
    dtcx = cx + rng.uniform(-fw*0.08, fw*0.08)
    dtcy = cy + rng.uniform(-fh*0.08, fh*0.08)

    if _fits(dtcx, dtcy, dt_w, dt_d, room['furniture_zone']):
        items.append(make_dining_table(dtcx, dtcy, w=dt_w, d=dt_d))

        # Chairs around table
        gap = 0.52
        chair_positions = [
            (dtcx - dt_w/2 - gap, dtcy),
            (dtcx + dt_w/2 + gap, dtcy),
            (dtcx, dtcy - dt_d/2 - gap),
            (dtcx, dtcy + dt_d/2 + gap),
        ]
        for chx, chy in chair_positions:
            if _fits(chx, chy, 0.42, 0.42, room['furniture_zone']):
                items.append(make_chair(chx, chy))

    # Bookshelf on a wall
    shelf_w = min(0.65, fw * 0.25)
    shelf_d = 0.22
    scx = fx0 + shelf_w/2
    scy = cy

    if _fits(scx, scy, shelf_w, shelf_d, room['furniture_zone']):
        items.append(make_bookshelf(scx, scy, w=shelf_w, d=shelf_d))

    return items


def _place_small_room(room: dict, seed: int) -> List[dict]:
    """Minimal furniture for small spaces."""
    rng = np.random.default_rng(seed)
    fx0, fx1, fy0, fy1 = room['furniture_zone']
    fw = fx1 - fx0
    fh = fy1 - fy0
    items = []

    # Just a small table and chair
    tw = min(0.70, fw * 0.50)
    td = min(0.50, fh * 0.40)
    tcx = (fx0 + fx1) / 2
    tcy = (fy0 + fy1) / 2

    if _fits(tcx, tcy, tw, td, room['furniture_zone']):
        items.append(make_coffee_table(tcx, tcy, w=tw, d=td))

    return items


# ── Main entry point ──────────────────────────────────────────────────────────

def place_furniture_in_rooms(
    vertices: np.ndarray,
    colors: np.ndarray,
    tris: np.ndarray,
    stem: str = "default",
) -> List[dict]:
    """
    Detect rooms from wall geometry and place furniture in each room.

    Args:
        vertices: (N,3) building vertices
        colors:   (N,3) per-vertex colors
        tris:     (T,3) triangle indices
        stem:     floorplan stem name — used as random seed for variation

    Returns:
        List of furniture mesh dicts (vertices, faces, colors)
    """
    from gan_part.inference.room_detector import detect_rooms

    # Detect open room spaces
    rooms = detect_rooms(vertices, colors, tris)
    print(f"  Detected {len(rooms)} room(s)")
    for i, r in enumerate(rooms):
        print(f"    Room {i}: {r['width']:.1f}×{r['height']:.1f}m "
              f"area={r['area']:.1f}m² "
              f"zone=({r['furniture_zone'][0]:.1f},{r['furniture_zone'][1]:.1f},"
              f"{r['furniture_zone'][2]:.1f},{r['furniture_zone'][3]:.1f})")

    if not rooms:
        print("  No rooms detected — skipping furniture")
        return []

    # Use stem as base seed for variation across floorplans
    base_seed = hash(stem) % (2**31)
    all_furniture = []

    for i, room in enumerate(rooms):
        room_seed = base_seed + i * 1000
        fw = room['furniture_zone'][1] - room['furniture_zone'][0]
        fh = room['furniture_zone'][3] - room['furniture_zone'][2]
        area = room['area']

        # Assign room type by size and index
        if area >= 8.0 and i == 0:
            # Largest room → living room
            items = _place_living_room(room, room_seed)
            rtype = "living"
        elif area >= 6.0 and fw >= 2.0 and fh >= 2.5:
            # Medium room → bedroom
            items = _place_bedroom(room, room_seed)
            rtype = "bedroom"
        elif area >= 4.0:
            # Smaller room → dining/study
            items = _place_dining(room, room_seed)
            rtype = "dining"
        else:
            # Small room → minimal
            items = _place_small_room(room, room_seed)
            rtype = "small"

        print(f"    Room {i} ({rtype}): {len(items)} furniture pieces")
        all_furniture.extend(items)

    return all_furniture


# ── Legacy entry point (used by gltf_texturer) ───────────────────────────────

def place_furniture(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> List[dict]:
    """
    Legacy fallback — places furniture using simple bounds.
    Used when vertex/color/tri data is not available.
    """
    WALL_MARGIN = 0.40
    ix0 = x_min + WALL_MARGIN
    ix1 = x_max - WALL_MARGIN
    iy0 = y_min + WALL_MARGIN
    iy1 = y_max - WALL_MARGIN
    iw = ix1 - ix0
    ih = iy1 - iy0

    if iw < 2.0 or ih < 2.0:
        return []

    cx = (ix0 + ix1) / 2
    cy = (iy0 + iy1) / 2
    zone = (ix0, ix1, iy0, iy1)
    room = {
        'x_min': ix0, 'x_max': ix1,
        'y_min': iy0, 'y_max': iy1,
        'cx': cx, 'cy': cy,
        'width': iw, 'height': ih,
        'area': iw * ih,
        'furniture_zone': zone,
    }
    return _place_living_room(room, seed=42)
