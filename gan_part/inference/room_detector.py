"""
room_detector.py
----------------
Detects open room spaces from the CV .gltf wall geometry.

Strategy:
1. Extract all wall segment footprints (XY bounding boxes at Z=0)
2. Build a 2D occupancy grid of the floor plan
3. Find connected open regions (rooms) using flood fill
4. Return room bounding boxes with clearance applied

This gives furniture placement zones that are guaranteed to be
inside rooms and away from walls.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def detect_rooms(
    vertices: np.ndarray,
    colors: np.ndarray,
    tris: np.ndarray,
    grid_resolution: float = 0.15,   # metres per grid cell
    min_room_area: float = 1.5,       # minimum room area in m²
    wall_clearance: float = 0.40,     # clearance from walls in metres
) -> List[dict]:
    """
    Detect open room spaces from wall geometry.

    Args:
        vertices:         (N, 3) float32 — building vertices in metres
        colors:           (N, 3) float32 — per-vertex RGB colors [0,1]
        tris:             (T, 3) int32   — triangle indices
        grid_resolution:  metres per grid cell
        min_room_area:    minimum room area to consider (m²)
        wall_clearance:   clearance from walls for furniture

    Returns:
        List of room dicts:
            {
              'x_min', 'x_max', 'y_min', 'y_max',  # room bounds (metres)
              'cx', 'cy',                            # room centre
              'width', 'height',                     # room dimensions
              'area',                                # room area m²
              'furniture_zone': (fx0,fx1,fy0,fy1)   # safe furniture zone
            }
    """
    # ── Classify triangles ────────────────────────────────────────────────────
    WALL_COLORS = [
        np.array([0.25, 0.25, 0.35]),
        np.array([0.40, 0.40, 0.50]),
        np.array([0.35, 0.35, 0.45]),
        np.array([0.30, 0.30, 0.40]),
    ]

    wall_tri_mask = np.zeros(len(tris), dtype=bool)
    for ti in range(len(tris)):
        c = colors[tris[ti]].mean(axis=0)
        if any(np.linalg.norm(c - wc) < 0.18 for wc in WALL_COLORS):
            wall_tri_mask[ti] = True

    # ── Build occupancy grid ──────────────────────────────────────────────────
    x_min = float(vertices[:, 0].min())
    x_max = float(vertices[:, 0].max())
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())

    # Add small padding
    pad = grid_resolution * 2
    gx0, gx1 = x_min - pad, x_max + pad
    gy0, gy1 = y_min - pad, y_max + pad

    nx = max(2, int((gx1 - gx0) / grid_resolution) + 1)
    ny = max(2, int((gy1 - gy0) / grid_resolution) + 1)

    # Grid: True = occupied by wall, False = open floor
    grid = np.zeros((ny, nx), dtype=bool)

    def world_to_grid(x, y):
        gx = int((x - gx0) / grid_resolution)
        gy = int((y - gy0) / grid_resolution)
        return np.clip(gx, 0, nx-1), np.clip(gy, 0, ny-1)

    # Mark wall triangles as occupied (with thickness padding)
    wall_tris = tris[wall_tri_mask]
    for tri in wall_tris:
        tv = vertices[tri]
        # Get XY bounding box of this triangle
        tx0, tx1 = tv[:, 0].min(), tv[:, 0].max()
        ty0, ty1 = tv[:, 1].min(), tv[:, 1].max()
        # Add wall clearance
        tx0 -= wall_clearance
        tx1 += wall_clearance
        ty0 -= wall_clearance
        ty1 += wall_clearance
        # Mark grid cells
        gx0_, gx1_ = world_to_grid(tx0, ty0)[0], world_to_grid(tx1, ty1)[0]
        gy0_, gy1_ = world_to_grid(tx0, ty0)[1], world_to_grid(tx1, ty1)[1]
        grid[gy0_:gy1_+1, gx0_:gx1_+1] = True

    # Also mark outside the building as occupied
    # Find building boundary and mark exterior
    bx0, bx1 = world_to_grid(x_min, y_min)[0], world_to_grid(x_max, y_max)[0]
    by0, by1 = world_to_grid(x_min, y_min)[1], world_to_grid(x_max, y_max)[1]
    # Mark border cells as occupied
    grid[:by0, :] = True
    grid[by1+1:, :] = True
    grid[:, :bx0] = True
    grid[:, bx1+1:] = True

    # ── Flood fill to find open rooms ─────────────────────────────────────────
    visited = np.zeros((ny, nx), dtype=bool)
    rooms = []

    def flood_fill(start_y, start_x):
        """BFS flood fill from a seed point. Returns list of (y,x) cells."""
        if grid[start_y, start_x] or visited[start_y, start_x]:
            return []
        cells = []
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        while queue:
            cy, cx = queue.pop(0)
            cells.append((cy, cx))
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny_, nx_ = cy+dy, cx+dx
                if 0 <= ny_ < ny and 0 <= nx_ < nx:
                    if not grid[ny_, nx_] and not visited[ny_, nx_]:
                        visited[ny_, nx_] = True
                        queue.append((ny_, nx_))
        return cells

    for gy in range(ny):
        for gx in range(nx):
            if not grid[gy, gx] and not visited[gy, gx]:
                cells = flood_fill(gy, gx)
                if not cells:
                    continue

                # Convert cells to world coordinates
                cell_arr = np.array(cells)
                ys = cell_arr[:, 0]
                xs = cell_arr[:, 1]

                # World bounds of this region
                rx0 = gx0 + xs.min() * grid_resolution
                rx1 = gx0 + (xs.max() + 1) * grid_resolution
                ry0 = gy0 + ys.min() * grid_resolution
                ry1 = gy0 + (ys.max() + 1) * grid_resolution

                rw = rx1 - rx0
                rh = ry1 - ry0
                area = rw * rh

                if area < min_room_area:
                    continue

                # Furniture zone — shrink further for safety
                extra = 0.10
                fz_x0 = rx0 + extra
                fz_x1 = rx1 - extra
                fz_y0 = ry0 + extra
                fz_y1 = ry1 - extra

                if fz_x1 - fz_x0 < 0.5 or fz_y1 - fz_y0 < 0.5:
                    continue

                rooms.append({
                    'x_min': rx0, 'x_max': rx1,
                    'y_min': ry0, 'y_max': ry1,
                    'cx': (rx0 + rx1) / 2,
                    'cy': (ry0 + ry1) / 2,
                    'width': rw,
                    'height': rh,
                    'area': area,
                    'furniture_zone': (fz_x0, fz_x1, fz_y0, fz_y1),
                })

    # Sort by area descending (largest room first)
    rooms.sort(key=lambda r: r['area'], reverse=True)
    return rooms
