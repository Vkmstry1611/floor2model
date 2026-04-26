"""
interior_mesh.py
----------------
Converts a (render image, depth map) pair into a 3D mesh that can be
embedded inside the building .gltf.

Strategy
--------
1. Downsample render + depth to a manageable grid (default 128x128).
2. Each grid cell (x, y) + its depth value → a 3D vertex.
3. Adjacent cells → triangles.
4. Render image pixels → per-vertex RGB colours.
5. Scale the mesh to fit inside the building footprint (in metres).
6. Return vertices (N,3), faces (M,3), colours (N,3) — all numpy arrays.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def depth_to_interior_mesh(
    render_image: Image.Image,
    depth_map: np.ndarray,
    building_x_min: float,
    building_x_max: float,
    building_y_min: float,
    building_y_max: float,
    floor_z: float = 0.01,
    depth_scale: float = 1.5,
    grid_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 3D interior mesh from a render + depth map.

    Parameters
    ----------
    render_image      : PIL Image — the GAN-generated interior photo
    depth_map         : (H, W) float32 in [0,1], 1=close 0=far
    building_x_min/max: X extent of the building footprint in metres
    building_y_min/max: Y extent of the building footprint in metres
    floor_z           : base Z height of the mesh (just above floor)
    depth_scale       : how many metres the depth range spans (Z height)
    grid_size         : resolution of the mesh grid (grid_size x grid_size)

    Returns
    -------
    vertices : (N, 3) float32
    faces    : (M, 3) int32   — triangle indices
    colours  : (N, 3) float32 — RGB in [0, 1]
    """
    # ── Resize to grid ────────────────────────────────────────────────
    render_small = render_image.convert("RGB").resize(
        (grid_size, grid_size), Image.BILINEAR
    )
    depth_small = Image.fromarray(
        (depth_map * 255).astype(np.uint8)
    ).resize((grid_size, grid_size), Image.BILINEAR)

    colours_grid = np.array(render_small, dtype=np.float32) / 255.0  # (G, G, 3)
    depth_grid   = np.array(depth_small,  dtype=np.float32) / 255.0  # (G, G)

    G = grid_size

    # ── Build vertex positions ────────────────────────────────────────
    # X spans building width, Y spans building depth, Z = depth value
    xs = np.linspace(building_x_min, building_x_max, G)
    ys = np.linspace(building_y_min, building_y_max, G)
    xv, yv = np.meshgrid(xs, ys)          # both (G, G)

    # Z: floor_z + depth * depth_scale
    zv = floor_z + depth_grid * depth_scale  # (G, G)

    # Flatten to (N, 3)
    vertices = np.stack([
        xv.flatten(),
        yv.flatten(),
        zv.flatten(),
    ], axis=1).astype(np.float32)

    colours = colours_grid.reshape(-1, 3).astype(np.float32)

    # ── Build triangle faces ──────────────────────────────────────────
    # Each 2x2 quad → 2 triangles
    faces = []
    for row in range(G - 1):
        for col in range(G - 1):
            tl = row * G + col
            tr = tl + 1
            bl = tl + G
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])

    faces = np.array(faces, dtype=np.int32)

    return vertices, faces, colours


def mesh_to_gltf_primitive(
    vertices: np.ndarray,
    faces: np.ndarray,
    colours: np.ndarray,
) -> dict:
    """
    Package vertices/faces/colours into a dict ready for the glTF builder.
    """
    return {
        "vertices": vertices,
        "faces":    faces,
        "colours":  colours,
    }
