"""
gltf_texturer.py
----------------
Reads a CV-generated .gltf, identifies face groups by their
per-vertex colors, generates textures for each group via GAN,
and writes a new _interior.gltf with realistic surface textures.

Face classification by per-vertex COLOR_0:
  Wall    → grey   (~0.35, 0.35, 0.45)
  Door    → brown  (~0.65, 0.45, 0.25)
  Window  → blue   (~0.60, 0.80, 0.95)
  Floor   → tan    (~0.60, 0.55, 0.50)
  Other   → wall fallback
"""

from __future__ import annotations

import json
import base64
import io
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ── Color → surface type mapping ─────────────────────────────────────────────
# These match Extruder.CLASS_COLORS

SURFACE_COLORS = {
    "wall":    np.array([0.35, 0.35, 0.45]),
    "door":    np.array([0.65, 0.45, 0.25]),
    "window":  np.array([0.60, 0.80, 0.95]),
    "floor":   np.array([0.60, 0.55, 0.50]),
    "ceiling": np.array([0.85, 0.85, 0.80]),
}

# Also match OuterWall / InnerWall colors
EXTRA_WALL_COLORS = [
    np.array([0.25, 0.25, 0.35]),  # OuterWall
    np.array([0.40, 0.40, 0.50]),  # InnerWall
    np.array([0.35, 0.35, 0.45]),  # wall
]


def classify_color(rgb: np.ndarray) -> str:
    """Return surface type for a given RGB color (values 0-1)."""
    # Check wall colors first (multiple shades)
    for wc in EXTRA_WALL_COLORS:
        if np.linalg.norm(rgb - wc) < 0.15:
            return "wall"

    best, best_dist = "wall", float("inf")
    for name, ref in SURFACE_COLORS.items():
        d = np.linalg.norm(rgb - ref)
        if d < best_dist:
            best_dist, best = d, name
    return best


def _pad4(b: bytes) -> bytes:
    r = len(b) % 4
    return b + b"\x00" * (4 - r) if r else b


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def apply_textures(
    cv_gltf_path: str,
    texture_generator,
    output_path: str,
) -> str:
    """
    Read CV .gltf, apply GAN-generated textures to wall/floor/door/window
    faces, add furniture, write new _interior.gltf.
    Removes the floorplan image planes from the CV output.
    """
    from gan_part.inference.furniture_placer import place_furniture_in_rooms

    print(f"  Reading CV model: {Path(cv_gltf_path).name}")
    gltf = json.loads(Path(cv_gltf_path).read_text())

    # ── Decode buffer ─────────────────────────────────────────────────────────
    buf_uri = gltf["buffers"][0]["uri"]
    raw = base64.b64decode(buf_uri.split(",", 1)[1])

    def read_accessor(acc_idx: int) -> np.ndarray:
        acc = gltf["accessors"][acc_idx]
        bv  = gltf["bufferViews"][acc["bufferView"]]
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        chunk  = raw[offset: offset + length]

        comp_type = acc["componentType"]
        acc_type  = acc["type"]
        count     = acc["count"]

        dtype_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                     5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = dtype_map[comp_type]

        dim_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                   "MAT2": 4, "MAT3": 9, "MAT4": 16}
        dim = dim_map[acc_type]

        arr = np.frombuffer(chunk, dtype=dtype)
        if dim > 1:
            arr = arr.reshape(count, dim)
        return arr

    # ── Find the building mesh — skip InteriorFloor / FloorPlan ──────────────
    building_mesh = None
    building_prim = None
    for mesh in gltf.get("meshes", []):
        if mesh["name"] not in ("InteriorFloor", "FloorPlan"):
            building_mesh = mesh
            building_prim = mesh["primitives"][0]
            break

    if building_mesh is None:
        raise ValueError("No building mesh found in CV gltf")

    print(f"  Building mesh: '{building_mesh['name']}'")

    # ── Read vertices, colors, indices ────────────────────────────────────────
    pos_acc = building_prim["attributes"]["POSITION"]
    col_acc = building_prim["attributes"]["COLOR_0"]
    idx_acc = building_prim["indices"]

    vertices = read_accessor(pos_acc).astype(np.float32)
    colors   = read_accessor(col_acc).astype(np.float32)
    indices  = read_accessor(idx_acc).astype(np.int32)

    print(f"  Vertices: {len(vertices)}  Indices: {len(indices)}")

    tris = indices.reshape(-1, 3)
    tri_colors = colors[tris].mean(axis=1)

    surface_tris: dict[str, list] = {s: [] for s in SURFACE_COLORS}
    for ti, tc in enumerate(tri_colors):
        surf = classify_color(tc)
        surface_tris[surf].append(ti)

    for surf, tlist in surface_tris.items():
        print(f"  {surf}: {len(tlist)} triangles")

    # ── Generate textures ─────────────────────────────────────────────────────
    textures_needed = [s for s, tlist in surface_tris.items() if tlist]
    # Always generate floor texture
    if "floor" not in textures_needed:
        textures_needed.append("floor")
    print(f"\n  Generating {len(textures_needed)} textures...")

    tex_images: dict[str, Image.Image] = {}
    for surf in textures_needed:
        print(f"    Generating {surf} texture...")
        tex_images[surf] = texture_generator.generate(surf, size=512)

    # ── Build new gltf ────────────────────────────────────────────────────────
    print("\n  Building textured gltf...")

    new_buffers_data = b""
    new_buffer_views = []
    new_accessors    = []
    new_meshes       = []
    new_nodes        = []
    new_materials    = []
    new_textures     = []
    new_images       = []

    def add_buf(data: bytes, target: Optional[int] = None) -> int:
        nonlocal new_buffers_data
        padded = _pad4(data)
        bv = {"buffer": 0,
              "byteOffset": len(new_buffers_data),
              "byteLength": len(padded)}
        if target:
            bv["target"] = target
        new_buffer_views.append(bv)
        new_buffers_data += padded
        return len(new_buffer_views) - 1

    def add_acc(bv_idx, comp_type, count, acc_type,
                min_v=None, max_v=None) -> int:
        acc = {"bufferView": bv_idx, "componentType": comp_type,
               "count": count, "type": acc_type}
        if min_v is not None: acc["min"] = min_v
        if max_v is not None: acc["max"] = max_v
        new_accessors.append(acc)
        return len(new_accessors) - 1

    # ── Building footprint ────────────────────────────────────────────────────
    x_min = float(vertices[:, 0].min())
    x_max = float(vertices[:, 0].max())
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())

    # ── Floor plane with proper top-down wood texture ─────────────────────────
    # NOTE: No floorplan image — just the GAN-generated floor texture
    floor_verts = np.array([
        [x_min, y_min, 0.005],
        [x_max, y_min, 0.005],
        [x_max, y_max, 0.005],
        [x_min, y_max, 0.005],
    ], dtype=np.float32)
    # Tile the texture: repeat 4x across the floor for realistic scale
    floor_uvs = np.array([[0,0],[4,0],[4,4],[0,4]], dtype=np.float32)
    floor_idx = np.array([0,1,2, 0,2,3], dtype=np.uint32)

    floor_b64 = _img_to_b64(tex_images["floor"])
    new_images.append({"uri": f"data:image/png;base64,{floor_b64}"})
    new_textures.append({"source": len(new_images) - 1})
    floor_mat_idx = len(new_materials)
    new_materials.append({
        "name": "FloorTexture",
        "pbrMetallicRoughness": {
            "baseColorTexture": {
                "index": len(new_textures) - 1,
                "extensions": {
                    "KHR_texture_transform": {
                        "scale": [4.0, 4.0]
                    }
                }
            },
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
        },
        "doubleSided": True,
    })

    bv_fv  = add_buf(floor_verts.tobytes(), 34962)
    bv_fuv = add_buf(floor_uvs.tobytes(), 34962)
    bv_fi  = add_buf(floor_idx.tobytes(), 34963)
    acc_fv  = add_acc(bv_fv,  5126, 4, "VEC3",
                      floor_verts.min(0).tolist(), floor_verts.max(0).tolist())
    acc_fuv = add_acc(bv_fuv, 5126, 4, "VEC2")
    acc_fi  = add_acc(bv_fi,  5125, 6, "SCALAR")

    new_meshes.append({"name": "Floor", "primitives": [{
        "attributes": {"POSITION": acc_fv, "TEXCOORD_0": acc_fuv},
        "indices": acc_fi, "mode": 4,
        "material": floor_mat_idx,
    }]})
    new_nodes.append({"mesh": len(new_meshes) - 1, "name": "Floor"})

    # ── Textured wall/door/window primitives ──────────────────────────────────
    for surf in ["wall", "door", "window", "ceiling"]:
        tri_list = surface_tris.get(surf, [])
        if not tri_list or surf not in tex_images:
            continue

        surf_tris = tris[tri_list]
        unique_verts, inv = np.unique(surf_tris.flatten(), return_inverse=True)
        sub_verts = vertices[unique_verts]
        sub_tris  = inv.reshape(-1, 3).astype(np.uint32)
        sub_uvs   = _planar_uv(sub_verts, surf)

        tex_b64 = _img_to_b64(tex_images[surf])
        new_images.append({"uri": f"data:image/png;base64,{tex_b64}"})
        new_textures.append({"source": len(new_images) - 1})
        mat_idx = len(new_materials)

        roughness = 0.85 if surf == "wall"   else \
                    0.70 if surf == "door"    else \
                    0.20 if surf == "window"  else 0.90

        new_materials.append({
            "name": f"{surf.capitalize()}Texture",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": len(new_textures) - 1},
                "metallicFactor": 0.05 if surf == "window" else 0.0,
                "roughnessFactor": roughness,
            },
            "doubleSided": True,
        })

        bv_v  = add_buf(sub_verts.astype(np.float32).tobytes(), 34962)
        bv_uv = add_buf(sub_uvs.astype(np.float32).tobytes(), 34962)
        bv_i  = add_buf(sub_tris.flatten().tobytes(), 34963)

        acc_v  = add_acc(bv_v,  5126, len(sub_verts), "VEC3",
                         sub_verts.min(0).tolist(), sub_verts.max(0).tolist())
        acc_uv = add_acc(bv_uv, 5126, len(sub_verts), "VEC2")
        acc_i  = add_acc(bv_i,  5125, len(sub_tris) * 3, "SCALAR")

        new_meshes.append({
            "name": f"{surf.capitalize()}Mesh",
            "primitives": [{
                "attributes": {"POSITION": acc_v, "TEXCOORD_0": acc_uv},
                "indices": acc_i, "mode": 4,
                "material": mat_idx,
            }]
        })
        new_nodes.append({"mesh": len(new_meshes) - 1,
                          "name": f"{surf.capitalize()}Mesh"})
        print(f"  Added {surf} mesh: {len(sub_verts)} verts, {len(sub_tris)} tris")

    # ── Furniture ─────────────────────────────────────────────────────────────
    print("\n  Placing furniture...")
    from gan_part.inference.furniture_placer import place_furniture_in_rooms
    stem = Path(output_path).stem.replace("_interior", "")
    furniture_items = place_furniture_in_rooms(vertices, colors, tris, stem=stem)
    print(f"  {len(furniture_items)} furniture pieces")

    for fi, item in enumerate(furniture_items):
        fv = item["vertices"].astype(np.float32)
        ff = item["faces"].astype(np.uint32).flatten()
        fc = item["colors"].astype(np.float32)

        bv_fv2 = add_buf(fv.tobytes(), 34962)
        bv_fc  = add_buf(fc.tobytes(), 34962)
        bv_fi2 = add_buf(ff.tobytes(), 34963)

        acc_fv2 = add_acc(bv_fv2, 5126, len(fv), "VEC3",
                          fv.min(0).tolist(), fv.max(0).tolist())
        acc_fc  = add_acc(bv_fc,  5126, len(fv), "VEC3")
        acc_fi2 = add_acc(bv_fi2, 5125, len(ff), "SCALAR")

        new_materials.append({
            "name": f"Furniture_{fi}",
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 0.85,
            },
        })

        new_meshes.append({
            "name": f"Furniture_{fi}",
            "primitives": [{
                "attributes": {"POSITION": acc_fv2, "COLOR_0": acc_fc},
                "indices": acc_fi2, "mode": 4,
                "material": len(new_materials) - 1,
            }]
        })
        new_nodes.append({"mesh": len(new_meshes) - 1,
                          "name": f"Furniture_{fi}"})

    # ── Assemble and write ────────────────────────────────────────────────────
    out_gltf = {
        "asset": {"version": "2.0", "generator": "Floor2Model-GAN-Interior"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(new_nodes)))}],
        "nodes": new_nodes,
        "meshes": new_meshes,
        "accessors": new_accessors,
        "bufferViews": new_buffer_views,
        "buffers": [{
            "byteLength": len(new_buffers_data),
            "uri": "data:application/octet-stream;base64,"
                   + base64.b64encode(new_buffers_data).decode("ascii"),
        }],
    }
    if new_materials: out_gltf["materials"] = new_materials
    if new_textures:  out_gltf["textures"]  = new_textures
    if new_images:    out_gltf["images"]    = new_images

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out_gltf, indent=2))
    print(f"\n  ✓ Interior gltf written: {Path(output_path).name}")
    return output_path


def _planar_uv(verts: np.ndarray, surf: str) -> np.ndarray:
    """
    Generate UV coordinates using planar projection.
    - Floor/ceiling → project onto XY plane
    - Walls → project onto XZ or YZ plane depending on orientation
    """
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    x_range = x.max() - x.min() + 1e-6
    y_range = y.max() - y.min() + 1e-6
    z_range = z.max() - z.min() + 1e-6

    if surf in ("floor", "ceiling"):
        u = (x - x.min()) / x_range
        v = (y - y.min()) / y_range
    else:
        if x_range >= y_range:
            u = (x - x.min()) / x_range
            v = (z - z.min()) / (z_range + 1e-6)
        else:
            u = (y - y.min()) / y_range
            v = (z - z.min()) / (z_range + 1e-6)

    return np.stack([u, v], axis=1).astype(np.float32)

    def read_accessor(acc_idx: int) -> np.ndarray:
        acc = gltf["accessors"][acc_idx]
        bv  = gltf["bufferViews"][acc["bufferView"]]
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        chunk  = raw[offset: offset + length]

        comp_type = acc["componentType"]
        acc_type  = acc["type"]
        count     = acc["count"]

        dtype_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                     5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = dtype_map[comp_type]

        dim_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                   "MAT2": 4, "MAT3": 9, "MAT4": 16}
        dim = dim_map[acc_type]

        arr = np.frombuffer(chunk, dtype=dtype)
        if dim > 1:
            arr = arr.reshape(count, dim)
        return arr

    # ── Find the building mesh (not InteriorFloor / FloorPlan) ────────────────
    building_mesh = None
    building_prim = None
    for mesh in gltf.get("meshes", []):
        if mesh["name"] not in ("InteriorFloor", "FloorPlan"):
            building_mesh = mesh
            building_prim = mesh["primitives"][0]
            break

    if building_mesh is None:
        raise ValueError("No building mesh found in CV gltf")

    print(f"  Building mesh: '{building_mesh['name']}'")

    # ── Read vertices, colors, indices ────────────────────────────────────────
    pos_acc   = building_prim["attributes"]["POSITION"]
    col_acc   = building_prim["attributes"]["COLOR_0"]
    idx_acc   = building_prim["indices"]

    vertices = read_accessor(pos_acc).astype(np.float32)   # (N, 3)
    colors   = read_accessor(col_acc).astype(np.float32)   # (N, 3)
    indices  = read_accessor(idx_acc).astype(np.int32)     # (M,) flat

    print(f"  Vertices: {len(vertices)}  Indices: {len(indices)}")

    # Triangulate — indices are already triangles (mode=4)
    tris = indices.reshape(-1, 3)   # (T, 3)

    # ── Classify each triangle by dominant vertex color ───────────────────────
    # Average the 3 vertex colors of each triangle
    tri_colors = colors[tris].mean(axis=1)   # (T, 3)

    surface_tris: dict[str, list] = {s: [] for s in SURFACE_COLORS}
    for ti, tc in enumerate(tri_colors):
        surf = classify_color(tc)
        surface_tris[surf].append(ti)

    for surf, tlist in surface_tris.items():
        print(f"  {surf}: {len(tlist)} triangles")

    # ── Generate textures ─────────────────────────────────────────────────────
    textures_needed = [s for s, tlist in surface_tris.items() if tlist]
    print(f"\n  Generating {len(textures_needed)} textures...")

    tex_images: dict[str, Image.Image] = {}
    for surf in textures_needed:
        print(f"    Generating {surf} texture...")
        tex_images[surf] = texture_generator.generate(surf, size=512)

    # ── Build new gltf ────────────────────────────────────────────────────────
    print("\n  Building textured gltf...")

    new_buffers_data = b""
    new_buffer_views = []
    new_accessors    = []
    new_meshes       = []
    new_nodes        = []
    new_materials    = []
    new_textures     = []
    new_images       = []

    def add_buf(data: bytes, target: Optional[int] = None) -> int:
        nonlocal new_buffers_data
        padded = _pad4(data)
        bv = {"buffer": 0,
              "byteOffset": len(new_buffers_data),
              "byteLength": len(padded)}
        if target:
            bv["target"] = target
        new_buffer_views.append(bv)
        new_buffers_data += padded
        return len(new_buffer_views) - 1

    def add_acc(bv_idx, comp_type, count, acc_type,
                min_v=None, max_v=None) -> int:
        acc = {"bufferView": bv_idx, "componentType": comp_type,
               "count": count, "type": acc_type}
        if min_v is not None: acc["min"] = min_v
        if max_v is not None: acc["max"] = max_v
        new_accessors.append(acc)
        return len(new_accessors) - 1

    # ── Keep original floor/floorplan planes from CV gltf ────────────────────
    # Re-encode them into the new buffer
    for mesh in gltf.get("meshes", []):
        if mesh["name"] in ("InteriorFloor", "FloorPlan"):
            for prim in mesh["primitives"]:
                # Read original data
                p_acc = prim["attributes"]["POSITION"]
                uv_acc = prim["attributes"]["TEXCOORD_0"]
                i_acc  = prim["indices"]

                p_data  = read_accessor(p_acc).astype(np.float32)
                uv_data = read_accessor(uv_acc).astype(np.float32)
                i_data  = read_accessor(i_acc).astype(np.uint32)

                bv_p  = add_buf(p_data.tobytes(), 34962)
                bv_uv = add_buf(uv_data.tobytes(), 34962)
                bv_i  = add_buf(i_data.tobytes(), 34963)

                acc_p  = add_acc(bv_p,  5126, len(p_data),  "VEC3",
                                 p_data.min(0).tolist(), p_data.max(0).tolist())
                acc_uv = add_acc(bv_uv, 5126, len(uv_data), "VEC2")
                acc_i  = add_acc(bv_i,  5125, len(i_data),  "SCALAR")

                # Copy material (texture) from original
                orig_mat_idx = prim.get("material")
                new_mat_idx  = None
                if orig_mat_idx is not None:
                    orig_mat  = gltf["materials"][orig_mat_idx]
                    orig_tex  = orig_mat["pbrMetallicRoughness"].get(
                        "baseColorTexture", {}).get("index")
                    if orig_tex is not None:
                        orig_img = gltf["images"][
                            gltf["textures"][orig_tex]["source"]
                        ]
                        new_images.append(orig_img)
                        new_textures.append({"source": len(new_images) - 1})
                        new_mat_idx = len(new_materials)
                        new_materials.append({
                            "name": orig_mat["name"],
                            "pbrMetallicRoughness": {
                                "baseColorTexture": {
                                    "index": len(new_textures) - 1
                                },
                                "metallicFactor": 0.0,
                                "roughnessFactor": 1.0,
                            },
                            "doubleSided": True,
                        })

                new_prim = {
                    "attributes": {"POSITION": acc_p, "TEXCOORD_0": acc_uv},
                    "indices": acc_i,
                    "mode": 4,
                }
                if new_mat_idx is not None:
                    new_prim["material"] = new_mat_idx

                new_meshes.append({"name": mesh["name"],
                                   "primitives": [new_prim]})
                new_nodes.append({"mesh": len(new_meshes) - 1,
                                  "name": mesh["name"]})

    # ── Add floor plane with floor texture ────────────────────────────────────
    # Create a proper floor quad at Z=0 covering the building footprint
    x_min = float(vertices[:, 0].min())
    x_max = float(vertices[:, 0].max())
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())

    floor_verts = np.array([
        [x_min, y_min, 0.01],
        [x_max, y_min, 0.01],
        [x_max, y_max, 0.01],
        [x_min, y_max, 0.01],
    ], dtype=np.float32)
    floor_uvs  = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    floor_idx  = np.array([0,1,2, 0,2,3], dtype=np.uint32)

    if "floor" in tex_images:
        floor_b64 = _img_to_b64(tex_images["floor"])
        new_images.append({"uri": f"data:image/png;base64,{floor_b64}"})
        new_textures.append({"source": len(new_images) - 1})
        floor_mat_idx = len(new_materials)
        new_materials.append({
            "name": "FloorTexture",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": len(new_textures) - 1},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "doubleSided": True,
        })

        bv_fv  = add_buf(floor_verts.tobytes(), 34962)
        bv_fuv = add_buf(floor_uvs.tobytes(), 34962)
        bv_fi  = add_buf(floor_idx.tobytes(), 34963)
        acc_fv  = add_acc(bv_fv,  5126, 4, "VEC3",
                          floor_verts.min(0).tolist(),
                          floor_verts.max(0).tolist())
        acc_fuv = add_acc(bv_fuv, 5126, 4, "VEC2")
        acc_fi  = add_acc(bv_fi,  5125, 6, "SCALAR")

        new_meshes.append({"name": "Floor", "primitives": [{
            "attributes": {"POSITION": acc_fv, "TEXCOORD_0": acc_fuv},
            "indices": acc_fi, "mode": 4,
            "material": floor_mat_idx,
        }]})
        new_nodes.append({"mesh": len(new_meshes) - 1, "name": "Floor"})

    # ── Add textured wall/door/window primitives ──────────────────────────────
    # For each surface type, collect its triangles, build a sub-mesh
    # with UV coordinates and the generated texture.

    for surf in ["wall", "door", "window", "ceiling"]:
        tri_list = surface_tris.get(surf, [])
        if not tri_list or surf not in tex_images:
            continue

        # Collect unique vertices for this surface
        surf_tris = tris[tri_list]                    # (T, 3)
        unique_verts, inv = np.unique(
            surf_tris.flatten(), return_inverse=True
        )
        sub_verts = vertices[unique_verts]            # (V, 3)
        sub_tris  = inv.reshape(-1, 3).astype(np.uint32)

        # Generate UV coordinates using planar projection
        # Project onto the dominant plane of each face
        sub_uvs = _planar_uv(sub_verts, surf)

        # Encode texture
        tex_b64 = _img_to_b64(tex_images[surf])
        new_images.append({"uri": f"data:image/png;base64,{tex_b64}"})
        new_textures.append({"source": len(new_images) - 1})
        mat_idx = len(new_materials)

        roughness = 0.85 if surf == "wall" else \
                    0.70 if surf == "floor" else \
                    0.60 if surf == "door"  else \
                    0.20   # window — slightly shiny

        new_materials.append({
            "name": f"{surf.capitalize()}Texture",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": len(new_textures) - 1},
                "metallicFactor": 0.0,
                "roughnessFactor": roughness,
            },
            "doubleSided": True,
        })

        bv_v  = add_buf(sub_verts.astype(np.float32).tobytes(), 34962)
        bv_uv = add_buf(sub_uvs.astype(np.float32).tobytes(), 34962)
        bv_i  = add_buf(sub_tris.flatten().tobytes(), 34963)

        acc_v  = add_acc(bv_v,  5126, len(sub_verts), "VEC3",
                         sub_verts.min(0).tolist(),
                         sub_verts.max(0).tolist())
        acc_uv = add_acc(bv_uv, 5126, len(sub_verts), "VEC2")
        acc_i  = add_acc(bv_i,  5125, len(sub_tris) * 3, "SCALAR")

        new_meshes.append({
            "name": f"{surf.capitalize()}Mesh",
            "primitives": [{
                "attributes": {"POSITION": acc_v, "TEXCOORD_0": acc_uv},
                "indices": acc_i,
                "mode": 4,
                "material": mat_idx,
            }]
        })
        new_nodes.append({
            "mesh": len(new_meshes) - 1,
            "name": f"{surf.capitalize()}Mesh",
        })
        print(f"  Added {surf} mesh: {len(sub_verts)} verts, "
              f"{len(sub_tris)} tris")

    # ── Assemble and write ────────────────────────────────────────────────────
    out_gltf = {
        "asset": {"version": "2.0", "generator": "Floor2Model-GAN-Interior"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(new_nodes)))}],
        "nodes": new_nodes,
        "meshes": new_meshes,
        "accessors": new_accessors,
        "bufferViews": new_buffer_views,
        "buffers": [{
            "byteLength": len(new_buffers_data),
            "uri": "data:application/octet-stream;base64,"
                   + base64.b64encode(new_buffers_data).decode("ascii"),
        }],
    }
    if new_materials: out_gltf["materials"] = new_materials
    if new_textures:  out_gltf["textures"]  = new_textures
    if new_images:    out_gltf["images"]    = new_images

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out_gltf, indent=2))
    print(f"\n  ✓ Interior gltf written: {Path(output_path).name}")
    return output_path


def _planar_uv(verts: np.ndarray, surf: str) -> np.ndarray:
    """
    Generate UV coordinates using planar projection.

    - Floor/ceiling → project onto XY plane
    - Walls → project onto XZ or YZ plane depending on orientation
    - Doors/windows → same as walls
    """
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    x_range = x.max() - x.min() + 1e-6
    y_range = y.max() - y.min() + 1e-6
    z_range = z.max() - z.min() + 1e-6

    if surf in ("floor", "ceiling"):
        # Top-down: U=X, V=Y
        u = (x - x.min()) / x_range
        v = (y - y.min()) / y_range
    else:
        # Walls/doors/windows: project onto the plane with most variation
        if x_range >= y_range:
            # Horizontal wall — U=X, V=Z
            u = (x - x.min()) / x_range
            v = (z - z.min()) / (z_range + 1e-6)
        else:
            # Vertical wall — U=Y, V=Z
            u = (y - y.min()) / y_range
            v = (z - z.min()) / (z_range + 1e-6)

    return np.stack([u, v], axis=1).astype(np.float32)
