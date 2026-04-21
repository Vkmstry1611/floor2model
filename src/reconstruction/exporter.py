"""
exporter.py
-----------
Exports Building3D models to OBJ, glTF 2.0, and binary STL formats.

Usage:
    from src.reconstruction.exporter import ModelExporter

    exporter = ModelExporter()
    paths = exporter.export(building, "outputs/models", "floorplan1")
    print(paths)  # {"obj": "...", "gltf": "...", "stl": "..."}
"""

from __future__ import annotations

import json
import struct
import base64
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .mesh_builder import Building3D


class ModelExporter:
    """
    Exports a Building3D to standard 3D file formats.

    Args:
        export_obj:  Write Wavefront OBJ + MTL.
        export_gltf: Write glTF 2.0 JSON (embedded buffers).
        export_stl:  Write binary STL.
    """

    def __init__(
        self,
        export_obj:  bool = True,
        export_gltf: bool = True,
        export_stl:  bool = False,
    ):
        self.export_obj  = export_obj
        self.export_gltf = export_gltf
        self.export_stl  = export_stl

    def export(
        self,
        building: Building3D,
        output_dir: str,
        stem: str,
    ) -> Dict[str, str]:
        """
        Export building to all configured formats.

        Args:
            building:   Building3D from MeshBuilder.
            output_dir: Directory to write files.
            stem:       Base filename (no extension).

        Returns:
            Dict mapping format → absolute file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        if self.export_obj:
            paths["obj"] = self._write_obj(building, out, stem)

        if self.export_gltf:
            paths["gltf"] = self._write_gltf(building, out, stem)

        if self.export_stl:
            paths["stl"] = self._write_stl(building, out, stem)

        return paths

    # ── OBJ ──────────────────────────────────────────────────────────────────

    def _write_obj(self, building: Building3D, out: Path, stem: str) -> str:
        """Write Wavefront OBJ + MTL files."""
        obj_path = out / f"{stem}.obj"
        mtl_path = out / f"{stem}.mtl"

        # Collect unique materials from meshes
        materials: Dict[str, tuple] = {}
        for mesh in building.meshes:
            mat_name = mesh.label.replace(" ", "_")
            materials[mat_name] = mesh.color

        # Write MTL
        with open(mtl_path, "w") as f:
            f.write("# floor2model material library\n\n")
            for mat_name, color in materials.items():
                r, g, b = color
                f.write(f"newmtl {mat_name}\n")
                f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
                f.write(f"Ka 0.1 0.1 0.1\n")
                f.write(f"Ks 0.0 0.0 0.0\n")
                f.write(f"d 1.0\n\n")

        # Write OBJ
        with open(obj_path, "w") as f:
            f.write(f"# floor2model OBJ export\n")
            f.write(f"mtllib {stem}.mtl\n\n")

            v_offset = 1  # OBJ is 1-indexed
            for mesh in building.meshes:
                if mesh.vertex_count == 0:
                    continue

                mat_name = mesh.label.replace(" ", "_")
                f.write(f"o {mesh.label}\n")
                f.write(f"usemtl {mat_name}\n")

                # Vertices
                for v in mesh.vertices:
                    f.write(f"v {v[0]:.6f} {v[2]:.6f} {-v[1]:.6f}\n")  # Y-up

                # Faces (triangulate quads)
                tris = self._triangulate(mesh.faces)
                for tri in tris:
                    indices = " ".join(str(i + v_offset) for i in tri)
                    f.write(f"f {indices}\n")

                f.write("\n")
                v_offset += mesh.vertex_count

        print(f"OBJ exported: {obj_path}")
        return str(obj_path)

    # ── glTF 2.0 ─────────────────────────────────────────────────────────────

    def _write_gltf(self, building: Building3D, out: Path, stem: str) -> str:
        """Write glTF 2.0 JSON with embedded base64 buffers."""
        gltf_path = out / f"{stem}.gltf"

        vertices = building.merged_vertices  # (N, 3) float32
        colors   = building.merged_colors    # (N, 3) float32

        # Triangulate all faces
        tris = self._triangulate(building.merged_faces)
        if not tris:
            # Write minimal valid glTF
            gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": []}
            with open(gltf_path, "w") as f:
                json.dump(gltf, f, indent=2)
            return str(gltf_path)

        indices = np.array(tris, dtype=np.uint32).flatten()

        # Encode buffers
        vert_bytes  = vertices.astype(np.float32).tobytes()
        color_bytes = colors.astype(np.float32).tobytes()
        idx_bytes   = indices.astype(np.uint32).tobytes()

        # Pad to 4-byte alignment
        def pad4(b: bytes) -> bytes:
            r = len(b) % 4
            return b + b"\x00" * (4 - r) if r else b

        vert_bytes  = pad4(vert_bytes)
        color_bytes = pad4(color_bytes)
        idx_bytes   = pad4(idx_bytes)

        buffer_data = vert_bytes + color_bytes + idx_bytes
        buffer_b64  = base64.b64encode(buffer_data).decode("ascii")

        n_verts = len(vertices)
        n_idx   = len(indices)

        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()

        vert_offset  = 0
        color_offset = len(vert_bytes)
        idx_offset   = color_offset + len(color_bytes)

        gltf = {
            "asset": {"version": "2.0", "generator": "floor2model"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": stem}],
            "meshes": [{
                "name": stem,
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "COLOR_0":  1,
                    },
                    "indices": 2,
                    "mode": 4,  # TRIANGLES
                }]
            }],
            "accessors": [
                {   # 0 — positions
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": n_verts,
                    "type": "VEC3",
                    "min": v_min,
                    "max": v_max,
                },
                {   # 1 — colors
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": n_verts,
                    "type": "VEC3",
                },
                {   # 2 — indices
                    "bufferView": 2,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": n_idx,
                    "type": "SCALAR",
                },
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": vert_offset,  "byteLength": len(vert_bytes)},
                {"buffer": 0, "byteOffset": color_offset, "byteLength": len(color_bytes)},
                {"buffer": 0, "byteOffset": idx_offset,   "byteLength": len(idx_bytes)},
            ],
            "buffers": [{
                "byteLength": len(buffer_data),
                "uri": f"data:application/octet-stream;base64,{buffer_b64}",
            }],
        }

        with open(gltf_path, "w") as f:
            json.dump(gltf, f, indent=2)

        print(f"glTF exported: {gltf_path}")
        return str(gltf_path)

    # ── Binary STL ────────────────────────────────────────────────────────────

    def _write_stl(self, building: Building3D, out: Path, stem: str) -> str:
        """Write binary STL file."""
        stl_path = out / f"{stem}.stl"

        vertices = building.merged_vertices
        tris = self._triangulate(building.merged_faces)

        with open(stl_path, "wb") as f:
            # 80-byte header
            header = f"floor2model STL {stem}".encode("ascii")
            f.write(header.ljust(80, b"\x00"))

            # Triangle count
            f.write(struct.pack("<I", len(tris)))

            for tri in tris:
                if len(tri) < 3:
                    continue
                v0 = vertices[tri[0]]
                v1 = vertices[tri[1]]
                v2 = vertices[tri[2]]

                # Compute face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                nm = np.linalg.norm(normal)
                if nm > 1e-10:
                    normal /= nm

                # Normal (3 floats) + 3 vertices (3×3 floats) + attribute (uint16)
                f.write(struct.pack("<fff", *normal))
                f.write(struct.pack("<fff", *v0))
                f.write(struct.pack("<fff", *v1))
                f.write(struct.pack("<fff", *v2))
                f.write(struct.pack("<H", 0))  # attribute byte count

        print(f"STL exported: {stl_path}")
        return str(stl_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _triangulate(self, faces: List[List[int]]) -> List[List[int]]:
        """Fan-triangulate polygons (quads → 2 tris, n-gons → n-2 tris)."""
        tris = []
        for face in faces:
            if len(face) < 3:
                continue
            if len(face) == 3:
                tris.append(face)
            else:
                # Fan triangulation from first vertex
                for i in range(1, len(face) - 1):
                    tris.append([face[0], face[i], face[i + 1]])
        return tris
