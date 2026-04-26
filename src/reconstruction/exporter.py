"""
exporter.py
-----------
Exports Building3D models to OBJ + MTL and glTF 2.0.
The original floor plan image is embedded as a textured ground plane at z=0,
with all 3D geometry (walls, doors, windows) extruded upward from it.
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
        floorplan_image_path: Optional[str] = None,
        render_image_path: Optional[str] = None,
    ) -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        if self.export_obj:
            paths["obj"] = self._write_obj(building, out, stem,
                                           floorplan_image_path, render_image_path)

        if self.export_gltf:
            paths["gltf"] = self._write_gltf(building, out, stem,
                                              floorplan_image_path, render_image_path)

        if self.export_stl:
            paths["stl"] = self._write_stl(building, out, stem)

        return paths

    # ── Ground plane helpers ──────────────────────────────────────────────────

    def _ground_plane_vertices(self, building: Building3D):
        """
        Compute a ground plane quad that fits the building footprint.
        Returns 4 corners at z=-0.001 (just below floor level).
        """
        verts = building.merged_vertices
        if len(verts) == 0:
            return None
        x_min, y_min = verts[:, 0].min(), verts[:, 1].min()
        x_max, y_max = verts[:, 0].max(), verts[:, 1].max()
        z = -0.001  # just below z=0 so it doesn't z-fight with the floor slab
        # quad: bottom-left, bottom-right, top-right, top-left
        return np.array([
            [x_min, y_min, z],
            [x_max, y_min, z],
            [x_max, y_max, z],
            [x_min, y_max, z],
        ], dtype=np.float32)

    def _load_image_as_png_b64(self, image_path: str) -> Optional[str]:
        """Load image and return as base64-encoded PNG string."""
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return None
            # Flip vertically — OpenCV loads top-down, glTF UVs are bottom-up
            img = cv2.flip(img, 0)
            success, buf = cv2.imencode(".png", img)
            if not success:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            return None

    # ── OBJ ──────────────────────────────────────────────────────────────────

    def _write_obj(self, building: Building3D, out: Path, stem: str,
                   floorplan_image_path: Optional[str],
                   render_image_path: Optional[str] = None) -> str:
        obj_path = out / f"{stem}.obj"
        mtl_path = out / f"{stem}.mtl"

        # Copy floorplan image next to OBJ if provided
        tex_filename = None
        if floorplan_image_path and Path(floorplan_image_path).exists():
            import shutil
            tex_filename = f"{stem}_floorplan.png"
            shutil.copy2(floorplan_image_path, out / tex_filename)

        # Copy render image next to OBJ if provided
        render_filename = None
        if render_image_path and Path(render_image_path).exists():
            import shutil
            render_filename = f"{stem}_render.png"
            if not (out / render_filename).exists():
                shutil.copy2(render_image_path, out / render_filename)

        # Collect materials
        materials: Dict[str, tuple] = {}
        for mesh in building.meshes:
            mat_name = mesh.label.replace(" ", "_")
            materials[mat_name] = mesh.color

        # Write MTL
        with open(mtl_path, "w") as f:
            f.write("# PlanForger material library\n\n")
            if tex_filename:
                f.write("newmtl FloorPlan\n")
                f.write(f"map_Kd {tex_filename}\n")
                f.write("Kd 1.0 1.0 1.0\nKa 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nd 1.0\n\n")
            if render_filename:
                f.write("newmtl InteriorRender\n")
                f.write(f"map_Kd {render_filename}\n")
                f.write("Kd 1.0 1.0 1.0\nKa 1.0 1.0 1.0\nKs 0.1 0.1 0.1\nd 1.0\n\n")
            for mat_name, color in materials.items():
                r, g, b = color
                f.write(f"newmtl {mat_name}\n")
                f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
                f.write("Ka 0.1 0.1 0.1\nKs 0.0 0.0 0.0\nd 1.0\n\n")

        with open(obj_path, "w") as f:
            f.write(f"# PlanForger OBJ export\nmtllib {stem}.mtl\n\n")

            v_offset = 1

            # Ground plane — use render texture if available, else floorplan
            gp = self._ground_plane_vertices(building)
            if gp is not None:
                f.write("o FloorPlan\n")
                if render_filename:
                    f.write("usemtl InteriorRender\n")
                elif tex_filename:
                    f.write("usemtl FloorPlan\n")
                for v in gp:
                    f.write(f"v {v[0]:.6f} {v[2]:.6f} {-v[1]:.6f}\n")
                f.write("vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n")
                o = v_offset
                f.write(f"f {o}/1 {o+1}/2 {o+2}/3\n")
                f.write(f"f {o}/1 {o+2}/3 {o+3}/4\n\n")
                v_offset += 4

            # 3D geometry meshes
            for mesh in building.meshes:
                if mesh.vertex_count == 0:
                    continue
                mat_name = mesh.label.replace(" ", "_")
                f.write(f"o {mesh.label}\nusemtl {mat_name}\n")
                for v in mesh.vertices:
                    f.write(f"v {v[0]:.6f} {v[2]:.6f} {-v[1]:.6f}\n")
                tris = self._triangulate(mesh.faces)
                for tri in tris:
                    indices = " ".join(str(i + v_offset) for i in tri)
                    f.write(f"f {indices}\n")
                f.write("\n")
                v_offset += mesh.vertex_count

        print(f"OBJ exported: {obj_path}")
        return str(obj_path)

    # ── glTF 2.0 ─────────────────────────────────────────────────────────────

    def _write_gltf(self, building: Building3D, out: Path, stem: str,
                    floorplan_image_path: Optional[str],
                    render_image_path: Optional[str] = None) -> str:
        gltf_path = out / f"{stem}.gltf"

        def pad4(b: bytes) -> bytes:
            r = len(b) % 4
            return b + b"\x00" * (4 - r) if r else b

        buffers_data = b""
        buffer_views = []
        accessors    = []
        meshes       = []
        nodes        = []
        materials    = []
        textures     = []
        images       = []

        def add_buffer(data: bytes, target: Optional[int] = None) -> int:
            nonlocal buffers_data
            padded = pad4(data)
            bv_idx = len(buffer_views)
            bv = {"buffer": 0, "byteOffset": len(buffers_data), "byteLength": len(padded)}
            if target:
                bv["target"] = target
            buffer_views.append(bv)
            buffers_data += padded
            return bv_idx

        def add_accessor(bv_idx, comp_type, count, acc_type, min_v=None, max_v=None) -> int:
            acc = {"bufferView": bv_idx, "componentType": comp_type,
                   "count": count, "type": acc_type}
            if min_v is not None: acc["min"] = min_v
            if max_v is not None: acc["max"] = max_v
            accessors.append(acc)
            return len(accessors) - 1

        # ── Ground plane mesh with floorplan texture ──────────────────────
        gp = self._ground_plane_vertices(building)
        has_texture = False
        tex_mat_idx = None

        if gp is not None:
            # Prefer render image for the floor — it shows the interior
            floor_img_path = render_image_path or floorplan_image_path
            img_b64 = None
            if floor_img_path:
                img_b64 = self._load_image_as_png_b64(floor_img_path)

            # Also load floorplan separately for a ceiling plane if render exists
            fp_b64 = None
            if render_image_path and floorplan_image_path:
                fp_b64 = self._load_image_as_png_b64(floorplan_image_path)

            if img_b64:
                images.append({"uri": f"data:image/png;base64,{img_b64}"})
                textures.append({"source": 0})
                materials.append({
                    "name": "InteriorFloor",
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.8,
                    },
                    "doubleSided": True,
                })
                tex_mat_idx = 0
                has_texture = True

            # Ground plane vertices (4 corners) — render texture on floor
            gp_verts = gp.astype(np.float32)
            gp_uvs   = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
            gp_idx   = np.array([0,1,2, 0,2,3], dtype=np.uint32)

            bv_gp_v  = add_buffer(gp_verts.tobytes(), 34962)
            bv_gp_uv = add_buffer(gp_uvs.tobytes(), 34962)
            bv_gp_i  = add_buffer(gp_idx.tobytes(), 34963)

            acc_gp_v  = add_accessor(bv_gp_v,  5126, 4, "VEC3",
                                     gp_verts.min(0).tolist(), gp_verts.max(0).tolist())
            acc_gp_uv = add_accessor(bv_gp_uv, 5126, 4, "VEC2")
            acc_gp_i  = add_accessor(bv_gp_i,  5125, 6, "SCALAR")

            gp_prim = {
                "attributes": {"POSITION": acc_gp_v, "TEXCOORD_0": acc_gp_uv},
                "indices": acc_gp_i,
                "mode": 4,
            }
            if tex_mat_idx is not None:
                gp_prim["material"] = tex_mat_idx

            meshes.append({"name": "InteriorFloor", "primitives": [gp_prim]})
            nodes.append({"mesh": len(meshes) - 1, "name": "InteriorFloor"})

            # ── Ceiling plane with floorplan texture (top-down view) ──────
            # Placed at wall height so you can see the floorplan from above
            if fp_b64:
                images.append({"uri": f"data:image/png;base64,{fp_b64}"})
                fp_tex_idx = len(textures)
                textures.append({"source": len(images) - 1})
                fp_mat_idx = len(materials)
                materials.append({
                    "name": "FloorPlan",
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": fp_tex_idx},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0,
                    },
                    "doubleSided": True,
                })
                # Ceiling quad at wall_height + small offset
                ceil_z = building.wall_height + 0.05
                ceil_verts = np.array([
                    [gp[0,0], gp[0,1], ceil_z],
                    [gp[1,0], gp[1,1], ceil_z],
                    [gp[2,0], gp[2,1], ceil_z],
                    [gp[3,0], gp[3,1], ceil_z],
                ], dtype=np.float32)
                ceil_uvs = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
                ceil_idx = np.array([0,2,1, 0,3,2], dtype=np.uint32)  # flipped winding

                bv_cv  = add_buffer(ceil_verts.tobytes(), 34962)
                bv_cuv = add_buffer(ceil_uvs.tobytes(), 34962)
                bv_ci  = add_buffer(ceil_idx.tobytes(), 34963)

                acc_cv  = add_accessor(bv_cv,  5126, 4, "VEC3",
                                       ceil_verts.min(0).tolist(), ceil_verts.max(0).tolist())
                acc_cuv = add_accessor(bv_cuv, 5126, 4, "VEC2")
                acc_ci  = add_accessor(bv_ci,  5125, 6, "SCALAR")

                ceil_prim = {
                    "attributes": {"POSITION": acc_cv, "TEXCOORD_0": acc_cuv},
                    "indices": acc_ci,
                    "mode": 4,
                    "material": fp_mat_idx,
                }
                meshes.append({"name": "FloorPlan", "primitives": [ceil_prim]})
                nodes.append({"mesh": len(meshes) - 1, "name": "FloorPlan"})

        # ── 3D geometry mesh ──────────────────────────────────────────────
        vertices = building.merged_vertices
        colors   = building.merged_colors
        tris     = self._triangulate(building.merged_faces)

        if tris and len(vertices) > 0:
            indices = np.array(tris, dtype=np.uint32).flatten()

            bv_v = add_buffer(vertices.astype(np.float32).tobytes(), 34962)
            bv_c = add_buffer(colors.astype(np.float32).tobytes(), 34962)
            bv_i = add_buffer(indices.tobytes(), 34963)

            acc_v = add_accessor(bv_v, 5126, len(vertices), "VEC3",
                                 vertices.min(0).tolist(), vertices.max(0).tolist())
            acc_c = add_accessor(bv_c, 5126, len(vertices), "VEC3")
            acc_i = add_accessor(bv_i, 5125, len(indices),  "SCALAR")

            # Unlit vertex-color material for the 3D geometry
            materials.append({
                "name": "Building",
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.9},
            })
            geom_mat_idx = len(materials) - 1

            meshes.append({
                "name": stem,
                "primitives": [{
                    "attributes": {"POSITION": acc_v, "COLOR_0": acc_c},
                    "indices": acc_i,
                    "mode": 4,
                    "material": geom_mat_idx,
                }]
            })
            nodes.append({"mesh": len(meshes) - 1, "name": stem})

        # ── Assemble glTF ─────────────────────────────────────────────────
        scene_nodes = list(range(len(nodes)))

        gltf: dict = {
            "asset": {"version": "2.0", "generator": "PlanForger"},
            "scene": 0,
            "scenes": [{"nodes": scene_nodes}],
            "nodes": nodes,
            "meshes": meshes,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{
                "byteLength": len(buffers_data),
                "uri": "data:application/octet-stream;base64,"
                       + base64.b64encode(buffers_data).decode("ascii"),
            }],
        }
        if materials: gltf["materials"] = materials
        if textures:  gltf["textures"]  = textures
        if images:    gltf["images"]    = images

        with open(gltf_path, "w") as f:
            json.dump(gltf, f, indent=2)

        print(f"glTF exported: {gltf_path}")
        return str(gltf_path)

    # ── Binary STL ────────────────────────────────────────────────────────────

    def _write_stl(self, building: Building3D, out: Path, stem: str) -> str:
        stl_path = out / f"{stem}.stl"
        vertices = building.merged_vertices
        tris = self._triangulate(building.merged_faces)

        with open(stl_path, "wb") as f:
            f.write(f"PlanForger STL {stem}".encode("ascii").ljust(80, b"\x00"))
            f.write(struct.pack("<I", len(tris)))
            for tri in tris:
                if len(tri) < 3: continue
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                nm = np.linalg.norm(normal)
                if nm > 1e-10: normal /= nm
                f.write(struct.pack("<fff", *normal))
                f.write(struct.pack("<fff", *v0))
                f.write(struct.pack("<fff", *v1))
                f.write(struct.pack("<fff", *v2))
                f.write(struct.pack("<H", 0))

        print(f"STL exported: {stl_path}")
        return str(stl_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _triangulate(self, faces: List[List[int]]) -> List[List[int]]:
        tris = []
        for face in faces:
            if len(face) < 3: continue
            if len(face) == 3:
                tris.append(face)
            else:
                for i in range(1, len(face) - 1):
                    tris.append([face[0], face[i], face[i + 1]])
        return tris
