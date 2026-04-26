from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class Mesh3D:
    vertices: np.ndarray
    faces: list
    normals: Optional[np.ndarray] = None
    label: str = "unknown"
    color: tuple = (0.8, 0.8, 0.8)

    @property
    def vertex_count(self): return len(self.vertices)

    @property
    def face_count(self): return len(self.faces)


class Extruder:
    CLASS_COLORS = {
        "OuterWall":  (0.25, 0.25, 0.35),
        "InnerWall":  (0.40, 0.40, 0.50),
        "wall":       (0.35, 0.35, 0.45),
        "Window":     (0.60, 0.80, 0.95),
        "window":     (0.60, 0.80, 0.95),
        "Door":       (0.65, 0.45, 0.25),
        "door":       (0.65, 0.45, 0.25),
        "Kitchen":    (0.95, 0.85, 0.60),
        "LivingRoom": (0.80, 0.90, 0.75),
        "Bedroom":    (0.75, 0.80, 0.95),
        "Bathroom":   (0.70, 0.90, 0.90),
        "Corridor":   (0.85, 0.85, 0.80),
        "Balcony":    (0.70, 0.95, 0.70),
        "Stairs":     (0.55, 0.55, 0.55),
        "Garage":     (0.65, 0.60, 0.55),
        "Floor":      (0.60, 0.55, 0.50),
    }

    CLASS_HEIGHTS = {
        "OuterWall": 3.0,
        "InnerWall": 2.7,
        "wall":      2.8,
        "Door":      2.1,
        "door":      2.1,
        "Window":    1.2,
        "window":    1.2,
        "Stairs":    3.0,
        "Railing":   1.1,
        "Balcony":   0.15,
        "Garage":    2.4,
        "Kitchen":   0.15,
        "LivingRoom":0.15,
        "Bedroom":   0.15,
        "Bathroom":  0.15,
        "Corridor":  0.15,
        "Floor":     0.2,
    }

    # Realistic wall thickness (metres)
    WALL_THICKNESS_M = {"OuterWall": 0.25, "InnerWall": 0.15, "wall": 0.20}
    # Window sill height above floor (metres)
    WINDOW_SILL_M    = 0.9
    # Lintel thickness above door/window opening (metres)
    LINTEL_H_M       = 0.2

    def __init__(self, wall_height: float = 2.8,
                 floor_thickness: float = 0.2,
                 pixels_per_metre: float = 100.0):
        self.wall_height      = wall_height
        self.floor_thickness  = floor_thickness
        self.pixels_per_metre = pixels_per_metre
        self._global_origin_px: Optional[np.ndarray] = None

    def set_global_origin(self, origin_px: np.ndarray):
        self._global_origin_px = np.array(origin_px, dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def build_wall_with_openings(
        self,
        wall_bbox_px: Tuple[int,int,int,int],
        openings: List[dict],          # list of {"bbox": (x1,y1,x2,y2), "type": "door"|"window"}
        label: str = "wall",
    ) -> List[Mesh3D]:
        """
        Build a wall with door/window openings cut into it.

        Strategy
        --------
        1. Determine wall orientation (horizontal / vertical) from bbox.
        2. Project every opening onto the wall's long axis → 1-D intervals.
        3. Split the wall into solid segments around each opening.
        4. For each opening:
           - Door  → gap in wall (no solid), door-frame panel at same XY
           - Window → sill segment + lintel segment + glass panel at sill height
        5. Return all Mesh3D pieces.
        """
        meshes: List[Mesh3D] = []
        wx1, wy1, wx2, wy2 = wall_bbox_px
        w_px = abs(wx2 - wx1)
        h_px = abs(wy2 - wy1)
        horizontal = w_px >= h_px

        wall_h = self._get_height(label)
        thick_m = self.WALL_THICKNESS_M.get(label, 0.20)
        thick_px = thick_m * self.pixels_per_metre

        # ── Wall centre-line and thickness ───────────────────────────────
        if horizontal:
            long_lo, long_hi = wx1, wx2          # along X
            cross_c = (wy1 + wy2) / 2.0          # centre Y
            half_t  = max(h_px / 2.0, thick_px / 2.0)
        else:
            long_lo, long_hi = wy1, wy2          # along Y
            cross_c = (wx1 + wx2) / 2.0          # centre X
            half_t  = max(w_px / 2.0, thick_px / 2.0)

        # ── Project openings onto long axis ──────────────────────────────
        # Each opening becomes an interval [op_lo, op_hi] on the long axis
        # plus a Z-profile (base_z, top_z) for the gap.
        intervals: List[dict] = []
        for op in openings:
            ox1, oy1, ox2, oy2 = op["bbox"]
            otype = op["type"]   # "door" or "window"

            if horizontal:
                op_lo = max(long_lo, min(ox1, ox2))
                op_hi = min(long_hi, max(ox1, ox2))
            else:
                op_lo = max(long_lo, min(oy1, oy2))
                op_hi = min(long_hi, max(oy1, oy2))

            if op_hi <= op_lo:
                continue   # opening doesn't overlap this wall

            if otype == "door":
                gap_base = 0.0
                gap_top  = self._get_height("door")
            else:  # window
                gap_base = self.WINDOW_SILL_M
                gap_top  = self.WINDOW_SILL_M + self._get_height("window")

            intervals.append({
                "lo": op_lo, "hi": op_hi,
                "base": gap_base, "top": gap_top,
                "type": otype,
                "bbox": op["bbox"],
            })

        # Sort intervals along long axis
        intervals.sort(key=lambda x: x["lo"])

        # ── Build wall segments ───────────────────────────────────────────
        # Walk along the long axis, emitting solid wall pieces between gaps.
        # For each gap we also emit sill/lintel pieces and the opening mesh.

        def _wall_seg(lo, hi, base_z, top_z):
            """Emit one rectangular wall segment."""
            seg_h = top_z - base_z
            if seg_h <= 0 or hi <= lo:
                return
            if horizontal:
                pts = [(lo, cross_c - half_t),
                       (hi, cross_c - half_t),
                       (hi, cross_c + half_t),
                       (lo, cross_c + half_t)]
            else:
                pts = [(cross_c - half_t, lo),
                       (cross_c + half_t, lo),
                       (cross_c + half_t, hi),
                       (cross_c - half_t, hi)]
            m = self.extrude_polygon(pts, label=label,
                                     height=seg_h, base_z=base_z)
            if m.vertex_count > 0:
                meshes.append(m)

        cursor = long_lo
        for iv in intervals:
            # Solid wall before this opening (full height)
            _wall_seg(cursor, iv["lo"], 0.0, wall_h)

            # Sill below opening (only for windows)
            if iv["type"] == "window" and iv["base"] > 0:
                _wall_seg(iv["lo"], iv["hi"], 0.0, iv["base"])

            # Lintel above opening
            lintel_base = iv["top"]
            if lintel_base < wall_h:
                _wall_seg(iv["lo"], iv["hi"], lintel_base, wall_h)

            # The opening element itself
            if iv["type"] == "door":
                m = self._door_panel(iv["bbox"], horizontal, cross_c, half_t)
                if m and m.vertex_count > 0:
                    meshes.append(m)
            else:
                m = self._window_glass(iv["bbox"], horizontal, cross_c, half_t)
                if m and m.vertex_count > 0:
                    meshes.append(m)

            cursor = iv["hi"]

        # Remaining wall after last opening
        _wall_seg(cursor, long_hi, 0.0, wall_h)

        return meshes

    def extrude_bbox_wall(self, bbox_px, label: str = "wall",
                          base_z: float = 0.0) -> Mesh3D:
        """Simple solid wall box — used when no openings overlap."""
        x1, y1, x2, y2 = bbox_px
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)
        thick_px = self.WALL_THICKNESS_M.get(
            label, self.WALL_THICKNESS_M.get(label.capitalize(), 0.20)
        ) * self.pixels_per_metre

        if w_px >= h_px:
            cy     = (y1 + y2) / 2.0
            half_t = max(h_px / 2.0, thick_px / 2.0)
            pts = [(x1, cy - half_t), (x2, cy - half_t),
                   (x2, cy + half_t), (x1, cy + half_t)]
        else:
            cx     = (x1 + x2) / 2.0
            half_t = max(w_px / 2.0, thick_px / 2.0)
            pts = [(cx - half_t, y1), (cx + half_t, y1),
                   (cx + half_t, y2), (cx - half_t, y2)]

        return self.extrude_polygon(pts, label=label,
                                    height=self._get_height(label),
                                    base_z=base_z)

    def extrude_standalone_window(self, bbox_px) -> Mesh3D:
        """
        Window not embedded in a detected wall — float it at sill height.
        """
        x1, y1, x2, y2 = bbox_px
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)
        thick_px = 0.08 * self.pixels_per_metre   # 8 cm glass thickness

        if w_px >= h_px:
            cy     = (y1 + y2) / 2.0
            half_t = max(h_px / 2.0, thick_px / 2.0)
            pts = [(x1, cy - half_t), (x2, cy - half_t),
                   (x2, cy + half_t), (x1, cy + half_t)]
        else:
            cx     = (x1 + x2) / 2.0
            half_t = max(w_px / 2.0, thick_px / 2.0)
            pts = [(cx - half_t, y1), (cx + half_t, y1),
                   (cx + half_t, y2), (cx - half_t, y2)]

        return self.extrude_polygon(pts, label="Window",
                                    height=self._get_height("Window"),
                                    base_z=self.WINDOW_SILL_M)

    def extrude_polygon(self, points_px, label="Wall",
                        height=None, base_z=0.0) -> Mesh3D:
        h     = height if height is not None else self._get_height(label)
        color = self.CLASS_COLORS.get(label, (0.7, 0.7, 0.7))
        pts_m = self._px_to_metres(points_px)
        if len(pts_m) > 1 and np.allclose(pts_m[0], pts_m[-1]):
            pts_m = pts_m[:-1]
        n = len(pts_m)
        if n < 3:
            return Mesh3D(np.zeros((0,3), dtype=np.float32), [],
                          label=label, color=color)
        bottom = np.column_stack(
            [pts_m[:,0], pts_m[:,1], np.full(n, base_z)]).astype(np.float32)
        top    = np.column_stack(
            [pts_m[:,0], pts_m[:,1], np.full(n, base_z + h)]).astype(np.float32)
        vertices = np.vstack([bottom, top])
        faces = [list(range(n-1, -1, -1)), list(range(n, 2*n))]
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j+n, i+n])
        normals = self._compute_vertex_normals(vertices, faces)
        return Mesh3D(vertices, faces, normals, label, color)

    def extrude_floor_slab(self, points_px, label="Floor") -> Mesh3D:
        return self.extrude_polygon(points_px, label=label,
                                    height=self.floor_thickness,
                                    base_z=-self.floor_thickness)

    def extrude_all(self, vectorization_result):
        meshes = []
        for poly in vectorization_result.all_polygons:
            if len(poly.points) < 3:
                continue
            mesh = self.extrude_polygon(poly.points, label=poly.class_name)
            if mesh.vertex_count > 0:
                meshes.append(mesh)
        return meshes

    # ── Private helpers ───────────────────────────────────────────────────────

    def _door_panel(self, bbox_px, horizontal: bool,
                    cross_c: float, half_t: float) -> Optional[Mesh3D]:
        """
        Thin door panel (same footprint as the gap, full door height).
        Coloured brown so it's visually distinct from the wall.
        """
        x1, y1, x2, y2 = bbox_px
        panel_thick_px = 0.05 * self.pixels_per_metre   # 5 cm door panel

        if horizontal:
            cy     = cross_c
            half_p = min(half_t, panel_thick_px / 2.0)
            pts = [(x1, cy - half_p), (x2, cy - half_p),
                   (x2, cy + half_p), (x1, cy + half_p)]
        else:
            cx     = cross_c
            half_p = min(half_t, panel_thick_px / 2.0)
            pts = [(cx - half_p, y1), (cx + half_p, y1),
                   (cx + half_p, y2), (cx - half_p, y2)]

        return self.extrude_polygon(pts, label="Door",
                                    height=self._get_height("door"),
                                    base_z=0.0)

    def _window_glass(self, bbox_px, horizontal: bool,
                      cross_c: float, half_t: float) -> Optional[Mesh3D]:
        """
        Glass pane centred in the wall thickness, at sill height.
        """
        x1, y1, x2, y2 = bbox_px
        glass_thick_px = 0.04 * self.pixels_per_metre   # 4 cm glass

        if horizontal:
            cy     = cross_c
            half_g = min(half_t, glass_thick_px / 2.0)
            pts = [(x1, cy - half_g), (x2, cy - half_g),
                   (x2, cy + half_g), (x1, cy + half_g)]
        else:
            cx     = cross_c
            half_g = min(half_t, glass_thick_px / 2.0)
            pts = [(cx - half_g, y1), (cx + half_g, y1),
                   (cx + half_g, y2), (cx - half_g, y2)]

        return self.extrude_polygon(pts, label="Window",
                                    height=self._get_height("Window"),
                                    base_z=self.WINDOW_SILL_M)

    def _get_height(self, label: str) -> float:
        return self.CLASS_HEIGHTS.get(label, self.wall_height)

    def _px_to_metres(self, points_px) -> np.ndarray:
        pts   = np.array(points_px, dtype=np.float32)
        pts_m = pts / self.pixels_per_metre
        if self._global_origin_px is not None:
            pts_m -= self._global_origin_px / self.pixels_per_metre
        else:
            pts_m -= pts_m.mean(axis=0)
        pts_m[:, 1] *= -1
        return pts_m

    def _compute_vertex_normals(self, vertices, faces) -> np.ndarray:
        normals = np.zeros_like(vertices)
        for face in faces:
            if len(face) < 3:
                continue
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            fn = np.cross(v1 - v0, v2 - v0)
            nm = np.linalg.norm(fn)
            if nm > 1e-10:
                fn /= nm
            for idx in face:
                normals[idx] += fn
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (normals / norms).astype(np.float32)
