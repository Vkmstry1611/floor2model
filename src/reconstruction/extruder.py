from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
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
        "OuterWall":(0.25,0.25,0.35),"InnerWall":(0.40,0.40,0.50),
        "Window":(0.60,0.80,0.95),"Door":(0.65,0.45,0.25),
        "Kitchen":(0.95,0.85,0.60),"LivingRoom":(0.80,0.90,0.75),
        "Bedroom":(0.75,0.80,0.95),"Bathroom":(0.70,0.90,0.90),
        "Corridor":(0.85,0.85,0.80),"Balcony":(0.70,0.95,0.70),
        "Stairs":(0.55,0.55,0.55),"Garage":(0.65,0.60,0.55),
        "Floor":(0.60,0.55,0.50),
    }

    # Realistic per-class heights (metres) based on standard AEC dimensions
    CLASS_HEIGHTS = {
        "OuterWall":  3.0,   # exterior walls — full floor-to-ceiling + slab
        "InnerWall":  2.7,   # interior partition walls — standard ceiling height
        "Door":       2.1,   # standard door height
        "Window":     1.2,   # window height (sill ~0.9m, head ~2.1m)
        "Stairs":     3.0,   # full storey height
        "Railing":    1.1,   # balustrade / handrail height
        "Balcony":    0.15,  # balcony slab thickness
        "Garage":     2.4,   # garage door / ceiling clearance
        # Rooms get a thin floor slab — they are spaces, not solid volumes
        "Kitchen":    0.15,
        "LivingRoom": 0.15,
        "Bedroom":    0.15,
        "Bathroom":   0.15,
        "Corridor":   0.15,
        "Floor":      0.2,
    }

    def __init__(self, wall_height=2.7, floor_thickness=0.2, pixels_per_metre=100.0):
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        self.pixels_per_metre = pixels_per_metre
        # Global origin in pixels — set once before extruding so all meshes
        # share the same coordinate frame and stay in their correct positions.
        self._global_origin_px: Optional[np.ndarray] = None

    def set_global_origin(self, origin_px: np.ndarray):
        """
        Set the shared pixel-space origin used by all subsequent extrusions.
        All meshes will be translated relative to this point so their
        positions match the original detection layout.

        Args:
            origin_px: (2,) array — the pixel coordinate that maps to (0, 0) in metres.
                       Typically the centroid of all detection bounding boxes.
        """
        self._global_origin_px = np.array(origin_px, dtype=np.float32)

    def _get_height(self, label: str) -> float:
        """Return realistic height for a given class label."""
        return self.CLASS_HEIGHTS.get(label, self.wall_height)

    def extrude_polygon(self, points_px, label="Wall", height=None, base_z=0.0):
        # Use per-class realistic height unless explicitly overridden
        h = height if height is not None else self._get_height(label)
        color = self.CLASS_COLORS.get(label, (0.7,0.7,0.7))
        pts_m = self._px_to_metres(points_px)
        if len(pts_m) > 1 and np.allclose(pts_m[0], pts_m[-1]):
            pts_m = pts_m[:-1]
        n = len(pts_m)
        if n < 3:
            return Mesh3D(np.zeros((0,3),dtype=np.float32),[],label=label,color=color)
        bottom = np.column_stack([pts_m[:,0],pts_m[:,1],np.full(n,base_z)]).astype(np.float32)
        top    = np.column_stack([pts_m[:,0],pts_m[:,1],np.full(n,base_z+h)]).astype(np.float32)
        vertices = np.vstack([bottom, top])
        faces = [list(range(n-1,-1,-1)), list(range(n,2*n))]
        for i in range(n):
            j = (i+1)%n
            faces.append([i, j, j+n, i+n])
        normals = self._compute_vertex_normals(vertices, faces)
        return Mesh3D(vertices, faces, normals, label, color)

    def extrude_floor_slab(self, points_px, label="Floor"):
        return self.extrude_polygon(points_px, label=label,
                                    height=self.floor_thickness,
                                    base_z=-self.floor_thickness)

    # ── Bbox-based generation ─────────────────────────────────────────────────

    # Standard wall thickness in metres (interior ~0.15m, exterior ~0.25m)
    WALL_THICKNESS = {"OuterWall": 0.25, "InnerWall": 0.15}
    # Standard door dimensions in metres
    DOOR_WIDTH_M   = 0.9
    DOOR_HEIGHT_M  = 2.1
    DOOR_THICKNESS_M = 0.1   # door panel depth

    def extrude_bbox_wall(self, bbox_px, label="OuterWall", base_z=0.0):
        """
        Generate a 3D wall box from a detection bounding box.

        Uses the full bbox footprint — the detection already captures the
        wall's real shape. The longer axis is the wall run, the shorter
        axis is the wall thickness (clamped to a realistic minimum so thin
        detections still produce solid geometry).

        Args:
            bbox_px: (x1, y1, x2, y2) in pixels
            label:   "OuterWall", "InnerWall", or "wall"
            base_z:  Z offset for the bottom of the wall
        """
        x1, y1, x2, y2 = bbox_px
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)

        # Minimum wall thickness in pixels so it's always visible
        min_thickness_m = self.WALL_THICKNESS.get(label, self.WALL_THICKNESS.get(label.capitalize(), 0.2))
        min_thickness_px = min_thickness_m * self.pixels_per_metre

        if w_px >= h_px:
            # Horizontal wall — enforce minimum thickness on Y axis
            cy = (y1 + y2) / 2.0
            half_t = max(h_px / 2.0, min_thickness_px / 2.0)
            rect_pts = [
                (x1, cy - half_t),
                (x2, cy - half_t),
                (x2, cy + half_t),
                (x1, cy + half_t),
            ]
        else:
            # Vertical wall — enforce minimum thickness on X axis
            cx = (x1 + x2) / 2.0
            half_t = max(w_px / 2.0, min_thickness_px / 2.0)
            rect_pts = [
                (cx - half_t, y1),
                (cx + half_t, y1),
                (cx + half_t, y2),
                (cx - half_t, y2),
            ]

        wall_h = self._get_height(label)
        return self.extrude_polygon(rect_pts, label=label, height=wall_h, base_z=base_z)

    def extrude_bbox_door(self, bbox_px, base_z=0.0):
        """
        Generate a 3D door from a detection bounding box.

        Uses the bbox's actual pixel width/position for placement so the
        door sits exactly where it was detected. Height is always the
        standard 2.1m regardless of bbox height (which is a 2D floorplan
        symbol, not a real height).

        Args:
            bbox_px: (x1, y1, x2, y2) in pixels
            base_z:  Z offset for the bottom of the door
        """
        x1, y1, x2, y2 = bbox_px
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)

        # Minimum door thickness in pixels so it's always visible
        min_t_px = self.DOOR_THICKNESS_M * self.pixels_per_metre

        if w_px >= h_px:
            # Door opening runs along X axis
            cy = (y1 + y2) / 2.0
            half_t = max(h_px / 2.0, min_t_px / 2.0)
            rect_pts = [
                (x1, cy - half_t),
                (x2, cy - half_t),
                (x2, cy + half_t),
                (x1, cy + half_t),
            ]
        else:
            # Door opening runs along Y axis
            cx = (x1 + x2) / 2.0
            half_t = max(w_px / 2.0, min_t_px / 2.0)
            rect_pts = [
                (cx - half_t, y1),
                (cx + half_t, y1),
                (cx + half_t, y2),
                (cx - half_t, y2),
            ]

        return self.extrude_polygon(
            rect_pts, label="Door",
            height=self.DOOR_HEIGHT_M,
            base_z=base_z,
        )

    def extrude_all(self, vectorization_result):
        meshes = []
        for poly in vectorization_result.all_polygons:
            if len(poly.points) < 3: continue
            mesh = self.extrude_polygon(poly.points, label=poly.class_name)
            if mesh.vertex_count > 0:
                meshes.append(mesh)
        return meshes

    def _px_to_metres(self, points_px):
        pts = np.array(points_px, dtype=np.float32)
        pts_m = pts / self.pixels_per_metre
        # Use shared global origin so all meshes stay in their correct
        # relative positions. Fall back to per-mesh centering only if no
        # global origin has been set (e.g. standalone calls).
        if self._global_origin_px is not None:
            origin_m = self._global_origin_px / self.pixels_per_metre
            pts_m -= origin_m
        else:
            pts_m -= pts_m.mean(axis=0)
        pts_m[:, 1] *= -1   # flip Y so image-top → 3D-front
        return pts_m

    def _compute_vertex_normals(self, vertices, faces):
        normals = np.zeros_like(vertices)
        for face in faces:
            if len(face) < 3: continue
            v0,v1,v2 = vertices[face[0]],vertices[face[1]],vertices[face[2]]
            fn = np.cross(v1-v0, v2-v0)
            nm = np.linalg.norm(fn)
            if nm > 1e-10: fn /= nm
            for idx in face: normals[idx] += fn
        norms = np.linalg.norm(normals,axis=1,keepdims=True)
        norms = np.where(norms<1e-10,1.0,norms)
        return (normals/norms).astype(np.float32)
