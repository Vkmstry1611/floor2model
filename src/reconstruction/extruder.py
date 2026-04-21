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

    def __init__(self, wall_height=2.8, floor_thickness=0.2, pixels_per_metre=100.0):
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        self.pixels_per_metre = pixels_per_metre

    def extrude_polygon(self, points_px, label="Wall", height=None, base_z=0.0):
        h = height if height is not None else self.wall_height
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
        pts_m -= pts_m.mean(axis=0)
        pts_m[:,1] *= -1
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
