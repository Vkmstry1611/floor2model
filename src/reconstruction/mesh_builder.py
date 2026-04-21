"""
mesh_builder.py
---------------
Merges individual Mesh3D objects into a single unified Building3D model.

Usage:
    from src.reconstruction.mesh_builder import MeshBuilder

    builder = MeshBuilder()
    building = builder.build(meshes)
    print(building.summary)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .extruder import Mesh3D, Extruder


@dataclass
class Building3D:
    """Merged 3D building model ready for export."""
    meshes:           List[Mesh3D]
    merged_vertices:  np.ndarray        # (N, 3) float32
    merged_faces:     List[List[int]]   # list of face index lists
    merged_colors:    np.ndarray        # (N, 3) float32 per-vertex RGB
    wall_height:      float = 2.8
    floor_area_m2:    float = 0.0
    room_count:       int   = 0

    @property
    def vertex_count(self) -> int:
        return len(self.merged_vertices)

    @property
    def face_count(self) -> int:
        return len(self.merged_faces)

    @property
    def summary(self) -> dict:
        return {
            "vertices":    self.vertex_count,
            "faces":       self.face_count,
            "meshes":      len(self.meshes),
            "wall_height": self.wall_height,
            "floor_area":  round(self.floor_area_m2, 2),
            "rooms":       self.room_count,
        }


class MeshBuilder:
    """
    Merges a list of Mesh3D objects into a single Building3D.

    Args:
        add_floor:      Whether to add a floor slab under the building.
        wall_height:    Default wall extrusion height (metres).
    """

    def __init__(self, add_floor: bool = True, wall_height: float = 2.8):
        self.add_floor = add_floor
        self.wall_height = wall_height

    def build(self, meshes: List[Mesh3D], floor_area_m2: float = 0.0, room_count: int = 0) -> Building3D:
        """
        Merge all meshes into one Building3D.

        Args:
            meshes:        List of Mesh3D objects from Extruder.
            floor_area_m2: Total floor area in m² (metadata only).
            room_count:    Number of rooms (metadata only).

        Returns:
            Building3D with merged geometry.
        """
        if not meshes:
            raise ValueError("Cannot build from empty mesh list.")

        all_vertices: List[np.ndarray] = []
        all_faces:    List[List[int]]  = []
        all_colors:   List[np.ndarray] = []
        vertex_offset = 0

        for mesh in meshes:
            if mesh.vertex_count == 0:
                continue

            all_vertices.append(mesh.vertices)

            # Offset face indices by current vertex count
            for face in mesh.faces:
                all_faces.append([idx + vertex_offset for idx in face])

            # Per-vertex color from mesh color
            color = np.array(mesh.color, dtype=np.float32)
            colors = np.tile(color, (mesh.vertex_count, 1))
            all_colors.append(colors)

            vertex_offset += mesh.vertex_count

        if not all_vertices:
            raise ValueError("All meshes were empty — nothing to merge.")

        merged_vertices = np.vstack(all_vertices).astype(np.float32)
        merged_colors   = np.vstack(all_colors).astype(np.float32)

        return Building3D(
            meshes=meshes,
            merged_vertices=merged_vertices,
            merged_faces=all_faces,
            merged_colors=merged_colors,
            wall_height=self.wall_height,
            floor_area_m2=floor_area_m2,
            room_count=room_count,
        )

    def build_from_geometry(self, geometry_result) -> Building3D:
        """
        Convenience method: extrude all polygons from a GeometryResult
        and merge into a Building3D.

        Args:
            geometry_result: GeometryResult from GeometryPipeline.

        Returns:
            Building3D ready for export.
        """
        extruder = Extruder(
            wall_height=self.wall_height,
            pixels_per_metre=geometry_result.scale.pixels_per_metre,
        )

        meshes = extruder.extrude_all(geometry_result.vectorization)

        # Compute floor area from room nodes
        floor_area = sum(n.area_m2 for n in geometry_result.graph.nodes)
        room_count = len(geometry_result.graph.nodes)

        return self.build(meshes, floor_area_m2=floor_area, room_count=room_count)
