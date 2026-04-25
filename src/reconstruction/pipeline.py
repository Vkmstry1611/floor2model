"""
pipeline.py
-----------
Phase 4: 3D Reconstruction Pipeline.

Converts GeometryResult (Phase 3) into a full 3D building model
and exports it to OBJ, glTF 2.0, and optionally STL.

Usage:
    from src.reconstruction.pipeline import ReconstructionPipeline

    pipeline = ReconstructionPipeline()
    model = pipeline.reconstruct(geometry_result, output_dir="outputs/models")
    print(model.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from .extruder import Extruder, Mesh3D
from .mesh_builder import MeshBuilder, Building3D
from .exporter import ModelExporter


@dataclass
class ReconstructionConfig:
    """Tunable parameters for Phase 4 reconstruction."""
    wall_height:      float = 2.8    # metres
    floor_thickness:  float = 0.2    # metres
    add_floor:        bool  = True   # add floor slab
    export_obj:       bool  = True
    export_gltf:      bool  = True
    export_stl:       bool  = False
    output_dir:       str   = "outputs/models"


@dataclass
class Model3D:
    """Full 3D reconstruction result."""
    building:     Building3D
    export_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        b = self.building
        parts = [
            f"vertices={b.vertex_count}",
            f"faces={b.face_count}",
            f"rooms={b.room_count}",
            f"floor_area={b.floor_area_m2:.1f}m²",
        ]
        if self.export_paths:
            parts.append("exports=" + ",".join(self.export_paths.keys()))
        return "Model3D(" + ", ".join(parts) + ")"


class ReconstructionPipeline:
    """
    Phase 4: Extrudes 2D geometry into a 3D building model and exports it.

    Example:
        pipeline = ReconstructionPipeline()
        model = pipeline.reconstruct(geo_result, output_dir="outputs/models")
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()

    def reconstruct(
        self,
        geometry_result,
        output_dir: Optional[str] = None,
        stem: Optional[str] = None,
        floorplan_image_path: Optional[str] = None,
    ) -> Model3D:
        """
        Run the full Phase 4 pipeline.

        Args:
            geometry_result:      GeometryResult from GeometryPipeline (Phase 3).
            output_dir:           Override output directory.
            stem:                 Override output filename stem.
            floorplan_image_path: Path to original floor plan image (used as ground texture).

        Returns:
            Model3D with building geometry and export paths.
        """
        cfg = self.config
        out_dir = output_dir or cfg.output_dir
        file_stem = stem or Path(geometry_result.source_path).stem

        # ── Step 1: Extrude polygons → meshes ─────────────────────────────
        print("[1/3] Extruding 2D polygons to 3D meshes...")
        extruder = Extruder(
            wall_height=cfg.wall_height,
            floor_thickness=cfg.floor_thickness,
            pixels_per_metre=geometry_result.scale.pixels_per_metre,
        )

        # ── Compute global origin from ALL detection points ────────────────
        # Collect every pixel coordinate from polygons + raw bboxes so that
        # all meshes share one coordinate frame and stay in their correct
        # positions relative to each other (matching the detection layout).
        seg = getattr(geometry_result, "segmentation_result", None)
        all_pts_for_origin = []

        for poly in geometry_result.vectorization.all_polygons:
            if poly.points:
                all_pts_for_origin.extend(poly.points)

        if seg is not None:
            for elem in seg.elements:
                x1, y1, x2, y2 = elem.bbox
                all_pts_for_origin.extend([(x1, y1), (x2, y2)])

        if all_pts_for_origin:
            origin = np.array(all_pts_for_origin, dtype=np.float32).mean(axis=0)
            extruder.set_global_origin(origin)
            print(f"  Global origin: ({origin[0]:.1f}, {origin[1]:.1f}) px")

        meshes: list[Mesh3D] = []

        WALL_LABELS = {"OuterWall", "InnerWall", "wall"}
        DOOR_LABELS = {"Door", "door"}

        seg = getattr(geometry_result, "segmentation_result", None)
        has_raw_detections = seg is not None and len(seg.elements) > 0

        if has_raw_detections:
            # ── Primary path: bbox-based extrusion ────────────────────────
            # When raw detections exist, generate walls and doors ONLY from
            # their bounding boxes — no polygon duplicates.
            # Non-wall/door polygons (windows, rooms) still come from masks.
            wall_count = door_count = 0
            for elem in seg.elements:
                if elem.class_name in WALL_LABELS:
                    m = extruder.extrude_bbox_wall(elem.bbox, label=elem.class_name)
                    if m.vertex_count > 0:
                        meshes.append(m)
                        wall_count += 1
                elif elem.class_name in DOOR_LABELS:
                    m = extruder.extrude_bbox_door(elem.bbox)
                    if m.vertex_count > 0:
                        meshes.append(m)
                        door_count += 1

            print(f"  Bbox walls: {wall_count}  Bbox doors: {door_count}")

            # Add non-wall/door polygons from vectorization (windows, rooms, etc.)
            skip_labels = WALL_LABELS | DOOR_LABELS
            for poly in geometry_result.vectorization.all_polygons:
                if poly.class_name in skip_labels or len(poly.points) < 3:
                    continue
                mesh = extruder.extrude_polygon(poly.points, label=poly.class_name)
                if mesh.vertex_count > 0:
                    meshes.append(mesh)

        else:
            # ── Fallback: polygon-based extrusion (no raw detections) ─────
            for poly in geometry_result.vectorization.all_polygons:
                if len(poly.points) < 3:
                    continue
                mesh = extruder.extrude_polygon(poly.points, label=poly.class_name)
                if mesh.vertex_count > 0:
                    meshes.append(mesh)

        # Add floor slab from outer wall footprint
        if cfg.add_floor and geometry_result.vectorization.walls:
            outer_walls = [w for w in geometry_result.vectorization.walls
                           if w.class_name == "OuterWall"]
            if outer_walls:
                largest = max(outer_walls, key=lambda w: w.area)
                floor_mesh = extruder.extrude_floor_slab(largest.points, label="Floor")
                if floor_mesh.vertex_count > 0:
                    meshes.append(floor_mesh)

        print(f"  Extruded {len(meshes)} meshes")

        # ── Step 2: Merge meshes → Building3D ─────────────────────────────
        print("[2/3] Merging meshes into building model...")
        builder = MeshBuilder(add_floor=False, wall_height=cfg.wall_height)

        floor_area = sum(n.area_m2 for n in geometry_result.graph.nodes)
        room_count = len(geometry_result.graph.nodes)

        building = builder.build(meshes, floor_area_m2=floor_area, room_count=room_count)
        print(f"  {building.vertex_count} vertices, {building.face_count} faces")
        print(f"  {room_count} rooms, {floor_area:.1f} m² total floor area")

        # ── Step 3: Export ─────────────────────────────────────────────────
        print("[3/3] Exporting 3D model...")
        exporter = ModelExporter(
            export_obj=cfg.export_obj,
            export_gltf=cfg.export_gltf,
            export_stl=cfg.export_stl,
        )
        export_paths = exporter.export(building, out_dir, file_stem,
                                       floorplan_image_path=floorplan_image_path)

        print(f"\nPhase 4 complete. Files saved to: {out_dir}/")
        for fmt, path in export_paths.items():
            print(f"  {fmt.upper()}: {path}")

        return Model3D(building=building, export_paths=export_paths)
