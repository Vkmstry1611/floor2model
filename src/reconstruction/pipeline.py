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
        render_image_path: Optional[str] = None,
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

        WALL_LABELS   = {"OuterWall", "InnerWall", "wall"}
        DOOR_LABELS   = {"Door", "door"}
        WINDOW_LABELS = {"Window", "window"}

        seg = getattr(geometry_result, "segmentation_result", None)
        has_raw_detections = seg is not None and len(seg.elements) > 0

        if has_raw_detections:
            # Separate elements by type
            wall_elems   = [e for e in seg.elements if e.class_name in WALL_LABELS]
            door_elems   = [e for e in seg.elements if e.class_name in DOOR_LABELS]
            window_elems = [e for e in seg.elements if e.class_name in WINDOW_LABELS]

            # ── Match openings to walls ───────────────────────────────────
            # For each door/window, find the wall whose bbox it overlaps most.
            # An opening overlaps a wall if their bboxes intersect.
            def _bbox_overlap(a, b):
                """Return True if bboxes (x1,y1,x2,y2) overlap."""
                ax1,ay1,ax2,ay2 = a
                bx1,by1,bx2,by2 = b
                return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

            def _overlap_area(a, b):
                ax1,ay1,ax2,ay2 = a
                bx1,by1,bx2,by2 = b
                ix = max(0, min(ax2,bx2) - max(ax1,bx1))
                iy = max(0, min(ay2,by2) - max(ay1,by1))
                return ix * iy

            # wall_idx → list of opening dicts
            wall_openings = {i: [] for i in range(len(wall_elems))}
            assigned_doors   = set()
            assigned_windows = set()

            for oi, op_elem in enumerate(door_elems + window_elems):
                otype = "door" if op_elem.class_name in DOOR_LABELS else "window"
                best_wall, best_area = -1, 0
                for wi, w in enumerate(wall_elems):
                    if _bbox_overlap(w.bbox, op_elem.bbox):
                        a = _overlap_area(w.bbox, op_elem.bbox)
                        if a > best_area:
                            best_area, best_wall = a, wi
                if best_wall >= 0:
                    wall_openings[best_wall].append(
                        {"bbox": op_elem.bbox, "type": otype}
                    )
                    if otype == "door":
                        assigned_doors.add(oi)
                    else:
                        assigned_windows.add(oi - len(door_elems))

            # ── Extrude walls (with openings cut in) ─────────────────────
            wall_count = door_count = window_count = 0
            for wi, w_elem in enumerate(wall_elems):
                ops = wall_openings[wi]
                if ops:
                    pieces = extruder.build_wall_with_openings(
                        w_elem.bbox, ops, label=w_elem.class_name
                    )
                    meshes.extend(pieces)
                    door_count   += sum(1 for o in ops if o["type"] == "door")
                    window_count += sum(1 for o in ops if o["type"] == "window")
                else:
                    m = extruder.extrude_bbox_wall(
                        w_elem.bbox, label=w_elem.class_name
                    )
                    if m.vertex_count > 0:
                        meshes.append(m)
                wall_count += 1

            # ── Unassigned doors (not overlapping any wall) ───────────────
            for oi, d_elem in enumerate(door_elems):
                if oi not in assigned_doors:
                    # Standalone door — simple panel at floor level
                    m = extruder.extrude_bbox_wall(
                        d_elem.bbox, label="door"
                    )
                    if m.vertex_count > 0:
                        meshes.append(m)
                    door_count += 1

            # ── Unassigned windows (not overlapping any wall) ─────────────
            for oi, w_elem in enumerate(window_elems):
                if oi not in assigned_windows:
                    m = extruder.extrude_standalone_window(w_elem.bbox)
                    if m.vertex_count > 0:
                        meshes.append(m)
                    window_count += 1

            print(f"  Walls: {wall_count}  Doors: {door_count}  Windows: {window_count}")

            # Non-wall/door/window polygons from vectorization (rooms etc.)
            skip_labels = WALL_LABELS | DOOR_LABELS | WINDOW_LABELS
            for poly in geometry_result.vectorization.all_polygons:
                if poly.class_name in skip_labels or len(poly.points) < 3:
                    continue
                m = extruder.extrude_polygon(poly.points, label=poly.class_name)
                if m.vertex_count > 0:
                    meshes.append(m)

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
                                       floorplan_image_path=floorplan_image_path,
                                       render_image_path=render_image_path)

        print(f"\nPhase 4 complete. Files saved to: {out_dir}/")
        for fmt, path in export_paths.items():
            print(f"  {fmt.upper()}: {path}")

        return Model3D(building=building, export_paths=export_paths)
