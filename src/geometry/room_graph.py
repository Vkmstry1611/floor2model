"""
room_graph.py
-------------
Builds a room connectivity graph from vectorized floor plan polygons.

The graph encodes:
  - Nodes: rooms (with metadata — type, area, centroid)
  - Edges: connections between rooms via doors or shared walls

This structured representation is the bridge between 2D geometry
and the 3D reconstruction in Phase 4.

Usage:
    from src.geometry.room_graph import RoomGraphBuilder

    builder = RoomGraphBuilder()
    graph = builder.build(vectorization_result, scale_estimate)
    print(graph.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
import cv2


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RoomNode:
    """A single room in the floor plan graph."""
    node_id:    int
    class_name: str
    class_id:   int
    centroid:   tuple[float, float]     # (x, y) in pixels
    area_px:    float
    area_m2:    float                   # real-world area
    bbox:       tuple[int, int, int, int]  # (x, y, w, h) pixels
    polygon:    list[tuple[int, int]]   # pixel boundary points

    @property
    def label(self) -> str:
        return f"{self.class_name} ({self.area_m2:.1f}m²)"


@dataclass
class RoomEdge:
    """A connection between two rooms."""
    node_a:      int           # node_id of room A
    node_b:      int           # node_id of room B
    edge_type:   str           # 'door', 'opening', 'shared_wall'
    via_element: Optional[int] = None   # index of door/window polygon if any
    distance_px: float = 0.0


@dataclass
class FloorPlanGraph:
    """Complete room connectivity graph for one floor plan."""
    nodes:  list[RoomNode] = field(default_factory=list)
    edges:  list[RoomEdge] = field(default_factory=list)
    scale_method: str = "unknown"

    @property
    def summary(self) -> dict:
        return {
            "rooms":       len(self.nodes),
            "connections": len(self.edges),
            "room_types":  [n.class_name for n in self.nodes],
            "total_area_m2": round(sum(n.area_m2 for n in self.nodes), 2),
            "scale_method": self.scale_method,
        }

    def get_node(self, node_id: int) -> Optional[RoomNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_neighbours(self, node_id: int) -> list[RoomNode]:
        neighbour_ids = set()
        for edge in self.edges:
            if edge.node_a == node_id:
                neighbour_ids.add(edge.node_b)
            elif edge.node_b == node_id:
                neighbour_ids.add(edge.node_a)
        return [self.get_node(nid) for nid in neighbour_ids if self.get_node(nid)]

    def to_dict(self) -> dict:
        """Serialisable representation for saving/loading."""
        return {
            "nodes": [
                {
                    "id":         n.node_id,
                    "class_name": n.class_name,
                    "class_id":   n.class_id,
                    "centroid":   list(n.centroid),
                    "area_px":    n.area_px,
                    "area_m2":    n.area_m2,
                    "bbox":       list(n.bbox),
                    "polygon":    n.polygon,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "node_a":    e.node_a,
                    "node_b":    e.node_b,
                    "edge_type": e.edge_type,
                    "distance":  e.distance_px,
                }
                for e in self.edges
            ],
            "summary": self.summary,
        }


# ── Graph builder ─────────────────────────────────────────────────────────────

class RoomGraphBuilder:
    """
    Builds a room connectivity graph from Phase 3 vectorization output.

    Connection strategy:
      1. Door proximity — if a door polygon centroid is between two room
         centroids, connect those rooms with a 'door' edge.
      2. Centroid proximity — rooms whose centroids are within
         proximity_threshold pixels are connected with a 'shared_wall' edge.

    Args:
        proximity_threshold: Max distance between room centroids to infer
                             adjacency (pixels). Scaled with image size.
        door_proximity:      Max distance from a door to a room centroid
                             to consider the door as connecting that room.
    """

    def __init__(
        self,
        proximity_threshold: int = 200,
        door_proximity: int = 150,
    ):
        self.proximity_threshold = proximity_threshold
        self.door_proximity = door_proximity

    def build(
        self,
        vectorization_result,
        scale_estimate=None,
    ) -> FloorPlanGraph:
        """
        Build the room graph from a VectorizationResult.

        Args:
            vectorization_result: Output from WallVectorizer.extract()
            scale_estimate:       Output from ScaleEstimator.estimate()

        Returns:
            FloorPlanGraph with nodes and edges.
        """
        graph = FloorPlanGraph(
            scale_method=scale_estimate.method if scale_estimate else "none"
        )

        # ── Step 1: Create room nodes ──────────────────────────────────────
        for i, room in enumerate(vectorization_result.rooms):
            area_m2 = 0.0
            if scale_estimate:
                area_m2 = scale_estimate.area_px_to_m2(room.area)

            cx, cy = room.centroid
            node = RoomNode(
                node_id=i,
                class_name=room.class_name,
                class_id=room.class_id,
                centroid=(cx, cy),
                area_px=room.area,
                area_m2=round(area_m2, 2),
                bbox=room.bbox,
                polygon=room.points,
            )
            graph.nodes.append(node)

        if not graph.nodes:
            return graph

        # ── Step 2: Connect rooms via doors ───────────────────────────────
        for door_idx, door in enumerate(vectorization_result.doors):
            door_cx, door_cy = door.centroid
            nearby_rooms = []

            for node in graph.nodes:
                dist = _euclidean(door_cx, door_cy, *node.centroid)
                if dist <= self.door_proximity:
                    nearby_rooms.append((dist, node.node_id))

            nearby_rooms.sort()
            if len(nearby_rooms) >= 2:
                _, id_a = nearby_rooms[0]
                _, id_b = nearby_rooms[1]
                if id_a != id_b and not _edge_exists(graph, id_a, id_b):
                    graph.edges.append(RoomEdge(
                        node_a=id_a,
                        node_b=id_b,
                        edge_type="door",
                        via_element=door_idx,
                        distance_px=nearby_rooms[1][0],
                    ))

        # ── Step 3: Connect adjacent rooms by centroid proximity ──────────
        n = len(graph.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                node_a = graph.nodes[i]
                node_b = graph.nodes[j]
                dist = _euclidean(*node_a.centroid, *node_b.centroid)

                if dist <= self.proximity_threshold:
                    if not _edge_exists(graph, node_a.node_id, node_b.node_id):
                        graph.edges.append(RoomEdge(
                            node_a=node_a.node_id,
                            node_b=node_b.node_id,
                            edge_type="shared_wall",
                            distance_px=dist,
                        ))

        return graph

    def draw(
        self,
        image: np.ndarray,
        graph: FloorPlanGraph,
    ) -> np.ndarray:
        """
        Visualise the room graph overlaid on the floor plan image.

        Nodes drawn as circles with room labels.
        Edges drawn as lines between centroids, coloured by type.
        """
        if len(image.shape) == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        edge_colors = {
            "door":        (50,  200, 200),
            "shared_wall": (150, 150, 150),
            "opening":     (200, 180,  50),
        }

        # Draw edges first (behind nodes)
        for edge in graph.edges:
            node_a = graph.get_node(edge.node_a)
            node_b = graph.get_node(edge.node_b)
            if node_a and node_b:
                color = edge_colors.get(edge.edge_type, (200, 200, 200))
                pt_a = (int(node_a.centroid[0]), int(node_a.centroid[1]))
                pt_b = (int(node_b.centroid[0]), int(node_b.centroid[1]))
                cv2.line(canvas, pt_a, pt_b, color, 2)

                # Label edge type at midpoint
                mid = ((pt_a[0] + pt_b[0]) // 2, (pt_a[1] + pt_b[1]) // 2)
                cv2.putText(canvas, edge.edge_type, mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Draw nodes
        for node in graph.nodes:
            cx, cy = int(node.centroid[0]), int(node.centroid[1])
            cv2.circle(canvas, (cx, cy), 12, (50, 50, 200), -1)
            cv2.circle(canvas, (cx, cy), 12, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                f"{node.class_name}",
                (cx - 30, cy - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
            cv2.putText(
                canvas,
                f"{node.area_m2:.1f}m²",
                (cx - 20, cy + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 200), 1,
            )

        return canvas


# ── Helpers ───────────────────────────────────────────────────────────────────

def _euclidean(x1, y1, x2, y2) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def _edge_exists(graph: FloorPlanGraph, id_a: int, id_b: int) -> bool:
    for edge in graph.edges:
        if (edge.node_a == id_a and edge.node_b == id_b) or \
           (edge.node_a == id_b and edge.node_b == id_a):
            return True
    return False
