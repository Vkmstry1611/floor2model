"""
pipeline.py
-----------
Orchestrates the complete Phase 3 geometry reconstruction pipeline:

  segmentation_result → vectorize → estimate_scale → build_graph → save

Usage:
    from src.geometry.pipeline import GeometryPipeline

    pipeline = GeometryPipeline()
    result = pipeline.run(segmentation_result, image)
    result.save("outputs/geometry/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .wall_vectorizer import WallVectorizer, VectorizationResult
from .scale_estimator import ScaleEstimator, ScaleEstimate
from .room_graph import RoomGraphBuilder, FloorPlanGraph

# Import refinement module (optional dependency)
try:
    from src.detection.refinement import RefinementConfig, refine_detections
    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False
    RefinementConfig = None


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GeometryConfig:
    """Tunable parameters for Phase 3 geometry pipeline."""
    epsilon_factor:      float = 0.008   # Polygon simplification
    min_area:            int   = 200     # Min polygon area (px²)
    morph_kernel:        int   = 3       # Morphological cleanup kernel
    simplify_walls:      bool  = True    # Extra wall simplification
    proximity_threshold: int   = 200     # Room adjacency threshold (px)
    door_proximity:      int   = 150     # Door-to-room proximity (px)
    target_image_size:   int   = 1024    # Phase 1 resize target
    
    # Detection refinement parameters
    use_refinement:      bool  = True    # Enable detection refinement
    refinement_config:   Optional['RefinementConfig'] = None  # Custom refinement config
    
    # Visualization parameters
    debug_visualization: bool  = False   # Enable debug visualizations
    visualization_dir:   str   = "outputs/debug_viz"  # Directory for debug images
    
    def __post_init__(self):
        """Initialize default refinement config if not provided."""
        if self.use_refinement and self.refinement_config is None and REFINEMENT_AVAILABLE:
            from src.detection.refinement import RefinementConfig
            self.refinement_config = RefinementConfig()


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class GeometryResult:
    """Complete Phase 3 output for one floor plan."""
    source_path:      str
    vectorization:    VectorizationResult
    scale:            ScaleEstimate
    graph:            FloorPlanGraph
    config:           GeometryConfig = field(repr=False)

    def save(self, output_dir: str, prefix: str = "") -> dict[str, str]:
        """
        Save all Phase 3 outputs.

        Returns:
            Dict mapping output type → file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem = Path(self.source_path).stem
        p = f"{prefix}{stem}" if prefix else stem

        paths = {
            "graph_json": str(out / f"{p}_graph.json"),
            "scale_json": str(out / f"{p}_scale.json"),
        }

        # Save room graph as JSON
        graph_data = self.graph.to_dict()
        with open(paths["graph_json"], "w") as f:
            json.dump(graph_data, f, indent=2)

        # Save scale estimate
        scale_data = {
            "pixels_per_metre": self.scale.pixels_per_metre,
            "metres_per_pixel": self.scale.metres_per_pixel,
            "confidence":       self.scale.confidence,
            "method":           self.scale.method,
            "notes":            self.scale.notes,
        }
        with open(paths["scale_json"], "w") as f:
            json.dump(scale_data, f, indent=2)

        # Print summary
        print(f"\nGeometry reconstruction complete: {self.source_path}")
        print(f"  Scale:  {self.scale.pixels_per_metre:.1f} px/m "
              f"(method: {self.scale.method}, "
              f"confidence: {self.scale.confidence:.0%})")
        print(f"  Rooms:  {len(self.graph.nodes)}")
        print(f"  Connections: {len(self.graph.edges)}")
        print(f"  Walls:  {len(self.vectorization.walls)} polygons")
        print(f"  Doors:  {len(self.vectorization.doors)} polygons")
        print(f"  Files saved to: {output_dir}/")

        return paths


# ── Pipeline ──────────────────────────────────────────────────────────────────

class GeometryPipeline:
    """
    Phase 3: Geometry reconstruction pipeline.

    Takes Phase 2 segmentation output and produces:
    - Vectorized wall/room/door/window polygons
    - Pixel-to-metre scale estimate
    - Room connectivity graph

    This output feeds directly into Phase 4 (3D extrusion).

    Example:
        from src.segmentation.predictor import FloorPlanPredictor
        from src.geometry.pipeline import GeometryPipeline

        predictor = FloorPlanPredictor("models/segmentation/best.pt")
        seg_result = predictor.predict("outputs/plan_4_cleaned.png")

        geo_pipeline = GeometryPipeline()
        geo_result = geo_pipeline.run(seg_result, image)
        geo_result.save("outputs/geometry/")
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        self.config = config or GeometryConfig()
        cfg = self.config

        self.vectorizer = WallVectorizer(
            epsilon_factor=cfg.epsilon_factor,
            min_area=cfg.min_area,
            morph_kernel=cfg.morph_kernel,
            simplify_walls=cfg.simplify_walls,
        )
        self.scale_estimator = ScaleEstimator(
            target_image_size=cfg.target_image_size,
        )
        self.graph_builder = RoomGraphBuilder(
            proximity_threshold=cfg.proximity_threshold,
            door_proximity=cfg.door_proximity,
        )

    def run(
        self,
        segmentation_result,
        image: np.ndarray,
        image_path: str = "unknown",
    ) -> GeometryResult:
        """
        Run the full Phase 3 pipeline.

        Args:
            segmentation_result: FloorPlanPredictor output (Phase 2).
            image:               The original floor plan image (numpy array).
            image_path:          Source path for naming output files.

        Returns:
            GeometryResult with all Phase 3 outputs.
        """
        # ── Step 1: Vectorize masks → polygons ────────────────────────────
        print("[1/3] Vectorizing segmentation masks...")
        vec_result = self.vectorizer.extract(
            segmentation_result,
            image_shape=image.shape[:2],
        )
        print(f"  Walls:   {len(vec_result.walls)}")
        print(f"  Rooms:   {len(vec_result.rooms)}")
        print(f"  Doors:   {len(vec_result.doors)}")
        print(f"  Windows: {len(vec_result.windows)}")
        
        # Store original for comparison if debug visualization enabled
        vec_result_before = None
        if self.config.debug_visualization:
            vec_result_before = vec_result
        
        # ── NEW: Apply detection refinement if enabled ────────────────────
        if self.config.use_refinement and REFINEMENT_AVAILABLE:
            print("[1.5/3] Applying detection refinement...")
            try:
                vec_result = refine_detections(
                    yolo_output=segmentation_result,
                    image=image,
                    geometry_output=vec_result,
                    config=self.config.refinement_config
                )
                print(f"  Refined Doors:   {len(vec_result.doors)}")
                print(f"  Refined Windows: {len(vec_result.windows)}")
                
                # Debug visualization if enabled
                if self.config.debug_visualization:
                    self._create_debug_visualizations(
                        image, vec_result_before, vec_result, image_path
                    )
                
            except Exception as e:
                print(f"  WARNING: Refinement failed: {e}")
                print(f"  Continuing with original YOLO detections")
        elif self.config.use_refinement and not REFINEMENT_AVAILABLE:
            print("  WARNING: Refinement requested but module not available")

        # ── Step 2: Estimate scale ─────────────────────────────────────────
        print("[2/3] Estimating scale...")
        scale = self.scale_estimator.estimate(image, vec_result)
        print(f"  {scale.pixels_per_metre:.1f} px/m "
              f"(method: {scale.method}, "
              f"confidence: {scale.confidence:.0%})")

        # ── Step 3: Build room graph ───────────────────────────────────────
        print("[3/3] Building room connectivity graph...")
        graph = self.graph_builder.build(vec_result, scale)
        print(f"  {len(graph.nodes)} rooms, {len(graph.edges)} connections")
        for node in graph.nodes:
            print(f"  → {node.class_name}: {node.area_m2:.1f} m²")

        return GeometryResult(
            source_path=image_path,
            vectorization=vec_result,
            scale=scale,
            graph=graph,
            config=self.config,
        )
    
    def _create_debug_visualizations(
        self,
        image: np.ndarray,
        vec_before: VectorizationResult,
        vec_after: VectorizationResult,
        image_path: str
    ):
        """
        Create debug visualizations for detection refinement.
        
        Args:
            image: Original image
            vec_before: VectorizationResult before refinement
            vec_after: VectorizationResult after refinement
            image_path: Source image path for naming
        """
        try:
            from src.utils.visualization import (
                visualize_vectorization_result,
                visualize_comparison
            )
            from pathlib import Path
            
            # Create output directory
            out_dir = Path(self.config.visualization_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Get filename stem
            stem = Path(image_path).stem
            
            # Create before visualization
            before_path = str(out_dir / f"{stem}_before_refinement.png")
            visualize_vectorization_result(image, vec_before, before_path, show_labels=False)
            
            # Create after visualization
            after_path = str(out_dir / f"{stem}_after_refinement.png")
            visualize_vectorization_result(image, vec_after, after_path, show_labels=True)
            
            # Create comparison
            comparison_path = str(out_dir / f"{stem}_comparison.png")
            detections_before = {
                "walls": [np.array(w.points) for w in vec_before.walls],
                "rooms": [np.array(r.points) for r in vec_before.rooms],
                "doors": [np.array(d.points) for d in vec_before.doors],
                "windows": [np.array(w.points) for w in vec_before.windows]
            }
            detections_after = {
                "walls": [np.array(w.points) for w in vec_after.walls],
                "rooms": [np.array(r.points) for r in vec_after.rooms],
                "doors": [np.array(d.points) for d in vec_after.doors],
                "windows": [np.array(w.points) for w in vec_after.windows]
            }
            
            from src.utils.visualization import visualize_comparison
            visualize_comparison(image, detections_before, detections_after, comparison_path)
            
            print(f"  Debug visualizations saved to: {out_dir}/")
            
        except Exception as e:
            print(f"  WARNING: Failed to create debug visualizations: {e}")

    def run_and_visualize(
        self,
        segmentation_result,
        image: np.ndarray,
        image_path: str = "unknown",
        output_dir: str = "outputs/geometry",
    ) -> tuple[GeometryResult, np.ndarray, np.ndarray]:
        """
        Run pipeline and generate visualization images.

        Returns:
            (GeometryResult, polygon_viz_image, graph_viz_image)
        """
        result = self.run(segmentation_result, image, image_path)

        # Visualization 1: vectorized polygons
        poly_viz = self.vectorizer.draw(image, result.vectorization)

        # Visualization 2: room graph
        graph_viz = self.graph_builder.draw(image, result.graph)

        # Save visualizations
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        cv2.imwrite(str(out / f"{stem}_polygons.png"), poly_viz)
        cv2.imwrite(str(out / f"{stem}_graph.png"), graph_viz)

        return result, poly_viz, graph_viz

    def run_batch(
        self,
        items: list[dict],
        output_dir: str,
    ) -> list[GeometryResult]:
        """
        Run geometry pipeline on multiple floor plans.

        Args:
            items: List of dicts with keys 'segmentation_result',
                   'image', 'image_path'.
            output_dir: Where to save outputs.

        Returns:
            List of GeometryResult objects.
        """
        results = []
        for i, item in enumerate(items, 1):
            print(f"\n── [{i}/{len(items)}] {item.get('image_path', '?')} ──")
            try:
                result = self.run(
                    item["segmentation_result"],
                    item["image"],
                    item.get("image_path", f"item_{i}"),
                )
                result.save(output_dir)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
        return results
