"""
Detection refinement module for improving door and window detection.

This module provides geometry-based refinement of YOLO segmentation results,
using wall structure, spatial relationships, and computer vision techniques.

Example usage:
    >>> from src.detection.refinement import refine_detections, RefinementConfig
    >>> 
    >>> # Use default configuration
    >>> refined_result = refine_detections(yolo_output, image, geometry_output)
    >>> 
    >>> # Use custom configuration
    >>> config = RefinementConfig(
    ...     canny_low_threshold=40,
    ...     canny_high_threshold=160,
    ...     door_width=100
    ... )
    >>> refined_result = refine_detections(
    ...     yolo_output, image, geometry_output, config=config
    ... )
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

import numpy as np
import cv2

from src.segmentation.predictor import SegmentationResult
from src.geometry.wall_vectorizer import WallPolygon, VectorizationResult


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RefinementConfig:
    """Configuration for detection refinement algorithms."""
    
    # Edge detection parameters (Canny)
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    
    # Line detection parameters (HoughLinesP)
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 10
    hough_threshold: int = 50
    
    # Window detection parameters
    parallel_line_proximity: int = 20  # max distance between parallel lines (px)
    parallel_angle_tolerance: float = 15.0  # degrees
    window_min_length: int = 30  # minimum window length (px)
    
    # Door detection parameters
    wall_gap_threshold: int = 20  # minimum gap width to consider as door (px)
    room_adjacency_threshold: int = 50  # max distance for rooms to be adjacent (px)
    door_width: int = 90  # standard door width in pixels (~0.9m)
    
    # Spatial analysis parameters
    wall_thickness_tolerance: int = 10  # tolerance for point-on-wall checks (px)
    
    # Performance parameters
    enable_door_detection: bool = True
    enable_window_detection: bool = True
    
    def __post_init__(self):
        """Validate and clamp configuration parameters to reasonable ranges."""
        # Validate Canny thresholds
        self.canny_low_threshold = max(1, min(255, self.canny_low_threshold))
        self.canny_high_threshold = max(1, min(255, self.canny_high_threshold))
        
        if self.canny_high_threshold <= self.canny_low_threshold:
            logger.warning(
                f"Invalid Canny thresholds: low={self.canny_low_threshold}, "
                f"high={self.canny_high_threshold}. Using defaults."
            )
            self.canny_low_threshold = 50
            self.canny_high_threshold = 150
        
        # Validate positive integer parameters
        self.hough_min_line_length = max(1, self.hough_min_line_length)
        self.hough_max_line_gap = max(0, self.hough_max_line_gap)
        self.hough_threshold = max(1, self.hough_threshold)
        self.parallel_line_proximity = max(1, self.parallel_line_proximity)
        self.window_min_length = max(1, self.window_min_length)
        self.wall_gap_threshold = max(1, self.wall_gap_threshold)
        self.room_adjacency_threshold = max(1, self.room_adjacency_threshold)
        self.door_width = max(1, self.door_width)
        self.wall_thickness_tolerance = max(1, self.wall_thickness_tolerance)
        
        # Validate angle tolerance
        self.parallel_angle_tolerance = max(0.0, min(90.0, self.parallel_angle_tolerance))


# Type aliases for clarity
LineSegment = Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1, y1), (x2, y2))


class DetectionRefiner:
    """Main orchestrator for detection refinement."""
    
    def __init__(self, config: Optional[RefinementConfig] = None):
        """
        Initialize the detection refiner.
        
        Parameters
        ----------
        config : RefinementConfig, optional
            Configuration for refinement algorithms. If None, uses defaults.
        """
        self.config = config or RefinementConfig()
        self.door_detector = DoorDetector(self.config)
        self.window_detector = WindowDetector(self.config)
    
    def refine(
        self,
        yolo_output: SegmentationResult,
        image: np.ndarray,
        geometry_output: VectorizationResult
    ) -> VectorizationResult:
        """
        Main refinement function that replaces YOLO doors/windows
        with geometry-based detections.
        
        Parameters
        ----------
        yolo_output : SegmentationResult
            Original YOLO segmentation result
        image : np.ndarray
            Preprocessed floor plan image (grayscale)
        geometry_output : VectorizationResult
            Vectorized geometry from Phase 3
        
        Returns
        -------
        VectorizationResult
            Refined VectorizationResult with improved doors/windows
        """
        logger.info("Starting detection refinement")
        
        refined_doors = []
        refined_windows = []
        
        # Try door detection
        if self.config.enable_door_detection:
            try:
                refined_doors = self.door_detector.detect(
                    geometry_output.walls,
                    geometry_output.rooms
                )
                logger.info(f"Door detection complete: {len(refined_doors)} doors detected")
            except Exception as e:
                logger.error(f"Door detection failed: {e}", exc_info=True)
                logger.warning("Falling back to original YOLO door detections")
                refined_doors = geometry_output.doors
        else:
            refined_doors = geometry_output.doors
        
        # Try window detection independently
        if self.config.enable_window_detection:
            try:
                refined_windows = self.window_detector.detect(
                    image,
                    geometry_output.walls
                )
                logger.info(f"Window detection complete: {len(refined_windows)} windows detected")
            except Exception as e:
                logger.error(f"Window detection failed: {e}", exc_info=True)
                logger.warning("Falling back to original YOLO window detections")
                refined_windows = geometry_output.windows
        else:
            refined_windows = geometry_output.windows
        
        # Merge results
        refined_result = VectorizationResult(
            walls=geometry_output.walls,
            rooms=geometry_output.rooms,
            doors=refined_doors,
            windows=refined_windows,
            other=geometry_output.other,
            image_shape=geometry_output.image_shape
        )
        
        logger.info(
            f"Refinement complete: {len(refined_doors)} doors, "
            f"{len(refined_windows)} windows"
        )
        
        return refined_result


def refine_detections(
    yolo_output: SegmentationResult,
    image: np.ndarray,
    geometry_output: VectorizationResult,
    config: Optional[RefinementConfig] = None
) -> VectorizationResult:
    """
    Public API function for detection refinement.
    
    This is the main entry point called by GeometryPipeline.
    
    Parameters
    ----------
    yolo_output : SegmentationResult
        Original YOLO segmentation result
    image : np.ndarray
        Preprocessed floor plan image (grayscale)
    geometry_output : VectorizationResult
        Vectorized geometry from Phase 3
    config : RefinementConfig, optional
        Optional custom configuration. If None, uses defaults.
    
    Returns
    -------
    VectorizationResult
        Refined VectorizationResult with improved doors/windows
    
    Examples
    --------
    >>> refined = refine_detections(yolo_output, image, geometry_output)
    >>> print(f"Detected {len(refined.doors)} doors")
    """
    try:
        refiner = DetectionRefiner(config)
        return refiner.refine(yolo_output, image, geometry_output)
    except Exception as e:
        logger.error(f"Refinement failed completely: {e}", exc_info=True)
        logger.warning("Returning original geometry output unchanged")
        return geometry_output


class SpatialAnalyzer:
    """Analyzes spatial relationships between rooms and walls."""
    
    def __init__(self, config: RefinementConfig):
        """
        Initialize spatial analyzer.
        
        Parameters
        ----------
        config : RefinementConfig
            Configuration for spatial analysis parameters
        """
        self.config = config
    
    def find_adjacent_rooms(
        self,
        room_polygons: List[WallPolygon]
    ) -> List[Tuple[WallPolygon, WallPolygon]]:
        """
        Find pairs of rooms that share a wall boundary.
        
        Uses polygon proximity analysis: rooms are adjacent if their
        boundaries come within room_adjacency_threshold pixels.
        
        Parameters
        ----------
        room_polygons : List[WallPolygon]
            List of room polygons
        
        Returns
        -------
        List[Tuple[WallPolygon, WallPolygon]]
            List of (room_a, room_b) tuples representing adjacent room pairs
        """
        if not room_polygons or len(room_polygons) < 2:
            return []
        
        adjacent_pairs = []
        n = len(room_polygons)
        
        for i in range(n):
            for j in range(i + 1, n):
                room_a = room_polygons[i]
                room_b = room_polygons[j]
                
                # Compute distance between room centroids
                centroid_a = room_a.centroid
                centroid_b = room_b.centroid
                distance = np.sqrt(
                    (centroid_a[0] - centroid_b[0])**2 +
                    (centroid_a[1] - centroid_b[1])**2
                )
                
                # Check if rooms are close enough to be adjacent
                if distance <= self.config.room_adjacency_threshold * 3:  # Use 3x threshold for centroid distance
                    # Verify actual boundary proximity using minimum distance between polygon points
                    points_a = np.array(room_a.points, dtype=np.float32)
                    points_b = np.array(room_b.points, dtype=np.float32)
                    
                    # Compute minimum distance between any two points
                    min_dist = float('inf')
                    for pt_a in points_a:
                        for pt_b in points_b:
                            dist = np.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)
                            min_dist = min(min_dist, dist)
                    
                    if min_dist <= self.config.room_adjacency_threshold:
                        adjacent_pairs.append((room_a, room_b))
        
        logger.debug(f"Found {len(adjacent_pairs)} adjacent room pairs")
        return adjacent_pairs
    
    def find_shared_wall_segment(
        self,
        room_a: WallPolygon,
        room_b: WallPolygon,
        wall_polygons: List[WallPolygon]
    ) -> Optional[np.ndarray]:
        """
        Extract the wall segment between two adjacent rooms.
        
        Algorithm:
        1. Find the line connecting room centroids
        2. Find wall polygons that intersect this line
        3. Extract the portion of wall between the two rooms
        
        Parameters
        ----------
        room_a : WallPolygon
            First room polygon
        room_b : WallPolygon
            Second room polygon
        wall_polygons : List[WallPolygon]
            List of wall polygons
        
        Returns
        -------
        np.ndarray or None
            Nx2 numpy array of wall segment points, or None if not found
        """
        if not wall_polygons:
            return None
        
        centroid_a = np.array(room_a.centroid, dtype=np.float32)
        centroid_b = np.array(room_b.centroid, dtype=np.float32)
        
        # Find walls that lie between the two rooms
        # A wall is between rooms if it's close to the line connecting centroids
        candidate_walls = []
        
        for wall in wall_polygons:
            if not wall.is_wall:
                continue
            
            wall_centroid = np.array(wall.centroid, dtype=np.float32)
            
            # Check if wall centroid is roughly between room centroids
            # using dot product to check if it's in the same direction
            vec_ab = centroid_b - centroid_a
            vec_aw = wall_centroid - centroid_a
            
            # Project wall centroid onto line AB
            if np.dot(vec_ab, vec_ab) > 0:
                t = np.dot(vec_aw, vec_ab) / np.dot(vec_ab, vec_ab)
                
                # Wall should be between rooms (0 < t < 1)
                if 0.2 < t < 0.8:  # Allow some tolerance
                    # Check perpendicular distance to line
                    projection = centroid_a + t * vec_ab
                    perp_dist = np.linalg.norm(wall_centroid - projection)
                    
                    if perp_dist < self.config.room_adjacency_threshold:
                        candidate_walls.append(wall)
        
        if not candidate_walls:
            logger.debug("No shared wall segment found between rooms")
            return None
        
        # Return the wall closest to the midpoint between rooms
        midpoint = (centroid_a + centroid_b) / 2
        closest_wall = min(
            candidate_walls,
            key=lambda w: np.linalg.norm(np.array(w.centroid) - midpoint)
        )
        
        return np.array(closest_wall.points, dtype=np.float32)
    
    def is_point_on_wall(
        self,
        point: Tuple[float, float],
        wall_polygon: WallPolygon
    ) -> bool:
        """
        Check if a point lies within wall polygon boundaries.
        
        Uses cv2.pointPolygonTest with tolerance from config.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point coordinates (x, y)
        wall_polygon : WallPolygon
            Wall polygon to test against
        
        Returns
        -------
        bool
            True if point is on or near the wall, False otherwise
        """
        wall_pts = np.array(wall_polygon.points, dtype=np.int32)
        
        # Use cv2.pointPolygonTest to compute signed distance
        # Positive = inside, 0 = on edge, negative = outside
        dist = cv2.pointPolygonTest(wall_pts, point, measureDist=True)
        
        # Point is on wall if distance is within tolerance
        return abs(dist) <= self.config.wall_thickness_tolerance
    
    def get_wall_direction(
        self,
        wall_segment: np.ndarray
    ) -> float:
        """
        Compute orientation angle of a wall segment.
        
        Uses PCA (principal component analysis) on wall points
        to find dominant direction.
        
        Parameters
        ----------
        wall_segment : np.ndarray
            Nx2 array of wall segment points
        
        Returns
        -------
        float
            Angle in degrees (0-180 range)
        """
        if len(wall_segment) < 2:
            logger.warning("Wall segment has fewer than 2 points, returning 0 degrees")
            return 0.0
        
        try:
            # Use PCA to find principal direction
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=1)
            pca.fit(wall_segment)
            
            # Get principal component (dominant direction)
            component = pca.components_[0]
            angle = np.arctan2(component[1], component[0])
            
            # Convert to degrees, normalize to [0, 180)
            angle_deg = np.degrees(angle) % 180
            
            return float(angle_deg)
        
        except Exception as e:
            logger.warning(f"Failed to compute wall direction using PCA: {e}")
            
            # Fallback: use simple line fitting
            try:
                # Fit line using first and last points
                p1 = wall_segment[0]
                p2 = wall_segment[-1]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                angle = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle) % 180
                
                return float(angle_deg)
            
            except Exception as e2:
                logger.error(f"Failed to compute wall direction: {e2}")
                return 0.0


class GapDetector:
    """Detects gaps and discontinuities in wall segments."""
    
    def __init__(self, config: RefinementConfig):
        """
        Initialize gap detector.
        
        Parameters
        ----------
        config : RefinementConfig
            Configuration for gap detection parameters
        """
        self.config = config
    
    def detect_wall_gaps(
        self,
        wall_segment: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Identify breaks or discontinuities in a wall segment.
        
        Algorithm:
        1. Sort wall points along the wall direction
        2. Compute distances between consecutive points
        3. Identify gaps where distance > wall_gap_threshold
        
        Parameters
        ----------
        wall_segment : np.ndarray
            Nx2 array of wall segment points
        
        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (start_point, end_point) tuples indicating gap locations
        """
        if len(wall_segment) < 2:
            logger.debug("Wall segment too short for gap detection")
            return []
        
        try:
            # Compute wall direction to sort points along it
            centroid = np.mean(wall_segment, axis=0)
            
            # Use PCA to find principal direction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(wall_segment)
            direction = pca.components_[0]
            
            # Project points onto principal direction
            projections = np.dot(wall_segment - centroid, direction)
            
            # Sort points by projection
            sorted_indices = np.argsort(projections)
            sorted_points = wall_segment[sorted_indices]
            
        except Exception as e:
            logger.warning(f"Failed to sort wall points using PCA: {e}, using simple sort")
            # Fallback: sort by x-coordinate
            sorted_indices = np.argsort(wall_segment[:, 0])
            sorted_points = wall_segment[sorted_indices]
        
        # Compute distances between consecutive points
        gaps = []
        for i in range(len(sorted_points) - 1):
            p1 = sorted_points[i]
            p2 = sorted_points[i + 1]
            
            distance = np.linalg.norm(p2 - p1)
            
            # If distance exceeds threshold, we found a gap
            if distance > self.config.wall_gap_threshold:
                gaps.append((p1, p2))
                logger.debug(f"Found gap of width {distance:.1f}px at {p1} -> {p2}")
        
        if not gaps:
            logger.debug("No wall gaps detected")
        else:
            logger.debug(f"Detected {len(gaps)} wall gaps")
        
        return gaps


# Placeholder classes - will be implemented in subsequent tasks
class DoorDetector:
    """Detects doors based on wall gaps and room adjacency."""
    
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.spatial_analyzer = SpatialAnalyzer(config)
        self.gap_detector = GapDetector(config)
    
    def detect(
        self,
        wall_polygons: List[WallPolygon],
        room_polygons: List[WallPolygon]
    ) -> List[WallPolygon]:
        """
        Detect door locations from wall geometry and room adjacency.
        
        Parameters
        ----------
        wall_polygons : List[WallPolygon]
            List of vectorized wall polygons
        room_polygons : List[WallPolygon]
            List of vectorized room polygons
        
        Returns
        -------
        List[WallPolygon]
            List of door polygons with class_id=3, class_name="Door"
        """
        # Input validation
        if not wall_polygons:
            logger.warning("No wall polygons provided, skipping door detection")
            return []
        
        if not room_polygons:
            logger.warning("No room polygons provided, skipping door detection")
            return []
        
        logger.info(f"Detecting doors from {len(wall_polygons)} walls and {len(room_polygons)} rooms")
        
        doors = []
        
        # Find all pairs of adjacent rooms
        adjacent_pairs = self.spatial_analyzer.find_adjacent_rooms(room_polygons)
        
        if not adjacent_pairs:
            logger.info("No adjacent rooms found, no doors to place")
            return []
        
        logger.info(f"Found {len(adjacent_pairs)} adjacent room pairs")
        
        # For each adjacent room pair, place a door
        for room_a, room_b in adjacent_pairs:
            try:
                # Find the shared wall segment
                shared_wall = self.spatial_analyzer.find_shared_wall_segment(
                    room_a, room_b, wall_polygons
                )
                
                if shared_wall is None:
                    logger.debug(f"No shared wall found between rooms, skipping")
                    continue
                
                # Detect gaps in the wall
                gaps = self.gap_detector.detect_wall_gaps(shared_wall)
                
                # Determine door position
                if gaps:
                    # Place door at first gap
                    gap_start, gap_end = gaps[0]
                    door_position = (gap_start + gap_end) / 2
                    logger.debug(f"Placing door at gap location: {door_position}")
                else:
                    # Fallback: place door at midpoint of shared wall
                    door_position = np.mean(shared_wall, axis=0)
                    logger.debug(f"No gaps found, placing door at wall midpoint: {door_position}")
                
                # Get wall direction
                wall_direction = self.spatial_analyzer.get_wall_direction(shared_wall)
                
                # Create door polygon
                door = create_door_polygon(
                    position=tuple(door_position),
                    wall_direction=wall_direction,
                    door_width=self.config.door_width
                )
                
                doors.append(door)
            
            except Exception as e:
                logger.warning(f"Failed to place door between rooms: {e}")
                continue
        
        logger.info(f"Successfully detected {len(doors)} doors")
        return doors


def create_door_polygon(
    position: Tuple[float, float],
    wall_direction: float,
    door_width: int
) -> WallPolygon:
    """
    Create a door polygon at the specified position.
    
    The door is oriented perpendicular to the wall direction.
    
    Parameters
    ----------
    position : Tuple[float, float]
        (x, y) center point for door
    wall_direction : float
        Wall orientation in degrees (0-180)
    door_width : int
        Door width in pixels
    
    Returns
    -------
    WallPolygon
        WallPolygon with class_id=3, class_name="Door"
    """
    # Door is perpendicular to wall
    door_angle = (wall_direction + 90) % 180
    door_angle_rad = np.radians(door_angle)
    
    # Create door as a rectangle perpendicular to wall
    half_width = door_width / 2
    half_depth = door_width / 4  # Door depth is half the width
    
    # Direction vectors
    dir_along = np.array([np.cos(door_angle_rad), np.sin(door_angle_rad)])
    dir_perp = np.array([-np.sin(door_angle_rad), np.cos(door_angle_rad)])
    
    # Four corners of the door rectangle
    center = np.array(position)
    p1 = center - half_width * dir_along - half_depth * dir_perp
    p2 = center + half_width * dir_along - half_depth * dir_perp
    p3 = center + half_width * dir_along + half_depth * dir_perp
    p4 = center - half_width * dir_along + half_depth * dir_perp
    
    # Convert to integer coordinates
    points = [
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        (int(p3[0]), int(p3[1])),
        (int(p4[0]), int(p4[1]))
    ]
    
    # Compute area and bbox
    points_array = np.array(points)
    area = float(cv2.contourArea(points_array))
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    bbox = (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max() - x_coords.min()),
        int(y_coords.max() - y_coords.min())
    )
    
    return WallPolygon(
        class_id=3,
        class_name="Door",
        points=points,
        area=area,
        bbox=bbox,
        confidence=1.0
    )


class EdgeAnalyzer:
    """Performs edge detection on floor plan images."""
    
    def __init__(self, config: RefinementConfig):
        """
        Initialize edge analyzer.
        
        Parameters
        ----------
        config : RefinementConfig
            Configuration for edge detection parameters
        """
        self.config = config
    
    def detect_edges(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply Canny edge detection to identify edges.
        
        Uses config.canny_low_threshold and config.canny_high_threshold.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale)
        
        Returns
        -------
        np.ndarray
            Binary edge map (same size as input)
        """
        if image is None or image.size == 0:
            logger.error("Invalid image provided to edge detection")
            return np.zeros((100, 100), dtype=np.uint8)
        
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                image,
                self.config.canny_low_threshold,
                self.config.canny_high_threshold
            )
            
            logger.debug(f"Edge detection complete: {np.count_nonzero(edges)} edge pixels")
            return edges
        
        except cv2.error as e:
            logger.error(f"OpenCV edge detection failed: {e}")
            return np.zeros_like(image, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return np.zeros_like(image, dtype=np.uint8)


class LineDetector:
    """Detects and analyzes line segments in images."""
    
    def __init__(self, config: RefinementConfig):
        """
        Initialize line detector.
        
        Parameters
        ----------
        config : RefinementConfig
            Configuration for line detection parameters
        """
        self.config = config
    
    def detect_lines(
        self,
        edge_map: np.ndarray
    ) -> List[LineSegment]:
        """
        Extract line segments using HoughLinesP.
        
        Uses config.hough_* parameters.
        
        Parameters
        ----------
        edge_map : np.ndarray
            Binary edge map from edge detection
        
        Returns
        -------
        List[LineSegment]
            List of ((x1, y1), (x2, y2)) line segments
        """
        if edge_map is None or edge_map.size == 0:
            logger.warning("Invalid edge map provided to line detection")
            return []
        
        try:
            # Apply HoughLinesP
            lines = cv2.HoughLinesP(
                edge_map,
                rho=1,
                theta=np.pi / 180,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.hough_min_line_length,
                maxLineGap=self.config.hough_max_line_gap
            )
            
            if lines is None:
                logger.debug("No lines detected by HoughLinesP")
                return []
            
            # Convert to list of line segments
            line_segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_segments.append(((int(x1), int(y1)), (int(x2), int(y2))))
            
            logger.debug(f"Detected {len(line_segments)} line segments")
            return line_segments
        
        except cv2.error as e:
            logger.error(f"OpenCV line detection failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Line detection failed: {e}")
            return []
    
    def compute_line_angle(
        self,
        line: LineSegment
    ) -> float:
        """
        Compute orientation angle of a line segment.
        
        Parameters
        ----------
        line : LineSegment
            Line segment ((x1, y1), (x2, y2))
        
        Returns
        -------
        float
            Angle in degrees (0-180 range)
        """
        (x1, y1), (x2, y2) = line
        dx = x2 - x1
        dy = y2 - y1
        
        angle = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle) % 180
        
        return float(angle_deg)
    
    def compute_line_distance(
        self,
        line_a: LineSegment,
        line_b: LineSegment
    ) -> float:
        """
        Compute perpendicular distance between two parallel lines.
        
        Parameters
        ----------
        line_a : LineSegment
            First line segment
        line_b : LineSegment
            Second line segment
        
        Returns
        -------
        float
            Distance in pixels
        """
        # Use midpoints of lines
        (x1_a, y1_a), (x2_a, y2_a) = line_a
        (x1_b, y1_b), (x2_b, y2_b) = line_b
        
        mid_a = np.array([(x1_a + x2_a) / 2, (y1_a + y2_a) / 2])
        mid_b = np.array([(x1_b + x2_b) / 2, (y1_b + y2_b) / 2])
        
        # Compute distance between midpoints
        distance = np.linalg.norm(mid_b - mid_a)
        
        return float(distance)
    
    def find_parallel_pairs(
        self,
        lines: List[LineSegment]
    ) -> List[Tuple[int, int]]:
        """
        Identify pairs of parallel lines with close proximity.
        
        Algorithm:
        1. Compute orientation angle for each line
        2. For each line pair:
           - Check if angles are within parallel_angle_tolerance
           - Check if distance between lines < parallel_line_proximity
        3. Return indices of parallel pairs
        
        Parameters
        ----------
        lines : List[LineSegment]
            List of line segments
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (line_idx_a, line_idx_b) tuples
        """
        if not lines or len(lines) < 2:
            return []
        
        pairs = []
        n = len(lines)
        
        # Compute angles for all lines
        angles = [self.compute_line_angle(line) for line in lines]
        
        for i in range(n):
            for j in range(i + 1, n):
                angle_i = angles[i]
                angle_j = angles[j]
                
                # Check angle similarity (handle wraparound at 180 degrees)
                angle_diff = abs(angle_i - angle_j)
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                if angle_diff > self.config.parallel_angle_tolerance:
                    continue
                
                # Check distance
                distance = self.compute_line_distance(lines[i], lines[j])
                
                if distance > self.config.parallel_line_proximity:
                    continue
                
                # Check minimum line length
                (x1_i, y1_i), (x2_i, y2_i) = lines[i]
                (x1_j, y1_j), (x2_j, y2_j) = lines[j]
                
                len_i = np.sqrt((x2_i - x1_i)**2 + (y2_i - y1_i)**2)
                len_j = np.sqrt((x2_j - x1_j)**2 + (y2_j - y1_j)**2)
                
                if len_i < self.config.window_min_length or len_j < self.config.window_min_length:
                    continue
                
                pairs.append((i, j))
                logger.debug(
                    f"Found parallel pair: lines {i} and {j}, "
                    f"angle_diff={angle_diff:.1f}°, distance={distance:.1f}px"
                )
        
        logger.debug(f"Found {len(pairs)} parallel line pairs")
        return pairs


class WindowDetector:
    """Detects windows using edge detection and line analysis."""
    
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.edge_analyzer = EdgeAnalyzer(config)
        self.line_detector = LineDetector(config)
    
    def detect(
        self,
        image: np.ndarray,
        wall_polygons: List[WallPolygon]
    ) -> List[WallPolygon]:
        """
        Detect window locations using edge detection and line analysis.
        
        Parameters
        ----------
        image : np.ndarray
            Preprocessed floor plan image (grayscale)
        wall_polygons : List[WallPolygon]
            List of vectorized wall polygons
        
        Returns
        -------
        List[WallPolygon]
            List of window polygons with class_id=2, class_name="Window"
        """
        # Input validation
        if image is None or image.size == 0:
            logger.warning("Invalid image provided, skipping window detection")
            return []
        
        if not wall_polygons:
            logger.warning("No wall polygons provided, skipping window detection")
            return []
        
        logger.info(f"Detecting windows from image and {len(wall_polygons)} walls")
        
        # Step 1: Edge detection
        edges = self.edge_analyzer.detect_edges(image)
        
        # Step 2: Line detection
        lines = self.line_detector.detect_lines(edges)
        
        if not lines:
            logger.info("No lines detected, no windows to place")
            return []
        
        # Step 3: Find parallel line pairs
        parallel_pairs = self.line_detector.find_parallel_pairs(lines)
        
        if not parallel_pairs:
            logger.info("No parallel line pairs found, no windows to place")
            return []
        
        logger.info(f"Found {len(parallel_pairs)} parallel line pairs")
        
        windows = []
        
        # Step 4: Create window polygons from parallel pairs
        for idx_a, idx_b in parallel_pairs:
            try:
                line_a = lines[idx_a]
                line_b = lines[idx_b]
                
                # Compute window center
                (x1_a, y1_a), (x2_a, y2_a) = line_a
                (x1_b, y1_b), (x2_b, y2_b) = line_b
                
                mid_a = np.array([(x1_a + x2_a) / 2, (y1_a + y2_a) / 2])
                mid_b = np.array([(x1_b + x2_b) / 2, (y1_b + y2_b) / 2])
                window_center = (mid_a + mid_b) / 2
                
                # Find which wall this window belongs to
                wall = self._find_containing_wall(tuple(window_center), wall_polygons)
                
                if wall is None:
                    logger.debug(f"Window at {window_center} not on any wall, skipping")
                    continue
                
                # Verify alignment with wall direction
                wall_points = np.array(wall.points, dtype=np.float32)
                spatial_analyzer = SpatialAnalyzer(self.config)
                wall_direction = spatial_analyzer.get_wall_direction(wall_points)
                
                line_direction = self.line_detector.compute_line_angle(line_a)
                
                # Check if line is parallel to wall (within tolerance)
                angle_diff = abs(wall_direction - line_direction)
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                if angle_diff > self.config.parallel_angle_tolerance:
                    logger.debug(
                        f"Window lines not aligned with wall "
                        f"(wall={wall_direction:.1f}°, line={line_direction:.1f}°), skipping"
                    )
                    continue
                
                # Create window polygon
                window = create_window_polygon(line_a, line_b)
                windows.append(window)
            
            except Exception as e:
                logger.warning(f"Failed to create window from parallel lines: {e}")
                continue
        
        logger.info(f"Successfully detected {len(windows)} windows")
        return windows
    
    def _find_containing_wall(
        self,
        point: Tuple[float, float],
        wall_polygons: List[WallPolygon]
    ) -> Optional[WallPolygon]:
        """
        Find which wall contains the given point.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point coordinates (x, y)
        wall_polygons : List[WallPolygon]
            List of wall polygons
        
        Returns
        -------
        WallPolygon or None
            Wall containing the point, or None if not found
        """
        spatial_analyzer = SpatialAnalyzer(self.config)
        
        for wall in wall_polygons:
            if not wall.is_wall:
                continue
            
            if spatial_analyzer.is_point_on_wall(point, wall):
                return wall
        
        return None


def create_window_polygon(
    line_a: LineSegment,
    line_b: LineSegment
) -> WallPolygon:
    """
    Create a window polygon from a pair of parallel lines.
    
    Parameters
    ----------
    line_a : LineSegment
        First line segment ((x1, y1), (x2, y2))
    line_b : LineSegment
        Second line segment ((x1, y1), (x2, y2))
    
    Returns
    -------
    WallPolygon
        WallPolygon with class_id=2, class_name="Window"
    """
    (x1_a, y1_a), (x2_a, y2_a) = line_a
    (x1_b, y1_b), (x2_b, y2_b) = line_b
    
    # Create rectangle from the two parallel lines
    # Use the endpoints of both lines as corners
    points = [
        (int(x1_a), int(y1_a)),
        (int(x2_a), int(y2_a)),
        (int(x2_b), int(y2_b)),
        (int(x1_b), int(y1_b))
    ]
    
    # Compute area and bbox
    points_array = np.array(points)
    area = float(cv2.contourArea(points_array))
    
    # Handle degenerate case where contour area is 0
    if area == 0:
        area = 1.0
    
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    bbox = (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max() - x_coords.min()),
        int(y_coords.max() - y_coords.min())
    )
    
    return WallPolygon(
        class_id=2,
        class_name="Window",
        points=points,
        area=area,
        bbox=bbox,
        confidence=1.0
    )
