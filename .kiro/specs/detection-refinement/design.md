# Design Document: Detection Refinement Module

## Overview

The Detection Refinement Module is a geometry-based enhancement system that improves door and window detection in 2D floor plans. It addresses the weakness of YOLOv8 segmentation in detecting small architectural elements by leveraging wall geometry, spatial relationships, and computer vision techniques.

The module operates as an optional post-processing step inserted between Phase 2 (Segmentation) and Phase 3 (Geometry) of the floor2model pipeline. It replaces unreliable YOLO-based door and window detections with geometry-derived detections while preserving walls and rooms.

### Key Design Principles

1. **Non-invasive Integration**: The module is optional and can be enabled/disabled via configuration without breaking existing functionality
2. **Geometry-First Approach**: Uses wall structure and spatial relationships rather than relying on ML predictions
3. **Hybrid Strategy**: Combines multiple detection techniques (gap analysis, edge detection, spatial reasoning)
4. **Backward Compatibility**: Preserves all existing data structures and interfaces

### Design Goals

- Improve door detection accuracy by analyzing wall gaps and room adjacency
- Improve window detection accuracy using edge detection and parallel line analysis
- Maintain processing time under 1.5 seconds for typical floor plans
- Provide configurable parameters for different floor plan styles
- Ensure robust error handling that never breaks the pipeline

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│                    Floor2Model Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Phase 1: Preprocessing                                      │
│  ┌──────────────────────────────────────────────┐           │
│  │ Loader → Binarizer → Skew Corrector          │           │
│  └──────────────────────────────────────────────┘           │
│                        ↓                                      │
│                  Cleaned Image                               │
│                        ↓                                      │
│  Phase 2: Segmentation (YOLOv8)                             │
│  ┌──────────────────────────────────────────────┐           │
│  │ FloorPlanPredictor                            │           │
│  └──────────────────────────────────────────────┘           │
│                        ↓                                      │
│              SegmentationResult                              │
│                        ↓                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ NEW: Detection Refinement (Optional)                   │ │
│  │ ┌────────────────────────────────────────────────────┐ │ │
│  │ │ RefinementConfig                                   │ │ │
│  │ │   ├─ DoorDetector                                  │ │ │
│  │ │   │   ├─ SpatialAnalyzer                           │ │ │
│  │ │   │   └─ GapDetector                               │ │ │
│  │ │   └─ WindowDetector                                │ │ │
│  │ │       ├─ EdgeAnalyzer                              │ │ │
│  │ │       └─ LineDetector                              │ │ │
│  │ └────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                        ↓                                      │
│          Refined VectorizationResult                         │
│                        ↓                                      │
│  Phase 3: Geometry                                           │
│  ┌──────────────────────────────────────────────┐           │
│  │ ScaleEstimator → RoomGraphBuilder            │           │
│  └──────────────────────────────────────────────┘           │
│                        ↓                                      │
│                 GeometryResult                               │
│                        ↓                                      │
│  Phase 4: Reconstruction                                     │
│  ┌──────────────────────────────────────────────┐           │
│  │ Extruder → MeshBuilder → Exporter            │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

The refinement module consists of four main components:

1. **RefinementConfig**: Configuration dataclass holding all tunable parameters
2. **DoorDetector**: Analyzes wall gaps and room adjacency to place doors
3. **WindowDetector**: Uses edge detection and line analysis to find windows
4. **DetectionRefiner**: Main orchestrator that coordinates detection and merges results

### Integration Points

#### Input Interfaces

The module receives three inputs:

1. **SegmentationResult** (from Phase 2): Contains YOLO-detected elements with masks and bounding boxes
2. **Preprocessed Image** (numpy array): The cleaned binary image from Phase 1
3. **VectorizationResult** (from Phase 3): Contains vectorized walls, rooms, doors, windows as polygons

#### Output Interface

The module produces:

- **Refined VectorizationResult**: Same structure as input, but with doors and windows replaced by geometry-based detections

#### Integration with GeometryPipeline

The module integrates into `src/geometry/pipeline.py` after vectorization:

```python
# In GeometryPipeline.run()
vec_result = self.vectorizer.extract(segmentation_result, image_shape)

# NEW: Apply refinement if enabled
if self.config.use_refinement:
    from src.detection.refinement import refine_detections
    vec_result = refine_detections(
        yolo_output=segmentation_result,
        image=image,
        geometry_output=vec_result,
        config=self.config.refinement_config
    )
```

## Components and Interfaces

### 1. Configuration Component

```python
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
```

### 2. Door Detection Component

#### DoorDetector Class

```python
class DoorDetector:
    """Detects doors based on wall gaps and room adjacency."""
    
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.spatial_analyzer = SpatialAnalyzer(config)
        self.gap_detector = GapDetector(config)
    
    def detect(
        self,
        wall_polygons: list[WallPolygon],
        room_polygons: list[WallPolygon]
    ) -> list[WallPolygon]:
        """
        Detect door locations from wall geometry and room adjacency.
        
        Args:
            wall_polygons: List of vectorized wall polygons
            room_polygons: List of vectorized room polygons
            
        Returns:
            List of door polygons with class_id=3, class_name="Door"
        """
```

**Algorithm**:

1. Find all pairs of adjacent rooms using `spatial_analyzer.find_adjacent_rooms()`
2. For each adjacent room pair:
   - Extract the shared wall segment using `spatial_analyzer.find_shared_wall_segment()`
   - Check for gaps in the wall using `gap_detector.detect_wall_gaps()`
   - If gap found: place door at gap location
   - If no gap: place door at midpoint of shared wall
3. Create door polygons with proper orientation using `create_door_polygon()`
4. Return list of door WallPolygon objects

#### SpatialAnalyzer Class

```python
class SpatialAnalyzer:
    """Analyzes spatial relationships between rooms and walls."""
    
    def find_adjacent_rooms(
        self,
        room_polygons: list[WallPolygon]
    ) -> list[tuple[WallPolygon, WallPolygon]]:
        """
        Find pairs of rooms that share a wall boundary.
        
        Uses polygon proximity analysis: rooms are adjacent if their
        boundaries come within room_adjacency_threshold pixels.
        
        Returns:
            List of (room_a, room_b) tuples
        """
    
    def find_shared_wall_segment(
        self,
        room_a: WallPolygon,
        room_b: WallPolygon,
        wall_polygons: list[WallPolygon]
    ) -> Optional[np.ndarray]:
        """
        Extract the wall segment between two adjacent rooms.
        
        Algorithm:
        1. Find the line connecting room centroids
        2. Find wall polygons that intersect this line
        3. Extract the portion of wall between the two rooms
        
        Returns:
            Nx2 numpy array of wall segment points, or None if not found
        """
    
    def is_point_on_wall(
        self,
        point: tuple[float, float],
        wall_polygon: WallPolygon
    ) -> bool:
        """
        Check if a point lies within wall polygon boundaries.
        
        Uses cv2.pointPolygonTest with tolerance from config.
        """
    
    def get_wall_direction(
        self,
        wall_segment: np.ndarray
    ) -> float:
        """
        Compute orientation angle of a wall segment.
        
        Uses PCA (principal component analysis) on wall points
        to find dominant direction.
        
        Returns:
            Angle in degrees (0-180 range)
        """
```

#### GapDetector Class

```python
class GapDetector:
    """Detects gaps and discontinuities in wall segments."""
    
    def detect_wall_gaps(
        self,
        wall_segment: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Identify breaks or discontinuities in a wall segment.
        
        Algorithm:
        1. Sort wall points along the wall direction
        2. Compute distances between consecutive points
        3. Identify gaps where distance > wall_gap_threshold
        
        Returns:
            List of (start_idx, end_idx) tuples indicating gap locations
        """
```

#### Helper Functions

```python
def create_door_polygon(
    position: tuple[float, float],
    wall_direction: float,
    door_width: int
) -> WallPolygon:
    """
    Create a door polygon at the specified position.
    
    Args:
        position: (x, y) center point for door
        wall_direction: Wall orientation in degrees
        door_width: Door width in pixels
        
    Returns:
        WallPolygon with class_id=3, class_name="Door"
    """
```

### 3. Window Detection Component

#### WindowDetector Class

```python
class WindowDetector:
    """Detects windows using edge detection and line analysis."""
    
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.edge_analyzer = EdgeAnalyzer(config)
        self.line_detector = LineDetector(config)
    
    def detect(
        self,
        image: np.ndarray,
        wall_polygons: list[WallPolygon]
    ) -> list[WallPolygon]:
        """
        Detect window locations using edge detection and line analysis.
        
        Args:
            image: Preprocessed floor plan image (grayscale)
            wall_polygons: List of vectorized wall polygons
            
        Returns:
            List of window polygons with class_id=2, class_name="Window"
        """
```

**Algorithm**:

1. Apply Canny edge detection using `edge_analyzer.detect_edges()`
2. Extract line segments using `line_detector.detect_lines()`
3. Find pairs of parallel lines using `line_detector.find_parallel_pairs()`
4. For each parallel pair:
   - Verify alignment with wall direction
   - Verify lines lie within wall boundaries
   - Check distance between lines is reasonable for window
5. Create window polygons using `create_window_polygon()`
6. Return list of window WallPolygon objects

#### EdgeAnalyzer Class

```python
class EdgeAnalyzer:
    """Performs edge detection on floor plan images."""
    
    def detect_edges(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply Canny edge detection to identify edges.
        
        Uses config.canny_low_threshold and config.canny_high_threshold.
        
        Returns:
            Binary edge map (same size as input)
        """
```

#### LineDetector Class

```python
class LineDetector:
    """Detects and analyzes line segments in images."""
    
    def detect_lines(
        self,
        edge_map: np.ndarray
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Extract line segments using HoughLinesP.
        
        Uses config.hough_* parameters.
        
        Returns:
            List of ((x1, y1), (x2, y2)) line segments
        """
    
    def find_parallel_pairs(
        self,
        lines: list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> list[tuple[int, int]]:
        """
        Identify pairs of parallel lines with close proximity.
        
        Algorithm:
        1. Compute orientation angle for each line
        2. For each line pair:
           - Check if angles are within parallel_angle_tolerance
           - Check if distance between lines < parallel_line_proximity
        3. Return indices of parallel pairs
        
        Returns:
            List of (line_idx_a, line_idx_b) tuples
        """
    
    def compute_line_angle(
        self,
        line: tuple[tuple[int, int], tuple[int, int]]
    ) -> float:
        """
        Compute orientation angle of a line segment.
        
        Returns:
            Angle in degrees (0-180 range)
        """
    
    def compute_line_distance(
        self,
        line_a: tuple[tuple[int, int], tuple[int, int]],
        line_b: tuple[tuple[int, int], tuple[int, int]]
    ) -> float:
        """
        Compute perpendicular distance between two parallel lines.
        
        Returns:
            Distance in pixels
        """
```

#### Helper Functions

```python
def create_window_polygon(
    line_a: tuple[tuple[int, int], tuple[int, int]],
    line_b: tuple[tuple[int, int], tuple[int, int]]
) -> WallPolygon:
    """
    Create a window polygon from a pair of parallel lines.
    
    Args:
        line_a: First line segment ((x1, y1), (x2, y2))
        line_b: Second line segment ((x1, y1), (x2, y2))
        
    Returns:
        WallPolygon with class_id=2, class_name="Window"
    """
```

### 4. Main Refinement Component

#### DetectionRefiner Class

```python
class DetectionRefiner:
    """Main orchestrator for detection refinement."""
    
    def __init__(self, config: Optional[RefinementConfig] = None):
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
        
        Args:
            yolo_output: Original YOLO segmentation result
            image: Preprocessed floor plan image
            geometry_output: Vectorized geometry from Phase 3
            
        Returns:
            Refined VectorizationResult with improved doors/windows
        """
```

**Algorithm**:

1. Extract walls and rooms from geometry_output (preserve unchanged)
2. If `config.enable_door_detection`:
   - Call `door_detector.detect(walls, rooms)`
   - Store refined door polygons
3. If `config.enable_window_detection`:
   - Call `window_detector.detect(image, walls)`
   - Store refined window polygons
4. Create new VectorizationResult with:
   - Original walls (unchanged)
   - Original rooms (unchanged)
   - Refined doors (from door_detector)
   - Refined windows (from window_detector)
   - Original other elements (unchanged)
5. Return refined result

#### Public API Function

```python
def refine_detections(
    yolo_output: SegmentationResult,
    image: np.ndarray,
    geometry_output: VectorizationResult,
    config: Optional[RefinementConfig] = None
) -> VectorizationResult:
    """
    Public API function for detection refinement.
    
    This is the main entry point called by GeometryPipeline.
    
    Args:
        yolo_output: Original YOLO segmentation result
        image: Preprocessed floor plan image
        geometry_output: Vectorized geometry from Phase 3
        config: Optional custom configuration
        
    Returns:
        Refined VectorizationResult with improved doors/windows
    """
    refiner = DetectionRefiner(config)
    return refiner.refine(yolo_output, image, geometry_output)
```

## Data Models

### Existing Data Structures (Unchanged)

The module uses existing data structures without modification:

#### WallPolygon (from src/geometry/wall_vectorizer.py)

```python
@dataclass
class WallPolygon:
    class_id: int                        # 0=OuterWall, 1=InnerWall, 2=Window, 3=Door, etc.
    class_name: str                      # "OuterWall", "Door", "Window", etc.
    points: list[tuple[int, int]]        # Pixel coordinates (x, y)
    area: float                          # Pixel area
    bbox: tuple[int, int, int, int]      # (x, y, w, h)
    confidence: float = 1.0
    
    @property
    def centroid(self) -> tuple[float, float]
    @property
    def is_wall(self) -> bool
    @property
    def is_room(self) -> bool
```

#### VectorizationResult (from src/geometry/wall_vectorizer.py)

```python
@dataclass
class VectorizationResult:
    walls: list[WallPolygon]
    rooms: list[WallPolygon]
    doors: list[WallPolygon]
    windows: list[WallPolygon]
    other: list[WallPolygon]
    image_shape: tuple[int, int]
    
    @property
    def all_polygons(self) -> list[WallPolygon]
    @property
    def summary(self) -> dict
```

#### SegmentationResult (from src/segmentation/predictor.py)

```python
@dataclass
class SegmentationResult:
    image_path: str
    image_shape: tuple[int, int]
    elements: list[DetectedElement]
    
    @property
    def walls(self) -> list[DetectedElement]
    @property
    def doors(self) -> list[DetectedElement]
    @property
    def windows(self) -> list[DetectedElement]
    @property
    def rooms(self) -> list[DetectedElement]
```

### Internal Data Structures

#### Line Segment Representation

```python
# Type alias for clarity
LineSegment = tuple[tuple[int, int], tuple[int, int]]  # ((x1, y1), (x2, y2))
```

#### Wall Gap Representation

```python
@dataclass
class WallGap:
    """Represents a gap or discontinuity in a wall segment."""
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    width: float  # pixels
    midpoint: tuple[float, float]
```

#### Room Adjacency Representation

```python
@dataclass
class RoomAdjacency:
    """Represents spatial relationship between two rooms."""
    room_a: WallPolygon
    room_b: WallPolygon
    shared_wall: Optional[np.ndarray]  # Nx2 array of wall segment points
    distance: float  # pixels between centroids
```

## 

## Error Handling

### Error Handling Strategy

The refinement module follows a fail-safe approach where errors never break the pipeline. If refinement fails, the system falls back to original YOLO detections.

### Error Categories and Handling

#### 1. Input Validation Errors

**Scenario**: Invalid or missing input data

**Handling**:
- Empty wall_polygons → Return empty door/window lists
- Empty room_polygons → Skip door detection, return empty list
- None or invalid image → Skip window detection, return empty list
- Invalid VectorizationResult → Return original unchanged

**Implementation**:
```python
def detect_doors_from_walls(wall_polygons, room_polygons):
    if not wall_polygons:
        logger.warning("No wall polygons provided, skipping door detection")
        return []
    if not room_polygons:
        logger.warning("No room polygons provided, skipping door detection")
        return []
    # ... proceed with detection
```

#### 2. Detection Failures

**Scenario**: No detections found despite valid inputs

**Handling**:
- No adjacent rooms found → Return empty door list (not an error)
- No parallel lines detected → Return empty window list (not an error)
- No wall gaps found → Place doors at midpoints (fallback strategy)

**Implementation**:
```python
def detect_wall_gaps(wall_segment):
    gaps = []
    # ... gap detection logic
    if not gaps:
        logger.info("No wall gaps detected, will use midpoint placement")
    return gaps
```

#### 3. Geometric Computation Errors

**Scenario**: Mathematical operations fail (division by zero, invalid angles, etc.)

**Handling**:
- Wrap geometric computations in try-except blocks
- Log warnings with context
- Skip problematic elements, continue with others
- Return partial results rather than failing completely

**Implementation**:
```python
def get_wall_direction(wall_segment):
    try:
        # PCA or line fitting
        pca = PCA(n_components=1)
        pca.fit(wall_segment)
        angle = compute_angle(pca.components_[0])
        return angle
    except Exception as e:
        logger.warning(f"Failed to compute wall direction: {e}")
        return 0.0  # Default to horizontal
```

#### 4. OpenCV Operation Errors

**Scenario**: OpenCV functions fail (invalid image format, memory issues, etc.)

**Handling**:
- Validate image format before processing
- Catch cv2 exceptions and log details
- Return empty results on failure

**Implementation**:
```python
def detect_edges(image):
    if image is None or image.size == 0:
        logger.error("Invalid image provided to edge detection")
        return np.zeros_like(image) if image is not None else None
    
    try:
        edges = cv2.Canny(image, self.config.canny_low_threshold, 
                         self.config.canny_high_threshold)
        return edges
    except cv2.error as e:
        logger.error(f"OpenCV edge detection failed: {e}")
        return np.zeros_like(image)
```

#### 5. Configuration Errors

**Scenario**: Invalid configuration parameters

**Handling**:
- Validate configuration on initialization
- Clamp values to reasonable ranges
- Use defaults for invalid values

**Implementation**:
```python
@dataclass
class RefinementConfig:
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    
    def __post_init__(self):
        # Validate and clamp values
        self.canny_low_threshold = max(1, min(255, self.canny_low_threshold))
        self.canny_high_threshold = max(1, min(255, self.canny_high_threshold))
        if self.canny_high_threshold <= self.canny_low_threshold:
            logger.warning("Invalid Canny thresholds, using defaults")
            self.canny_low_threshold = 50
            self.canny_high_threshold = 150
```

### Logging Strategy

The module uses Python's logging module with the following levels:

- **DEBUG**: Detailed algorithm steps, intermediate results
- **INFO**: Normal operation messages (detections found, processing steps)
- **WARNING**: Recoverable issues (no detections, fallback strategies used)
- **ERROR**: Serious issues that prevent refinement (invalid inputs, OpenCV failures)

**Example**:
```python
import logging

logger = logging.getLogger(__name__)

def refine_detections(yolo_output, image, geometry_output, config=None):
    logger.info("Starting detection refinement")
    
    try:
        # ... refinement logic
        logger.info(f"Refinement complete: {len(doors)} doors, {len(windows)} windows")
        return refined_result
    except Exception as e:
        logger.error(f"Refinement failed: {e}", exc_info=True)
        logger.warning("Falling back to original YOLO detections")
        return geometry_output  # Return original unchanged
```

### Graceful Degradation

The module implements graceful degradation at multiple levels:

1. **Module Level**: If entire refinement fails, return original VectorizationResult
2. **Component Level**: If door detection fails, still attempt window detection
3. **Element Level**: If one door placement fails, continue with other doors

**Implementation**:
```python
def refine(self, yolo_output, image, geometry_output):
    refined_doors = []
    refined_windows = []
    
    # Try door detection
    if self.config.enable_door_detection:
        try:
            refined_doors = self.door_detector.detect(
                geometry_output.walls, 
                geometry_output.rooms
            )
        except Exception as e:
            logger.error(f"Door detection failed: {e}")
            refined_doors = geometry_output.doors  # Use original
    
    # Try window detection independently
    if self.config.enable_window_detection:
        try:
            refined_windows = self.window_detector.detect(
                image, 
                geometry_output.walls
            )
        except Exception as e:
            logger.error(f"Window detection failed: {e}")
            refined_windows = geometry_output.windows  # Use original
    
    # Merge results
    return VectorizationResult(
        walls=geometry_output.walls,
        rooms=geometry_output.rooms,
        doors=refined_doors,
        windows=refined_windows,
        other=geometry_output.other,
        image_shape=geometry_output.image_shape
    )
```

## Testing Strategy

### Dual Testing Approach

The refinement module will use both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs using randomized testing

Both approaches are complementary and necessary. Unit tests catch concrete bugs and verify specific behaviors, while property tests ensure general correctness across a wide range of inputs.

### Unit Testing

Unit tests will focus on:

1. **Specific examples**: Known floor plans with expected door/window locations
2. **Edge cases**: Empty inputs, single room, no walls, complex geometries
3. **Error conditions**: Invalid inputs, malformed data, OpenCV failures
4. **Integration points**: Interaction with existing data structures

**Test File**: `tests/test_detection.py`

**Key Test Cases**:

```python
def test_detect_doors_from_adjacent_rooms():
    """Verify door placement between two adjacent rooms."""
    # Create two adjacent room polygons
    # Run door detection
    # Assert door is placed between rooms
    
def test_detect_doors_with_wall_gaps():
    """Verify door placement at detected wall gaps."""
    # Create wall with gap
    # Run door detection
    # Assert door is placed at gap location
    
def test_detect_windows_from_edges():
    """Verify window detection from parallel lines."""
    # Create image with parallel edges
    # Run window detection
    # Assert windows are detected at parallel line locations
    
def test_refine_detections_preserves_walls():
    """Verify walls are unchanged after refinement."""
    # Create VectorizationResult
    # Run refinement
    # Assert walls are identical
    
def test_door_placement_on_wall():
    """Verify doors lie on wall geometry."""
    # Detect doors
    # For each door, verify centroid is on wall
    
def test_window_alignment_with_wall():
    """Verify windows align with wall direction."""
    # Detect windows
    # For each window, verify orientation matches wall
    
def test_empty_wall_polygons():
    """Verify graceful handling of empty wall list."""
    # Call detect_doors_from_walls with empty list
    # Assert returns empty list without exception
    
def test_empty_room_polygons():
    """Verify graceful handling of empty room list."""
    # Call detect_doors_from_walls with empty rooms
    # Assert returns empty list without exception
    
def test_invalid_image():
    """Verify graceful handling of invalid image."""
    # Call detect_windows_from_walls with None image
    # Assert returns empty list without exception
```

### Property-Based Testing

Property-based tests will use a PBT library (pytest-hypothesis for Python) to verify universal properties across randomized inputs.

**Configuration**: Each property test will run minimum 100 iterations with randomized inputs.

**Test Tagging**: Each property test will include a comment tag referencing the design property:
```python
# Feature: detection-refinement, Property 1: Preservation of walls and rooms
```

The specific correctness properties will be defined in the Correctness Properties section below.

### Integration Testing

Integration tests will verify the refinement module works correctly within the full pipeline:

**Test File**: `tests/test_detection_integration.py`

```python
def test_refinement_in_geometry_pipeline():
    """Verify refinement integrates correctly with GeometryPipeline."""
    # Create GeometryConfig with use_refinement=True
    # Run full pipeline
    # Verify refined detections are present
    
def test_refinement_disabled():
    """Verify pipeline works with refinement disabled."""
    # Create GeometryConfig with use_refinement=False
    # Run full pipeline
    # Verify original YOLO detections are used
    
def test_refinement_backward_compatibility():
    """Verify existing tests still pass with refinement."""
    # Run all existing geometry tests
    # Assert no regressions
```

### Performance Testing

Performance tests will verify the module meets timing requirements:

```python
def test_door_detection_performance():
    """Verify door detection completes within 500ms."""
    # Create typical floor plan (1024x1024)
    # Time door detection
    # Assert time < 500ms
    
def test_window_detection_performance():
    """Verify window detection completes within 1000ms."""
    # Create typical floor plan (1024x1024)
    # Time window detection
    # Assert time < 1000ms
```

### Test Data

Test data will include:

1. **Synthetic floor plans**: Programmatically generated simple geometries
2. **Real floor plans**: Sample images from CubiCasa5k dataset
3. **Edge cases**: Unusual geometries, rotated plans, complex layouts


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all acceptance criteria, I identified the following testable properties. I then performed reflection to eliminate redundancy:

**Redundancy Analysis**:
- Properties 4.2 and 4.3 (preserve walls/rooms) can be combined with 11.8 (preservation round-trip) into a single comprehensive preservation property
- Properties 2.5 (door on wall) and 3.6 (window within wall) both verify element placement validity and can be combined
- Properties 3.5 and 3.7 (window alignment) are redundant - 3.7 subsumes 3.5
- Properties 7.5 and 7.6 (polygon format) can be combined into a single format validation property

After eliminating redundancy, the following properties provide unique validation value:

### Property 1: Preservation of Walls and Rooms

For any VectorizationResult with walls and rooms, refining the detections should preserve the original wall polygons and room polygons unchanged (same points, same order, same metadata).

**Validates: Requirements 4.2, 4.3, 11.8**

### Property 2: Shared Wall Identification

For any pair of adjacent rooms (rooms whose boundaries come within the adjacency threshold), the system should identify a shared wall segment that lies between their centroids.

**Validates: Requirements 2.2**

### Property 3: Door Placement at Wall Gaps

For any wall segment containing a gap (discontinuity wider than the gap threshold), when a door is placed, it should be positioned at the gap location rather than elsewhere on the wall.

**Validates: Requirements 2.3**

### Property 4: Door Placement Fallback

For any wall segment between adjacent rooms that contains no detectable gaps, a door should be placed at the midpoint of the shared wall segment.

**Validates: Requirements 2.4**

### Property 5: Element Placement Validity

For any detected door or window polygon, its centroid should lie within or on the boundary of at least one wall polygon (within the wall thickness tolerance).

**Validates: Requirements 2.5, 3.6**

### Property 6: Door-Room Connectivity

For any detected door polygon, it should be spatially associated with exactly two room polygons (within the door proximity threshold).

**Validates: Requirements 2.6**

### Property 7: Door Orientation Perpendicularity

For any door placed on a wall segment, the door's orientation angle should be perpendicular (90 degrees ± tolerance) to the wall segment's direction angle.

**Validates: Requirements 2.7**

### Property 8: Parallel Line Identification

For any pair of lines identified as parallel, their orientation angles should differ by less than the parallel angle tolerance, and their perpendicular distance should be less than the parallel line proximity threshold.

**Validates: Requirements 3.4**

### Property 9: Window Alignment with Walls

For any detected window polygon, its orientation should be parallel (within angle tolerance) to the wall segment it lies on.

**Validates: Requirements 3.5, 3.7**

### Property 10: Window Centering

For any window polygon placed on a wall segment, the window's centroid should be approximately centered (within tolerance) along the wall segment's length.

**Validates: Requirements 2.8**

### Property 11: Door and Window Replacement

For any VectorizationResult that undergoes refinement, the output should contain door polygons from geometry-based detection (not from YOLO) and window polygons from edge-based detection (not from YOLO).

**Validates: Requirements 4.4, 4.5**

### Property 12: Polygon Format Validity

For any polygon returned by the refinement module (doors or windows), it should be a NumPy array with shape (N, 2) where N ≥ 3, all coordinates should be integers, and all coordinates should lie within the image boundaries.

**Validates: Requirements 7.5, 7.6**

### Property 13: Wall Direction Angle Range

For any wall segment, the computed wall direction angle should be in the range [0, 180) degrees.

**Validates: Requirements 8.3**


## Implementation Details

### File Structure

```
src/detection/
├── __init__.py
└── refinement.py
    ├── RefinementConfig (dataclass)
    ├── DetectionRefiner (main class)
    ├── DoorDetector (class)
    │   ├── SpatialAnalyzer (class)
    │   └── GapDetector (class)
    ├── WindowDetector (class)
    │   ├── EdgeAnalyzer (class)
    │   └── LineDetector (class)
    └── refine_detections() (public API function)
```

### Dependencies

**External Libraries**:
- `numpy`: Array operations and geometric computations
- `cv2` (OpenCV): Edge detection, line detection, morphological operations
- `dataclasses`: Configuration and data structures
- `typing`: Type hints
- `logging`: Error handling and debugging

**Internal Modules**:
- `src.segmentation.predictor`: SegmentationResult, DetectedElement
- `src.geometry.wall_vectorizer`: WallPolygon, VectorizationResult

### Algorithm Pseudocode

#### Door Detection Algorithm

```
function detect_doors_from_walls(wall_polygons, room_polygons):
    if wall_polygons is empty or room_polygons is empty:
        return []
    
    doors = []
    adjacent_pairs = find_adjacent_rooms(room_polygons)
    
    for (room_a, room_b) in adjacent_pairs:
        shared_wall = find_shared_wall_segment(room_a, room_b, wall_polygons)
        
        if shared_wall is None:
            continue
        
        gaps = detect_wall_gaps(shared_wall)
        
        if gaps is not empty:
            # Place door at first gap
            gap_midpoint = (gaps[0].start_point + gaps[0].end_point) / 2
            door_position = gap_midpoint
        else:
            # Fallback: place door at wall midpoint
            door_position = compute_midpoint(shared_wall)
        
        wall_direction = get_wall_direction(shared_wall)
        door = create_door_polygon(door_position, wall_direction, door_width)
        doors.append(door)
    
    return doors
```

#### Window Detection Algorithm

```
function detect_windows_from_walls(image, wall_polygons):
    if image is None or wall_polygons is empty:
        return []
    
    # Step 1: Edge detection
    edges = apply_canny_edge_detection(image)
    
    # Step 2: Line detection
    lines = detect_lines_hough(edges)
    
    if lines is empty:
        return []
    
    # Step 3: Find parallel line pairs
    parallel_pairs = find_parallel_pairs(lines)
    
    windows = []
    for (line_a, line_b) in parallel_pairs:
        # Verify lines are close together (window width)
        distance = compute_line_distance(line_a, line_b)
        if distance > parallel_line_proximity:
            continue
        
        # Find which wall this window belongs to
        window_center = compute_midpoint(line_a, line_b)
        wall = find_containing_wall(window_center, wall_polygons)
        
        if wall is None:
            continue
        
        # Verify alignment with wall direction
        wall_direction = get_wall_direction(wall)
        line_direction = compute_line_angle(line_a)
        
        if abs(wall_direction - line_direction) > parallel_angle_tolerance:
            continue
        
        # Create window polygon
        window = create_window_polygon(line_a, line_b)
        windows.append(window)
    
    return windows
```

#### Main Refinement Algorithm

```
function refine_detections(yolo_output, image, geometry_output, config):
    try:
        # Initialize refiner
        refiner = DetectionRefiner(config)
        
        # Detect doors
        refined_doors = []
        if config.enable_door_detection:
            try:
                refined_doors = refiner.door_detector.detect(
                    geometry_output.walls,
                    geometry_output.rooms
                )
            except Exception as e:
                log_error("Door detection failed", e)
                refined_doors = geometry_output.doors  # Fallback to original
        
        # Detect windows
        refined_windows = []
        if config.enable_window_detection:
            try:
                refined_windows = refiner.window_detector.detect(
                    image,
                    geometry_output.walls
                )
            except Exception as e:
                log_error("Window detection failed", e)
                refined_windows = geometry_output.windows  # Fallback to original
        
        # Merge results
        return VectorizationResult(
            walls=geometry_output.walls,
            rooms=geometry_output.rooms,
            doors=refined_doors,
            windows=refined_windows,
            other=geometry_output.other,
            image_shape=geometry_output.image_shape
        )
    
    except Exception as e:
        log_error("Refinement failed completely", e)
        return geometry_output  # Return original unchanged
```

### Geometric Computations

#### Point-on-Wall Test

```python
def is_point_on_wall(point: tuple[float, float], wall: WallPolygon) -> bool:
    """Check if point lies within wall polygon boundaries."""
    wall_pts = np.array(wall.points, dtype=np.int32)
    dist = cv2.pointPolygonTest(wall_pts, point, measureDist=True)
    return abs(dist) <= config.wall_thickness_tolerance
```

#### Wall Direction Computation

```python
def get_wall_direction(wall_segment: np.ndarray) -> float:
    """Compute wall orientation using PCA."""
    from sklearn.decomposition import PCA
    
    if len(wall_segment) < 2:
        return 0.0
    
    pca = PCA(n_components=1)
    pca.fit(wall_segment)
    
    # Get principal component (dominant direction)
    component = pca.components_[0]
    angle = np.arctan2(component[1], component[0])
    
    # Convert to degrees, normalize to [0, 180)
    angle_deg = np.degrees(angle) % 180
    return angle_deg
```

#### Parallel Line Detection

```python
def find_parallel_pairs(lines: list[LineSegment]) -> list[tuple[int, int]]:
    """Find pairs of parallel lines with close proximity."""
    pairs = []
    n = len(lines)
    
    for i in range(n):
        for j in range(i + 1, n):
            angle_i = compute_line_angle(lines[i])
            angle_j = compute_line_angle(lines[j])
            
            # Check angle similarity
            angle_diff = abs(angle_i - angle_j)
            if angle_diff > config.parallel_angle_tolerance:
                continue
            
            # Check distance
            distance = compute_line_distance(lines[i], lines[j])
            if distance > config.parallel_line_proximity:
                continue
            
            pairs.append((i, j))
    
    return pairs
```

### Performance Optimizations

1. **Spatial Indexing**: Use KD-tree or R-tree for fast room adjacency queries
2. **Vectorized Operations**: Use NumPy broadcasting for distance computations
3. **Early Termination**: Skip processing if no walls or rooms detected
4. **Caching**: Cache wall directions and centroids to avoid recomputation
5. **Parallel Processing**: Use multiprocessing for independent door/window detection

### Configuration Integration

The refinement configuration will be added to `GeometryConfig`:

```python
@dataclass
class GeometryConfig:
    # ... existing fields ...
    
    # Refinement configuration
    use_refinement: bool = True
    refinement_config: Optional[RefinementConfig] = None
    
    def __post_init__(self):
        if self.refinement_config is None:
            self.refinement_config = RefinementConfig()
```

### Usage Example

```python
from src.preprocessing.pipeline import PreprocessingPipeline
from src.segmentation.predictor import FloorPlanPredictor
from src.geometry.pipeline import GeometryPipeline, GeometryConfig
from src.detection.refinement import RefinementConfig

# Phase 1: Preprocessing
prep_pipeline = PreprocessingPipeline()
cleaned_image = prep_pipeline.run("samples/floorplan1.png")

# Phase 2: Segmentation
predictor = FloorPlanPredictor("src/segmentation/best.pt")
seg_result = predictor.predict(cleaned_image)

# Phase 3: Geometry with Refinement
refinement_config = RefinementConfig(
    canny_low_threshold=50,
    canny_high_threshold=150,
    door_width=90,
    enable_door_detection=True,
    enable_window_detection=True
)

geo_config = GeometryConfig(
    use_refinement=True,
    refinement_config=refinement_config
)

geo_pipeline = GeometryPipeline(config=geo_config)
geo_result = geo_pipeline.run(seg_result, cleaned_image, "floorplan1.png")

# Results
print(f"Detected {len(geo_result.vectorization.doors)} doors")
print(f"Detected {len(geo_result.vectorization.windows)} windows")
```

### Disabling Refinement

```python
# Disable refinement to use original YOLO detections
geo_config = GeometryConfig(use_refinement=False)
geo_pipeline = GeometryPipeline(config=geo_config)
geo_result = geo_pipeline.run(seg_result, cleaned_image, "floorplan1.png")
```

## Validation and Verification

### Validation Criteria

The implementation will be considered successful if:

1. **Functional Correctness**: All 13 correctness properties pass with 100+ randomized test cases each
2. **Performance**: Door detection < 500ms, window detection < 1000ms for 1024x1024 images
3. **Robustness**: No crashes or exceptions for any valid input (graceful degradation)
4. **Integration**: All existing tests continue to pass with refinement disabled
5. **Accuracy**: Improved detection rates compared to YOLO baseline (measured on test set)

### Verification Methods

1. **Property-Based Testing**: Use pytest-hypothesis to verify all 13 properties with randomized inputs
2. **Unit Testing**: Verify specific examples and edge cases
3. **Integration Testing**: Verify end-to-end pipeline with refinement enabled/disabled
4. **Performance Testing**: Benchmark execution time on typical floor plans
5. **Visual Inspection**: Manual review of detection results on sample images

### Acceptance Metrics

- **Property Test Pass Rate**: 100% (all properties must pass)
- **Unit Test Coverage**: ≥ 90% code coverage
- **Performance**: ≤ 1.5 seconds total refinement time
- **Accuracy Improvement**: ≥ 20% increase in door/window detection F1-score vs YOLO baseline
- **Backward Compatibility**: 100% of existing tests pass with refinement disabled

## Future Enhancements

### Potential Improvements

1. **Machine Learning Hybrid**: Combine geometry-based detection with lightweight ML classifier for validation
2. **Multi-Scale Detection**: Detect windows at multiple scales for better small window detection
3. **Semantic Context**: Use room types to inform door/window placement (e.g., bathrooms typically have one door)
4. **Confidence Scoring**: Assign confidence scores to detections based on geometric evidence strength
5. **Interactive Refinement**: Allow user to manually adjust detected doors/windows in UI
6. **Adaptive Thresholds**: Automatically tune detection thresholds based on image characteristics

### Extension Points

The design includes extension points for future enhancements:

1. **Custom Detectors**: Abstract detector interface allows adding new detection strategies
2. **Pluggable Algorithms**: Wall direction, gap detection, and line detection can be swapped
3. **Configuration Profiles**: Pre-defined configuration sets for different floor plan styles
4. **Post-Processing Hooks**: Allow custom validation and filtering of detections

## References

### Related Work

- **Floor Plan Analysis**: Dodge et al. (2017) "Parsing Floor Plan Images"
- **Architectural Element Detection**: Liu et al. (2017) "Raster-to-Vector: Revisiting Floorplan Transformation"
- **Edge-Based Detection**: Canny (1986) "A Computational Approach to Edge Detection"
- **Line Detection**: Hough (1962) "Method and Means for Recognizing Complex Patterns"

### Internal Documentation

- `ANALYSIS.md`: Codebase analysis and integration details
- `.kiro/specs/detection-refinement/requirements.md`: Feature requirements
- `src/geometry/wall_vectorizer.py`: Vectorization implementation
- `src/geometry/pipeline.py`: Geometry pipeline implementation

