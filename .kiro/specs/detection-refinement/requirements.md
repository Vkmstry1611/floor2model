# Requirements Document

## Introduction

The Hybrid Detection Refinement System addresses the weakness of YOLOv8 segmentation in detecting doors and windows in 2D floor plans. The current Phase 2 segmentation misses small objects and produces unreliable results for these critical architectural elements. This feature introduces a geometry-based refinement module that leverages wall geometry, spatial relationships, and computer vision techniques to improve door and window detection accuracy without relying solely on YOLO predictions.

The system will be implemented as a new module (`src/detection/refinement.py`) that integrates between Phase 2 (Segmentation) and Phase 3 (Geometry) of the floor2model pipeline, providing refined detection results while preserving existing walls and rooms.

## Glossary

- **YOLO_Segmentation**: YOLOv8-based segmentation model that detects floor plan elements (Phase 2)
- **Refinement_Module**: New geometry-based detection system for doors and windows
- **Wall_Polygon**: Vectorized wall representation with pixel coordinates and metadata
- **Room_Polygon**: Vectorized room boundary representation
- **VectorizationResult**: Data structure containing walls, rooms, doors, and windows as polygon arrays
- **SegmentationResult**: Output from YOLO containing detected elements with masks and bounding boxes
- **Shared_Wall**: Wall segment that forms a boundary between two adjacent rooms
- **Wall_Gap**: Break or discontinuity in a wall segment indicating potential door location
- **Edge_Detection**: Computer vision technique using Canny algorithm to detect edges in images
- **Line_Detection**: Hough transform technique to identify straight line segments in images
- **Parallel_Lines**: Pair of lines with similar orientation and close proximity
- **Geometry_Pipeline**: Phase 3 pipeline that converts segmentation to vector geometry

## Requirements

### Requirement 1: Refinement Module Creation

**User Story:** As a developer, I want a dedicated refinement module, so that door and window detection logic is isolated and maintainable.

#### Acceptance Criteria

1. THE Refinement_Module SHALL be located at `src/detection/refinement.py`
2. THE Refinement_Module SHALL import SegmentationResult and DetectedElement from `src/segmentation/predictor`
3. THE Refinement_Module SHALL import WallPolygon and VectorizationResult from `src/geometry/wall_vectorizer`
4. THE Refinement_Module SHALL use only OpenCV and NumPy for computer vision operations
5. THE Refinement_Module SHALL NOT modify existing data structure definitions

### Requirement 2: Door Detection from Wall Geometry

**User Story:** As a system, I want to detect doors based on wall gaps and room adjacency, so that door placement is spatially correct even when YOLO misses them.

#### Acceptance Criteria

1. THE Refinement_Module SHALL provide a function `detect_doors_from_walls(wall_polygons, room_polygons)` that returns a list of door polygon arrays
2. WHEN two rooms are adjacent, THE Refinement_Module SHALL identify the shared wall segment between them
3. WHEN a wall gap is detected in a shared wall, THE Refinement_Module SHALL place a door polygon at the gap location
4. IF no clear wall gap exists in a shared wall, THEN THE Refinement_Module SHALL place a door polygon at the midpoint of the shared wall segment
5. THE Refinement_Module SHALL ensure each door polygon lies ON the wall geometry
6. THE Refinement_Module SHALL ensure each door connects exactly two rooms
7. WHEN placing doors, THE Refinement_Module SHALL align door orientation perpendicular to wall direction

### Requirement 3: Window Detection from Edge Analysis

**User Story:** As a system, I want to detect windows using edge detection and line analysis, so that windows are identified based on visual features rather than unreliable YOLO predictions.

#### Acceptance Criteria

1. THE Refinement_Module SHALL provide a function `detect_windows_from_walls(image, wall_polygons)` that returns a list of window polygon arrays
2. WHEN analyzing an image, THE Refinement_Module SHALL apply Canny edge detection to identify edges
3. WHEN edges are detected, THE Refinement_Module SHALL use HoughLinesP to extract line segments
4. THE Refinement_Module SHALL identify pairs of parallel lines with close proximity (distance less than wall thickness)
5. THE Refinement_Module SHALL verify that parallel line pairs align with wall direction (within 15 degrees)
6. THE Refinement_Module SHALL verify that detected lines lie within wall polygon boundaries
7. WHEN valid parallel line pairs are found, THE Refinement_Module SHALL create window polygons aligned with wall geometry
8. THE Refinement_Module SHALL center window polygons within their respective wall segments

### Requirement 4: Detection Refinement Integration

**User Story:** As a developer, I want a main refinement function that orchestrates door and window detection, so that I can easily integrate refinement into the existing pipeline.

#### Acceptance Criteria

1. THE Refinement_Module SHALL provide a function `refine_detections(yolo_output, image, geometry_output)` that returns a refined VectorizationResult
2. WHEN refining detections, THE Refinement_Module SHALL preserve wall polygons from the geometry_output unchanged
3. WHEN refining detections, THE Refinement_Module SHALL preserve room polygons from the geometry_output unchanged
4. WHEN refining detections, THE Refinement_Module SHALL replace door polygons with geometry-based detections from `detect_doors_from_walls()`
5. WHEN refining detections, THE Refinement_Module SHALL replace window polygons with geometry-based detections from `detect_windows_from_walls()`
6. THE Refinement_Module SHALL return a new VectorizationResult with refined doors and windows merged with preserved walls and rooms

### Requirement 5: Spatial Relationship Analysis

**User Story:** As a system, I want to analyze spatial relationships between rooms and walls, so that door placement respects architectural constraints.

#### Acceptance Criteria

1. THE Refinement_Module SHALL provide a helper function `find_adjacent_rooms()` that identifies pairs of rooms sharing a wall boundary
2. WHEN identifying adjacent rooms, THE Refinement_Module SHALL use polygon proximity analysis with a threshold of 50 pixels
3. THE Refinement_Module SHALL provide a helper function `find_shared_wall_segment()` that extracts the wall segment between two adjacent rooms
4. THE Refinement_Module SHALL provide a helper function `detect_wall_gaps()` that identifies breaks or discontinuities in wall segments
5. WHEN detecting wall gaps, THE Refinement_Module SHALL use a minimum gap width threshold of 20 pixels
6. THE Refinement_Module SHALL provide a helper function `is_point_on_wall()` that verifies if a point lies within wall polygon boundaries

### Requirement 6: Geometry Pipeline Integration

**User Story:** As a developer, I want refinement integrated into the geometry pipeline, so that improved detections are automatically applied during processing.

#### Acceptance Criteria

1. THE Geometry_Pipeline SHALL provide a configuration option `use_refinement` with default value True
2. WHEN `use_refinement` is True, THE Geometry_Pipeline SHALL call `refine_detections()` after vectorization
3. WHEN calling refinement, THE Geometry_Pipeline SHALL pass the original SegmentationResult, preprocessed image, and VectorizationResult
4. WHEN `use_refinement` is False, THE Geometry_Pipeline SHALL skip refinement and use original YOLO detections
5. THE Geometry_Pipeline SHALL preserve backward compatibility with existing code when refinement is disabled

### Requirement 7: Polygon Creation and Formatting

**User Story:** As a system, I want to create properly formatted polygon representations for detected doors and windows, so that they integrate seamlessly with existing geometry structures.

#### Acceptance Criteria

1. THE Refinement_Module SHALL provide a helper function `create_door_polygon()` that generates a door polygon from a wall segment and position
2. THE Refinement_Module SHALL provide a helper function `create_window_polygon()` that generates a window polygon from parallel line pairs
3. WHEN creating door polygons, THE Refinement_Module SHALL use a standard door width of 90 pixels (approximately 0.9 meters at typical scale)
4. WHEN creating window polygons, THE Refinement_Module SHALL use the distance between parallel lines as window width
5. THE Refinement_Module SHALL return all polygons as NumPy arrays with shape (N, 2) representing pixel coordinates
6. THE Refinement_Module SHALL ensure all polygon coordinates are integer values within image boundaries

### Requirement 8: Wall Direction Analysis

**User Story:** As a system, I want to determine wall orientation, so that doors and windows are aligned correctly with wall geometry.

#### Acceptance Criteria

1. THE Refinement_Module SHALL provide a helper function `get_wall_direction()` that computes the orientation angle of a wall segment
2. WHEN computing wall direction, THE Refinement_Module SHALL use principal component analysis or line fitting on wall polygon points
3. THE Refinement_Module SHALL return wall direction as an angle in degrees (0-180 range)
4. WHEN placing doors, THE Refinement_Module SHALL orient door polygons perpendicular to the wall direction
5. WHEN placing windows, THE Refinement_Module SHALL orient window polygons parallel to the wall direction

### Requirement 9: Configuration and Tunability

**User Story:** As a developer, I want configurable parameters for refinement algorithms, so that I can tune detection behavior for different floor plan styles.

#### Acceptance Criteria

1. THE Refinement_Module SHALL accept configuration parameters for edge detection thresholds (Canny low: 50, high: 150)
2. THE Refinement_Module SHALL accept configuration parameters for line detection (HoughLinesP min_line_length: 30, max_line_gap: 10)
3. THE Refinement_Module SHALL accept configuration parameters for parallel line proximity threshold (default: 20 pixels)
4. THE Refinement_Module SHALL accept configuration parameters for wall gap detection threshold (default: 20 pixels)
5. THE Refinement_Module SHALL accept configuration parameters for room adjacency threshold (default: 50 pixels)
6. WHERE custom configuration is not provided, THE Refinement_Module SHALL use default parameter values

### Requirement 10: Error Handling and Robustness

**User Story:** As a system, I want robust error handling, so that refinement failures do not break the entire pipeline.

#### Acceptance Criteria

1. WHEN no adjacent rooms are found, THE Refinement_Module SHALL return an empty list of door polygons
2. WHEN no valid parallel lines are detected, THE Refinement_Module SHALL return an empty list of window polygons
3. IF wall_polygons input is empty, THEN THE Refinement_Module SHALL return empty detection results without raising exceptions
4. IF room_polygons input is empty, THEN THE Refinement_Module SHALL skip door detection and return empty door list
5. WHEN image input is invalid or None, THE Refinement_Module SHALL skip window detection and return empty window list
6. THE Refinement_Module SHALL log warnings for edge cases (no detections, invalid inputs) without terminating execution

### Requirement 11: Testing and Validation

**User Story:** As a developer, I want comprehensive tests for the refinement module, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE project SHALL include a test file `tests/test_detection.py` for refinement module testing
2. THE test file SHALL include a test `test_detect_doors_from_adjacent_rooms()` that verifies door placement between adjacent rooms
3. THE test file SHALL include a test `test_detect_doors_with_wall_gaps()` that verifies door placement at wall gaps
4. THE test file SHALL include a test `test_detect_windows_from_edges()` that verifies window detection from edge analysis
5. THE test file SHALL include a test `test_refine_detections_preserves_walls()` that verifies walls are unchanged after refinement
6. THE test file SHALL include a test `test_door_placement_on_wall()` that verifies doors lie on wall geometry
7. THE test file SHALL include a test `test_window_alignment_with_wall()` that verifies windows align with wall direction
8. FOR ALL valid VectorizationResult inputs, refining then extracting walls and rooms SHALL produce equivalent wall and room polygons (preservation property)

### Requirement 12: Performance and Efficiency

**User Story:** As a user, I want refinement to execute quickly, so that it does not significantly slow down the overall pipeline.

#### Acceptance Criteria

1. WHEN processing a typical floor plan (1024x1024 pixels), THE Refinement_Module SHALL complete door detection within 500 milliseconds
2. WHEN processing a typical floor plan (1024x1024 pixels), THE Refinement_Module SHALL complete window detection within 1000 milliseconds
3. THE Refinement_Module SHALL use vectorized NumPy operations for geometric computations where possible
4. THE Refinement_Module SHALL avoid nested loops over pixel coordinates
5. WHEN multiple rooms exist, THE Refinement_Module SHALL process room adjacency in O(n²) time complexity or better

### Requirement 13: Documentation and Usage

**User Story:** As a developer, I want clear documentation for the refinement module, so that I can understand how to use and extend it.

#### Acceptance Criteria

1. THE Refinement_Module SHALL include docstrings for all public functions following NumPy documentation style
2. THE Refinement_Module SHALL include usage examples in the module-level docstring
3. THE Refinement_Module SHALL document all function parameters with types and descriptions
4. THE Refinement_Module SHALL document return value formats and data structures
5. THE Refinement_Module SHALL include inline comments explaining complex geometric algorithms
6. THE project SHALL update `ANALYSIS.md` with integration instructions and configuration examples

### Requirement 14: Backward Compatibility

**User Story:** As a developer, I want the refinement feature to be non-breaking, so that existing code continues to work without modifications.

#### Acceptance Criteria

1. THE Refinement_Module SHALL NOT modify the signature of existing functions in other modules
2. THE Refinement_Module SHALL NOT change the structure of VectorizationResult, WallPolygon, or SegmentationResult
3. WHEN refinement is disabled via configuration, THE Geometry_Pipeline SHALL produce identical output to the pre-refinement implementation
4. THE Refinement_Module SHALL be importable as an optional dependency
5. THE project SHALL maintain all existing tests passing after refinement integration
