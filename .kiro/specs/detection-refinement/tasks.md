# Implementation Plan: Detection Refinement Module

## Overview

This plan implements a geometry-based detection refinement system for doors and windows in 2D floor plans. The module addresses YOLOv8's weakness in detecting small architectural elements by leveraging wall geometry, spatial relationships, and computer vision techniques. It will be integrated as an optional post-processing step between Phase 2 (Segmentation) and Phase 3 (Geometry) of the floor2model pipeline.

The implementation uses Python with OpenCV and NumPy, following the existing codebase patterns and data structures.

## Tasks

- [x] 1. Set up module structure and configuration
  - Create `src/detection/` package directory
  - Create `src/detection/__init__.py` with package exports
  - Implement `RefinementConfig` dataclass with all tunable parameters
  - Add validation logic in `__post_init__` to clamp parameter values
  - _Requirements: 1.1, 9.1-9.6_

- [ ] 2. Implement spatial analysis components
  - [x] 2.1 Implement `SpatialAnalyzer` class
    - Write `find_adjacent_rooms()` method using polygon proximity analysis
    - Write `find_shared_wall_segment()` method to extract wall between rooms
    - Write `is_point_on_wall()` method using `cv2.pointPolygonTest`
    - Write `get_wall_direction()` method using PCA for orientation
    - _Requirements: 5.1-5.6, 8.1-8.3_
  
  - [ ]* 2.2 Write property test for spatial analysis
    - **Property 2: Shared Wall Identification**
    - **Validates: Requirements 2.2**
  
  - [ ]* 2.3 Write unit tests for `SpatialAnalyzer`
    - Test adjacent room detection with various room configurations
    - Test shared wall segment extraction
    - Test point-on-wall validation
    - Test wall direction computation
    - _Requirements: 11.1-11.7_

- [ ] 3. Implement gap detection for doors
  - [x] 3.1 Implement `GapDetector` class
    - Write `detect_wall_gaps()` method to identify discontinuities
    - Sort wall points along wall direction
    - Compute distances between consecutive points
    - Identify gaps where distance exceeds threshold
    - _Requirements: 5.5_
  
  - [ ]* 3.2 Write property test for gap detection
    - **Property 3: Door Placement at Wall Gaps**
    - **Validates: Requirements 2.3**
  
  - [ ]* 3.3 Write unit tests for `GapDetector`
    - Test gap detection with known wall discontinuities
    - Test with continuous walls (no gaps)
    - Test edge cases (empty walls, single point)
    - _Requirements: 11.3_

- [ ] 4. Implement door detection logic
  - [x] 4.1 Implement `DoorDetector` class
    - Initialize with `SpatialAnalyzer` and `GapDetector` instances
    - Write `detect()` method that orchestrates door placement
    - Find adjacent room pairs using spatial analyzer
    - Extract shared wall segments
    - Detect gaps and place doors at gap locations
    - Implement fallback: place doors at midpoints when no gaps found
    - _Requirements: 2.1-2.7_
  
  - [x] 4.2 Implement `create_door_polygon()` helper function
    - Create door polygon at specified position
    - Orient door perpendicular to wall direction
    - Use standard door width from config
    - Return as NumPy array with shape (N, 2)
    - _Requirements: 7.1, 7.3, 7.5, 7.6_
  
  - [ ]* 4.3 Write property tests for door detection
    - **Property 4: Door Placement Fallback**
    - **Validates: Requirements 2.4**
    - **Property 6: Door-Room Connectivity**
    - **Validates: Requirements 2.6**
    - **Property 7: Door Orientation Perpendicularity**
    - **Validates: Requirements 2.7**
  
  - [ ]* 4.4 Write unit tests for `DoorDetector`
    - Test door placement between adjacent rooms
    - Test door placement at wall gaps
    - Test fallback to midpoint placement
    - Test door orientation alignment
    - _Requirements: 11.2, 11.6_

- [ ] 5. Checkpoint - Ensure door detection tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement edge and line detection for windows
  - [x] 6.1 Implement `EdgeAnalyzer` class
    - Write `detect_edges()` method using Canny edge detection
    - Use config parameters for low/high thresholds
    - Return binary edge map
    - _Requirements: 3.2_
  
  - [x] 6.2 Implement `LineDetector` class
    - Write `detect_lines()` method using HoughLinesP
    - Use config parameters for line detection
    - Write `compute_line_angle()` method for line orientation
    - Write `compute_line_distance()` method for perpendicular distance
    - Write `find_parallel_pairs()` method to identify parallel lines
    - _Requirements: 3.3, 3.4_
  
  - [ ]* 6.3 Write property test for line detection
    - **Property 8: Parallel Line Identification**
    - **Validates: Requirements 3.4**
  
  - [ ]* 6.4 Write unit tests for edge and line detection
    - Test Canny edge detection on sample images
    - Test HoughLinesP line extraction
    - Test parallel line pair identification
    - Test line angle and distance computations
    - _Requirements: 11.4_

- [ ] 7. Implement window detection logic
  - [x] 7.1 Implement `WindowDetector` class
    - Initialize with `EdgeAnalyzer` and `LineDetector` instances
    - Write `detect()` method that orchestrates window detection
    - Apply edge detection to input image
    - Extract line segments from edges
    - Find parallel line pairs
    - Verify lines align with wall direction
    - Verify lines lie within wall boundaries
    - _Requirements: 3.1-3.7_
  
  - [x] 7.2 Implement `create_window_polygon()` helper function
    - Create window polygon from parallel line pair
    - Use distance between lines as window width
    - Align window with wall direction
    - Center window on wall segment
    - Return as NumPy array with shape (N, 2)
    - _Requirements: 7.2, 7.4, 7.5, 7.6_
  
  - [ ]* 7.3 Write property tests for window detection
    - **Property 9: Window Alignment with Walls**
    - **Validates: Requirements 3.5, 3.7**
    - **Property 10: Window Centering**
    - **Validates: Requirements 2.8**
  
  - [ ]* 7.4 Write unit tests for `WindowDetector`
    - Test window detection from parallel edges
    - Test window alignment with walls
    - Test window centering on wall segments
    - _Requirements: 11.4, 11.7_

- [ ] 8. Checkpoint - Ensure window detection tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement main refinement orchestrator
  - [ ] 9.1 Implement `DetectionRefiner` class
    - Initialize with `DoorDetector` and `WindowDetector` instances
    - Write `refine()` method that orchestrates full refinement
    - Preserve walls and rooms from geometry output
    - Call door detector if enabled in config
    - Call window detector if enabled in config
    - Merge refined detections with preserved elements
    - Return new `VectorizationResult`
    - _Requirements: 4.1-4.6_
  
  - [ ] 9.2 Implement `refine_detections()` public API function
    - Create `DetectionRefiner` instance with config
    - Call refiner.refine() with inputs
    - Return refined `VectorizationResult`
    - _Requirements: 4.1_
  
  - [ ]* 9.3 Write property tests for refinement orchestrator
    - **Property 1: Preservation of Walls and Rooms**
    - **Validates: Requirements 4.2, 4.3, 11.8**
    - **Property 5: Element Placement Validity**
    - **Validates: Requirements 2.5, 3.6**
    - **Property 11: Door and Window Replacement**
    - **Validates: Requirements 4.4, 4.5**
    - **Property 12: Polygon Format Validity**
    - **Validates: Requirements 7.5, 7.6**
    - **Property 13: Wall Direction Angle Range**
    - **Validates: Requirements 8.3**
  
  - [ ]* 9.4 Write unit tests for refinement orchestrator
    - Test full refinement with valid inputs
    - Test preservation of walls and rooms
    - Test replacement of doors and windows
    - _Requirements: 11.5_

- [ ] 10. Implement error handling and robustness
  - [x] 10.1 Add input validation to all detection methods
    - Check for empty wall_polygons and return empty lists
    - Check for empty room_polygons and skip door detection
    - Check for invalid images and skip window detection
    - _Requirements: 10.1-10.5_
  
  - [x] 10.2 Add try-except blocks for graceful degradation
    - Wrap door detection in try-except, fallback to original
    - Wrap window detection in try-except, fallback to original
    - Wrap geometric computations in try-except with defaults
    - Wrap OpenCV operations in try-except with error logging
    - _Requirements: 10.1-10.6_
  
  - [x] 10.3 Add logging throughout the module
    - Add logger initialization at module level
    - Log INFO messages for normal operations
    - Log WARNING messages for fallback strategies
    - Log ERROR messages for failures with context
    - _Requirements: 10.6_
  
  - [ ]* 10.4 Write unit tests for error handling
    - Test with empty wall polygons
    - Test with empty room polygons
    - Test with invalid image (None)
    - Test graceful degradation on failures
    - _Requirements: 11.1-11.7_

- [ ] 11. Checkpoint - Ensure error handling tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Integrate with GeometryPipeline
  - [x] 12.1 Modify `GeometryConfig` dataclass
    - Add `use_refinement: bool = True` field
    - Add `refinement_config: Optional[RefinementConfig] = None` field
    - Add `__post_init__` to initialize default refinement config
    - _Requirements: 6.1_
  
  - [x] 12.2 Modify `GeometryPipeline.run()` method
    - After vectorization, check if `use_refinement` is enabled
    - If enabled, import and call `refine_detections()`
    - Pass segmentation result, image, and vectorization result
    - Replace vectorization result with refined result
    - _Requirements: 6.2, 6.3_
  
  - [ ]* 12.3 Write integration tests
    - Test pipeline with refinement enabled
    - Test pipeline with refinement disabled
    - Verify backward compatibility
    - _Requirements: 6.4, 6.5, 14.3_

- [ ] 13. Add documentation and examples
  - [x] 13.1 Add docstrings to all public functions and classes
    - Use NumPy documentation style
    - Document all parameters with types
    - Document return values and formats
    - Add usage examples to module docstring
    - _Requirements: 13.1-13.4_
  
  - [x] 13.2 Add inline comments for complex algorithms
    - Comment geometric computation logic
    - Comment edge detection and line detection steps
    - Comment spatial analysis algorithms
    - _Requirements: 13.5_
  
  - [x] 13.3 Update ANALYSIS.md with integration instructions
    - Document how to enable/disable refinement
    - Document configuration parameters
    - Add usage examples
    - _Requirements: 13.6_

- [ ] 14. Performance optimization and validation
  - [x] 14.1 Optimize spatial computations
    - Use NumPy vectorized operations for distance calculations
    - Cache wall directions and centroids
    - Add early termination for empty inputs
    - _Requirements: 12.3, 12.4_
  
  - [ ]* 14.2 Write performance tests
    - Test door detection completes within 500ms for 1024x1024 images
    - Test window detection completes within 1000ms for 1024x1024 images
    - _Requirements: 12.1, 12.2_
  
  - [x] 14.3 Run end-to-end validation
    - Test with sample floor plans from `samples/` directory
    - Verify refined detections are reasonable
    - Compare with/without refinement
    - _Requirements: 14.1-14.5_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties across randomized inputs
- Unit tests validate specific examples and edge cases
- The module is designed to be non-invasive and backward compatible
- All error handling follows fail-safe approach (never break the pipeline)
