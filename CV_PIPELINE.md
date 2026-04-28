# CV Pipeline — Technical Documentation

## Project Overview

**Floor2Model** is a computer vision pipeline that converts a 2D architectural floor plan image into a 3D building model. It uses YOLOv8 instance segmentation to detect structural elements (walls, doors, windows), reconstructs their geometry, and exports a fully textured 3D model in OBJ and glTF formats.

---

## Full Pipeline Flow

```
Input Image (PNG/JPG/PDF)
        │
        ▼
[Phase 1] Preprocessing
  load → grayscale → CLAHE contrast → adaptive threshold → deskew → denoise
        │
        ▼
[Phase 2] Segmentation (YOLOv8)
  detect walls / doors / windows → bounding boxes + binary masks + polygons
        │
        ▼
[Phase 3] Geometry Reconstruction
  vectorize masks → estimate scale → build room graph → detection refinement
        │
        ▼
[Phase 4] 3D Reconstruction
  extrude bboxes → cut door/window openings → merge meshes → export OBJ + glTF
        │
        ▼
[Phase 5] GAN Interior Render (separate — see GAN docs)
  ControlNet / Pix2Pix → interior image → MiDaS depth → 3D interior mesh
```

---

## Phase 1 — Preprocessing

**Entry point:** `main.py` → `src/preprocessing/pipeline.py`

### Purpose
Normalises raw floor plan images (scanned, photographed, or digital) into clean, consistent binary images suitable for YOLO segmentation.

### Files

#### `src/preprocessing/loader.py`
Loads images from disk. Supports PNG, JPG, JPEG, BMP, TIFF, PDF (first page via PIL).

**Key function:** `load_image(image_path, target_size=1024)`
- Converts to grayscale
- Resizes so the longer dimension = `target_size` pixels (preserves aspect ratio)
- Uses `INTER_AREA` for downscaling (best for line drawings)

#### `src/preprocessing/binarizer.py`
Converts grayscale to clean binary (black/white) image.

**Pipeline:**
1. Gaussian blur (kernel=5) — suppresses scan noise
2. Adaptive threshold (ADAPTIVE_THRESH_GAUSSIAN_C, block=25, C=10) — handles uneven lighting
3. Morphological close (kernel=3) — fills tiny gaps in wall lines
4. Morphological open (kernel=3) — removes isolated noise specks

**Key functions:**
- `binarize(img, blur_kernel, block_size, c_offset, morph_kernel)` → binary image
- `enhance_contrast(img)` → CLAHE (clipLimit=2.0, tileGrid=8×8)
- `remove_small_components(binary, min_area=100)` → removes components < min_area px²

#### `src/preprocessing/skew_corrector.py`
Detects and corrects rotational skew in scanned floor plans.

**Strategy:**
1. Canny edge detection on binary image
2. Probabilistic Hough Transform to find dominant lines
3. Median angle of near-horizontal lines → skew angle
4. Rotate image to correct (skips if angle < 0.3°)

**Key functions:**
- `detect_skew_angle(binary, angle_threshold=45.0)` → angle in degrees
- `correct_skew(img, angle, background_color=255)` → rotated image
- `deskew(img, binary)` → (corrected_image, angle)

#### `src/preprocessing/pipeline.py`
Orchestrates all preprocessing steps.

**Config:** `PreprocessingConfig`
| Parameter | Default | Description |
|---|---|---|
| target_size | 1024 | Resize longer dimension to this |
| enhance_contrast | True | Apply CLAHE |
| blur_kernel | 5 | Gaussian blur kernel |
| block_size | 25 | Adaptive threshold block size |
| c_offset | 10 | Adaptive threshold C constant |
| morph_kernel | 3 | Morphological ops kernel |
| min_component_area | 150 | Remove components smaller than this (px²) |
| correct_skew | True | Enable skew correction |

**Output:** `PreprocessingResult` with stages: original, grayscale, contrast_enhanced, binary, deskewed, cleaned

**Saved files:**
- `{stem}_1_grayscale.png`
- `{stem}_2_binary.png`
- `{stem}_3_deskewed.png`
- `{stem}_4_cleaned.png`

---

## Phase 2 — Segmentation (YOLOv8)

**Entry point:** `endToEnd2.py` → `src/segmentation/predictor.py`

### Purpose
Runs YOLOv8 instance segmentation on the floor plan image to detect structural elements with bounding boxes, binary masks, and polygon contours.

### Model

**File:** `models/best.pt`

**Architecture:** YOLOv8 segmentation (instance segmentation head)

**Actual detected classes (current model):**
| Class | Description |
|---|---|
| `wall` | Wall segments |
| `door` | Door openings |
| `window` | Window openings |

> Note: The model was trained with 3 classes. The codebase supports up to 13 classes (OuterWall, InnerWall, Window, Door, Stairs, Railing, Kitchen, LivingRoom, Bedroom, Bathroom, Corridor, Balcony, Garage) but the current `best.pt` only outputs the 3 above.

**Confidence sweep:** The pipeline tries confidence thresholds [0.35, 0.25, 0.15, 0.10, 0.05] and uses the first one that produces detections.

### Files

#### `src/segmentation/predictor.py`

**Data structures:**

`DetectedElement`
```python
class_id: int
class_name: str
confidence: float
bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) pixels
mask: np.ndarray                   # binary mask, same size as image
polygon: list[tuple]               # simplified contour points
```

`SegmentationResult`
```python
image_path: str
image_shape: tuple[int, int]       # (H, W)
elements: list[DetectedElement]
# Properties: .walls, .doors, .windows, .rooms
```

**Key class:** `FloorPlanPredictor`
- Lazy-loads YOLO model on first call
- `predict(image_path)` → `SegmentationResult`
- Uses `retina_masks=True` for higher quality masks
- Converts masks to simplified polygons via Douglas-Peucker (epsilon_factor=0.005)

#### `src/segmentation/visualizer.py`
Draws detection results on images for debugging.

**Features:**
- Semi-transparent filled masks (alpha=0.40)
- Bounding box rectangles
- Class name + confidence labels
- Contour outlines (thicker for walls)
- Legend in bottom-left corner

**Color palette (BGR):**
| Class | Color |
|---|---|
| OuterWall | Dark blue (50, 50, 180) |
| InnerWall | Medium blue (100, 100, 220) |
| Window | Amber (255, 200, 50) |
| Door | Teal (50, 200, 200) |
| Kitchen | Green (50, 180, 80) |
| Bedroom | Pink (200, 100, 150) |

#### `src/segmentation/trainer.py`
YOLOv8 training wrapper (used for training, not inference).

**Device priority:** MPS (Apple Silicon) > CUDA (if sm compatible) > CPU

**Training config:**
- Model: YOLOv8s-seg (11.8M params, recommended)
- Augmentation: rotation ±10°, translation, scale, horizontal flip
- Optimizer: AdamW, lr=0.001
- Early stopping: patience=30

---

## Phase 3 — Geometry Reconstruction

**Entry point:** `endToEnd2.py` → `src/geometry/pipeline.py`

### Purpose
Converts raw YOLO detections into clean 2D vector geometry: wall polygons, scale estimate, and room connectivity graph.

### Files

#### `src/geometry/wall_vectorizer.py`
Converts binary segmentation masks into clean 2D polygons.

**Pipeline per mask:**
1. Binarize mask (threshold at 127)
2. Morphological close + open (kernel=3) — cleanup
3. Find external contours (RETR_EXTERNAL)
4. Douglas-Peucker simplification (epsilon_factor=0.008, ×1.5 for walls)
5. Filter by area (min_area=200 px²)
6. Sort by area descending

**Data structures:**

`WallPolygon`
```python
class_id: int
class_name: str
points: list[tuple[int, int]]      # pixel coordinates (x, y)
area: float                         # pixel area
bbox: tuple[int, int, int, int]    # (x, y, w, h)
confidence: float
# Properties: .is_wall, .is_room, .centroid
```

`VectorizationResult`
```python
walls: list[WallPolygon]
rooms: list[WallPolygon]
doors: list[WallPolygon]
windows: list[WallPolygon]
other: list[WallPolygon]
image_shape: tuple[int, int]
# Property: .all_polygons
```

**Class ID mapping:**
| IDs | Category |
|---|---|
| 0, 1 | Walls (OuterWall, InnerWall) |
| 2 | Windows |
| 3 | Doors |
| 6-12 | Rooms (Kitchen, LivingRoom, Bedroom, Bathroom, Corridor, Balcony, Garage) |

#### `src/geometry/scale_estimator.py`
Estimates pixel-to-metre conversion factor.

**Strategies (priority order):**
1. **OCR** (confidence=0.75) — detects dimension annotations like "4.5m", "450cm", "3500mm" using pytesseract. Assumes reference dimension spans ~40% of image.
2. **Room size heuristics** (confidence=0.55) — compares detected room pixel areas to known priors (Bedroom: 9-20m², LivingRoom: 15-40m², etc.)
3. **Fallback** (confidence=0.30) — assumes building footprint ≈ 12m on longer axis

**Current behaviour:** Almost always falls back to strategy 3 since the model doesn't detect room types.

**Output:** `ScaleEstimate`
```python
pixels_per_metre: float
confidence: float        # 0-1
method: str              # 'ocr', 'room_size', 'fallback'
notes: str
```

#### `src/geometry/room_graph.py`
Builds a room connectivity graph.

**Nodes:** `RoomNode` — room type, centroid, area in m², polygon
**Edges:** `RoomEdge` — connection type ('door', 'shared_wall'), distance

**Connection strategy:**
1. Door proximity — if a door centroid is within `door_proximity` (150px) of two rooms, connect them with a 'door' edge
2. Centroid proximity — rooms within `proximity_threshold` (200px) get a 'shared_wall' edge

**Current behaviour:** Produces empty graph (0 nodes, 0 edges) since the model doesn't detect room types.

#### `src/detection/refinement.py`
Geometry-based refinement of door and window detections.

**Classes:**
- `DetectionRefiner` — orchestrates refinement
- `SpatialAnalyzer` — finds adjacent rooms, shared wall segments, wall directions (uses PCA via sklearn)
- `GapDetector` — finds discontinuities in wall segments
- `DoorDetector` — places doors at wall gaps between adjacent rooms
- `WindowDetector` — finds parallel line pairs (Canny + HoughLinesP) aligned with walls
- `EdgeAnalyzer` — Canny edge detection wrapper
- `LineDetector` — HoughLinesP wrapper

**Config:** `RefinementConfig`
| Parameter | Default | Description |
|---|---|---|
| canny_low_threshold | 50 | Canny low threshold |
| canny_high_threshold | 150 | Canny high threshold |
| hough_min_line_length | 30 | Min line length (px) |
| parallel_line_proximity | 20 | Max distance between parallel lines (px) |
| door_width | 90 | Standard door width (px, ~0.9m) |
| wall_gap_threshold | 20 | Min gap width to consider as door (px) |

**Known issue:** `sklearn` is not installed → PCA-based wall direction fails silently, falls back to simple line fitting. Door detection skips when no room polygons exist.

#### `src/geometry/pipeline.py`
Orchestrates Phase 3.

**Config:** `GeometryConfig`
| Parameter | Default | Description |
|---|---|---|
| epsilon_factor | 0.008 | Polygon simplification |
| min_area | 200 | Min polygon area (px²) |
| proximity_threshold | 200 | Room adjacency threshold (px) |
| use_refinement | True | Enable detection refinement |
| debug_visualization | False | Save before/after comparison images |

**Output:** `GeometryResult`
```python
source_path: str
vectorization: VectorizationResult
scale: ScaleEstimate
graph: FloorPlanGraph
config: GeometryConfig
segmentation_result: SegmentationResult   # raw detections, passed to Phase 4
```

**Saved files:**
- `{stem}_polygons.png` — vectorized polygon overlay
- `{stem}_graph.png` — room connectivity graph overlay
- `{stem}_graph.json` — room graph as JSON
- `{stem}_scale.json` — scale estimate as JSON

---

## Phase 4 — 3D Reconstruction

**Entry point:** `endToEnd2.py` → `src/reconstruction/pipeline.py`

### Purpose
Extrudes 2D detection bounding boxes into 3D meshes, cuts door/window openings into walls, merges everything into a single building model, and exports to OBJ + glTF.

### Key Design Decisions

**Bbox-based extrusion (not polygon-based):** When raw detections are available, walls and doors are generated from their bounding boxes rather than mask polygons. This gives more consistent geometry since YOLO bboxes are clean rectangles.

**Global coordinate origin:** All meshes share a single origin (centroid of all detection points) so they stay in correct relative positions matching the floorplan layout.

**Wall splitting:** Each wall bbox is checked against overlapping door/window bboxes. Where they overlap, the wall is split into segments with the opening cut out.

### Files

#### `src/reconstruction/extruder.py`

**Realistic heights (metres):**
| Element | Height |
|---|---|
| OuterWall | 3.0m |
| InnerWall / wall | 2.7-2.8m |
| Door | 2.1m |
| Window | 1.2m (sill at 0.9m) |
| Stairs | 3.0m |
| Garage | 2.4m |
| Rooms (floor slab) | 0.15m |

**Realistic wall thickness:**
| Type | Thickness |
|---|---|
| OuterWall | 0.25m |
| InnerWall | 0.15m |
| wall (generic) | 0.20m |

**Key methods:**

`build_wall_with_openings(wall_bbox_px, openings, label)` → `List[Mesh3D]`
- Determines wall orientation (horizontal if width ≥ height)
- Projects each opening onto the wall's long axis
- Splits wall into segments: solid | sill | opening | lintel | solid
- For doors: gap in wall + thin door panel (5cm, brown)
- For windows: sill (0→0.9m) + glass pane (0.9→2.1m, 4cm thick) + lintel (2.1→wall_height)

`extrude_bbox_wall(bbox_px, label)` → `Mesh3D`
- Simple solid wall box for walls with no openings
- Enforces minimum thickness from WALL_THICKNESS_M

`extrude_standalone_window(bbox_px)` → `Mesh3D`
- Window not overlapping any wall — floats at sill height (0.9m)

`extrude_polygon(points_px, label, height, base_z)` → `Mesh3D`
- Core extrusion: bottom + top vertices, side faces, vertex normals

`_px_to_metres(points_px)` → `np.ndarray`
- Divides by pixels_per_metre
- Subtracts global origin (shared across all meshes)
- Flips Y axis (image top → 3D front)

#### `src/reconstruction/mesh_builder.py`

`MeshBuilder.build(meshes)` → `Building3D`
- Merges all Mesh3D objects into one
- Offsets face indices correctly
- Assigns per-vertex colors from mesh.color

`Building3D`
```python
meshes: List[Mesh3D]
merged_vertices: np.ndarray    # (N, 3) float32
merged_faces: List[List[int]]
merged_colors: np.ndarray      # (N, 3) float32 per-vertex RGB
wall_height: float
floor_area_m2: float
room_count: int
```

#### `src/reconstruction/exporter.py`

**OBJ export** (`_write_obj`):
- Writes `.obj` + `.mtl` files
- Ground plane quad with render texture (or floorplan texture as fallback)
- Each mesh as a named object with its material
- Copies `_floorplan.png` and `_render.png` next to the OBJ

**glTF 2.0 export** (`_write_gltf`):
- Self-contained (all data base64-encoded in the JSON)
- Three mesh nodes:
  1. `InteriorFloor` — ground plane with render texture (GAN output)
  2. `FloorPlan` — ceiling plane with floorplan texture (top-down view)
  3. `{stem}` — building geometry with per-vertex colors
- PBR materials (metallicFactor=0, roughnessFactor=0.8-0.9)
- `doubleSided: true` on textured planes

**STL export** (`_write_stl`):
- Binary STL with computed face normals
- Disabled by default

#### `src/reconstruction/pipeline.py`

**Config:** `ReconstructionConfig`
| Parameter | Default | Description |
|---|---|---|
| wall_height | 2.8m | Default wall height |
| floor_thickness | 0.2m | Floor slab thickness |
| add_floor | True | Add floor slab from outer wall footprint |
| export_obj | True | Export OBJ |
| export_gltf | True | Export glTF |
| export_stl | False | Export STL |

**Pipeline steps:**
1. Compute global origin from all detection points
2. Separate detections into walls, doors, windows
3. Match each door/window to its overlapping wall (by bbox intersection area)
4. Extrude walls with openings cut in (or solid walls if no openings)
5. Handle unassigned doors/windows as standalone elements
6. Merge all meshes into Building3D
7. Export to OBJ + glTF

---

## Main Entry Point — `endToEnd2.py`

Processes all images in `samples/` folder. Supports PNG, JPG, JPEG, BMP, TIFF, TIF.

**Pipeline phases per image:**
1. **Phase 2** — YOLO detection with confidence sweep [0.35, 0.25, 0.15, 0.10, 0.05]
2. **Phase 3** — Geometry reconstruction + vectorization
3. **Phase 5** — ControlNet/GAN interior render (runs BEFORE Phase 4)
4. **Phase 4** — 3D reconstruction with render embedded in glTF

**Output directory:** `generated_models/{stem}/`

---

## Output Files

For each input image `samples/{stem}.png`:

```
generated_models/{stem}/
├── {stem}_detections.png     # YOLO detections drawn on original image
├── {stem}_floorplan.png      # Clean floorplan (used as ceiling texture)
├── {stem}_render.png         # GAN-generated interior render
├── {stem}.obj                # 3D model (Wavefront OBJ)
├── {stem}.mtl                # Materials for OBJ
└── {stem}.gltf               # 3D model (glTF 2.0, self-contained)
```

**How to view:**
- `.gltf` → drag to https://gltf-viewer.donmccurdy.com
- `.obj` → open in Blender, MeshLab, or Windows 3D Viewer

---

## Configuration Files

**`configs/segmentation.yaml`** — YOLO dataset configuration for training

---

## Known Limitations

| Issue | Impact | Status |
|---|---|---|
| Model only detects 3 classes (wall/door/window) | No room type labels → scale estimation always uses fallback, room graph always empty | Current model limitation |
| `sklearn` not installed | PCA-based wall direction fails → door detection degrades | Non-critical, fallback works |
| Scale estimation uses fallback (30% confidence) | Scale may be inaccurate for unusual floorplan sizes | Acceptable for demo |
| RTX 5060 not supported by PyTorch stable | GPU acceleration unavailable, all inference on CPU | PyTorch nightly needed |
| No roof generation | Building has no roof (intentional design decision) | By design |
| Window refinement produces many false positives | `Refined Windows: 23+` from Hough lines on wall textures | Known issue |

---

## Dependencies

**Core CV:**
- `ultralytics` — YOLOv8 inference and training
- `opencv-python` — image processing
- `numpy` — array operations
- `torch`, `torchvision` — PyTorch (CPU mode)

**Optional:**
- `pytesseract` — OCR for scale detection from dimension annotations
- `sklearn` — PCA for wall direction in refinement

**3D export:**
- Pure Python (json, struct, base64) — no external 3D library needed

---

## Module Structure

```
src/
├── preprocessing/
│   ├── loader.py          # Image loading + resizing
│   ├── binarizer.py       # Grayscale → binary + CLAHE
│   ├── skew_corrector.py  # Hough-based deskew
│   └── pipeline.py        # Phase 1 orchestrator
├── segmentation/
│   ├── predictor.py       # YOLOv8 inference wrapper
│   ├── visualizer.py      # Detection overlay drawing
│   └── trainer.py         # YOLOv8 training wrapper
├── geometry/
│   ├── wall_vectorizer.py # Mask → polygon conversion
│   ├── scale_estimator.py # Pixel-to-metre scale
│   ├── room_graph.py      # Room connectivity graph
│   └── pipeline.py        # Phase 3 orchestrator
├── detection/
│   └── refinement.py      # Geometry-based door/window refinement
├── reconstruction/
│   ├── extruder.py        # 2D bbox/polygon → 3D mesh
│   ├── mesh_builder.py    # Merge meshes → Building3D
│   ├── exporter.py        # OBJ + glTF export
│   └── pipeline.py        # Phase 4 orchestrator
└── utils/
    └── visualization.py   # Debug visualization helpers
```
