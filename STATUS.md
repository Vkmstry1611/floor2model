# floor2model — Project Status

## What Is This

A Python pipeline that converts a 2D floor plan image into a 3D building model (OBJ / glTF / STL).

---

## Pipeline Overview

```
Image → Phase 1: Preprocessing → Phase 2: Segmentation (YOLO) → Phase 3: Geometry → Phase 4: 3D Reconstruction
```

---

## What's Implemented

### Phase 1 — Preprocessing (`src/preprocessing/`)
- `loader.py` — Load images (PNG/JPG/BMP/TIFF/PDF), resize keeping aspect ratio
- `binarizer.py` — CLAHE contrast enhancement, adaptive thresholding, morphological cleanup, small component removal
- `skew_corrector.py` — Hough-line skew detection and rotation correction
- `pipeline.py` — Orchestrates all steps, saves intermediate outputs (grayscale, binary, deskewed, cleaned)

### Phase 2 — Segmentation (`src/segmentation/`)
- `predictor.py` — YOLOv8 inference, lazy model loading, mask-to-polygon conversion, structured `SegmentationResult`
- `trainer.py` — YOLOv8 training wrapper with device auto-detection (MPS / CUDA / CPU), augmentation config, export to ONNX/CoreML
- `dataset.py` — CubiCasa5k SVG annotation parser, YOLO label writer
- `visualizer.py` — Draws segmentation masks and bounding boxes on images

**Class mapping (14 classes):**
`OuterWall, InnerWall, Window, Door, Stairs, Railing, Kitchen, LivingRoom, Bedroom, Bathroom, Corridor, Balcony, Garage`

### Phase 3 — Geometry (`src/geometry/`)
- `wall_vectorizer.py` — Converts YOLO masks → simplified 2D polygons (Douglas-Peucker), categorises into walls/rooms/doors/windows
- `scale_estimator.py` — Pixel-to-metre scale via OCR → room-size heuristics → fallback (3 strategies)
- `room_graph.py` — Builds room connectivity graph (nodes = rooms, edges = door/shared-wall connections)
- `pipeline.py` — Orchestrates vectorization → scale → graph, supports debug visualizations, saves JSON outputs

### Detection Refinement (`src/detection/`)
- `refinement.py` — Geometry-based door/window detection to improve on raw YOLO results:
  - `SpatialAnalyzer` — finds adjacent rooms, shared wall segments, wall direction via PCA
  - `GapDetector` — detects gaps/discontinuities in wall segments
  - `DoorDetector` — places doors at wall gaps between adjacent rooms
  - `EdgeAnalyzer` — Canny edge detection
  - `LineDetector` — HoughLinesP line extraction, parallel pair detection
  - `WindowDetector` — detects windows from parallel line pairs on walls
  - `RefinementConfig` — fully validated configuration dataclass

### Phase 4 — 3D Reconstruction (`src/reconstruction/`)
- `extruder.py` — Extrudes 2D polygons into 3D meshes (walls, rooms, floor slab), computes vertex normals, per-class colours
- `mesh_builder.py` — Merges `Mesh3D` list into unified `Building3D` with merged vertices/faces/colours
- `exporter.py` — Exports `Building3D` to:
  - **OBJ + MTL** — Wavefront format with per-material colours
  - **glTF 2.0** — JSON with embedded base64 buffers, vertex colours, valid accessor/bufferView structure
  - **Binary STL** — with computed face normals
- `pipeline.py` — Orchestrates extrude → merge → export, wires to `GeometryResult`

### Utilities (`src/utils/`)
- `visualization.py` — `visualize_detections`, `visualize_comparison`, `visualize_vectorization_result`, `create_detection_report`, legend/title overlays

### Tests (`tests/`)
- `test_preprocessing.py` — Binarizer, skew corrector, loader helpers
- `test_geometry.py` — WallVectorizer, ScaleEstimator, RoomGraphBuilder
- `test_segmentation.py` — Dataset parser, SegmentationResult filters, Visualizer
- `test_reconstruction.py` — Extruder, MeshBuilder, ModelExporter (OBJ/glTF/STL)

### Entry Points
- `main.py` — CLI for Phase 1 preprocessing only
- `endToEnd2.py` — Full pipeline Phase 1→4 on a sample image
- `run_with_refinement.py` — Full pipeline with detection refinement enabled
- `example_with_visualization.py` — Demonstrates all visualization and refinement config options

---

## Running the Pipeline

### Quick start (full pipeline)
```bash
cd floor2model
python endToEnd2.py
```

### Preprocessing only
```bash
python main.py --input samples/floorplan1.png --output outputs/
```

### With refinement
```bash
python run_with_refinement.py samples/floorplan1.png
```

### Run tests
```bash
pytest tests/ -v
```

---

## Things Left To Do / Known Gaps

| Area | Status | Notes |
|------|--------|-------|
| YOLO model weights | Not included | `src/segmentation/best.pt` must be trained or downloaded. Training notebook in `notebooks/` |
| OCR scale detection | Partial | Works when pytesseract is installed; floor plans without dimension text fall back to room-size heuristics |
| Window detection accuracy | Moderate | Parallel-line approach works on clean plans; noisy scans may produce false positives |
| Door detection accuracy | Moderate | Relies on room adjacency; works best when YOLO detects rooms correctly |
| glTF binary (GLB) | Not implemented | Currently exports glTF JSON only; GLB (single binary file) would be a useful addition |
| Texture/UV mapping | Not implemented | Models use vertex colours only; UV-mapped textures would improve visual quality |
| Multi-storey support | Not implemented | Pipeline assumes single floor; stacked floors would require storey detection |
| Web viewer | Not implemented | A simple Three.js viewer for the glTF output would be useful |
| CI / GitHub Actions | Not set up | No automated test runner configured |

---

## Dependencies

```
opencv-python-headless>=4.8.0
numpy>=1.23.0,<2.0.0
pillow>=8.0.0
matplotlib>=3.5.0
pytesseract>=0.3.10      # optional — for OCR scale detection
scipy>=1.7.0
scikit-learn>=1.0.0      # for PCA in refinement
ultralytics>=8.0.0       # for YOLO inference/training
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.60.0
shapely>=2.0.0
networkx>=3.0
```

Install:
```bash
pip install -r requirements.txt
```
