# Floor2Model — CV Pipeline

Converts a 2D architectural floor plan image into a 3D building model (`.gltf` / `.obj`) using YOLOv8 instance segmentation and geometric extrusion.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full CV pipeline on all images in samples/
python cv_pipeline.py

# Or run CV + GAN together
python main.py
```

Outputs are saved to `generated_models/<stem>/` for each input image.

---

## What It Does

```
Input image (PNG / JPG)
        │
        ▼
Phase 2 — YOLOv8 Segmentation
  Detects: wall · door · window
  Output:  bounding boxes + binary masks
        │
        ▼
Phase 3 — Geometry Reconstruction
  Vectorizes masks → 2D polygons
  Estimates pixel-to-metre scale
  Builds room connectivity graph
        │
        ▼
Phase 4 — 3D Reconstruction
  Extrudes bboxes → 3D wall/door/window meshes
  Cuts door gaps + window openings in walls
  Exports .gltf + .obj
        │
        ▼
Phase 5 — Interior Render (ControlNet, optional)
  Generates a photorealistic interior image
  Embeds it as floor texture in the .gltf
```

---

## Pipeline Phases

### Phase 2 — Segmentation (`src/segmentation/`)

**Model:** `models/best.pt` — YOLOv8 segmentation, 3 classes: `wall`, `door`, `window`

**Confidence sweep:** tries [0.35, 0.25, 0.15, 0.10, 0.05] and uses the first threshold that produces detections.

**Output per detection:**
- Bounding box `(x1, y1, x2, y2)` in pixels
- Binary mask (same size as image)
- Simplified polygon contour

**Key files:**
- `src/segmentation/predictor.py` — `FloorPlanPredictor` class, runs YOLO inference
- `src/segmentation/visualizer.py` — draws detection overlays on images

---

### Phase 3 — Geometry (`src/geometry/`)

**Step 1 — Vectorization** (`wall_vectorizer.py`)
Converts binary masks → clean 2D polygons via:
1. Morphological cleanup (close gaps, remove noise)
2. Contour extraction
3. Douglas-Peucker simplification (epsilon=0.008)
4. Area filter (min 200 px²)

**Step 2 — Scale Estimation** (`scale_estimator.py`)
Priority order:
1. OCR — reads dimension annotations (e.g. "4.5m") from the image
2. Room size priors — compares room pixel areas to known real-world sizes
3. Wall thickness — uses median wall bbox thickness (assumes 0.15m standard)
4. Wall extents — uses span of all detected walls (assumes ~10m building)
5. Fallback — assumes 12m building at full image size

**Step 3 — Room Graph** (`room_graph.py`)
Builds a connectivity graph of rooms (nodes) connected via doors (edges).
Currently produces empty graph since the model only detects wall/door/window, not room types.

**Step 4 — Detection Refinement** (`src/detection/refinement.py`)
Geometry-based refinement of door/window positions using Canny edges + HoughLinesP.
Requires `sklearn` for PCA-based wall direction (currently falls back gracefully).

**Key file:** `src/geometry/pipeline.py` — `GeometryPipeline` orchestrates all steps.

---

### Phase 4 — 3D Reconstruction (`src/reconstruction/`)

**Wall extrusion** (`extruder.py`)
- Each wall bbox → 3D box with realistic thickness (0.12m generic, 0.15m inner, 0.15m outer)
- Walls with overlapping door/window bboxes are split into segments with openings cut out
- **Doors:** gap in wall (0→2.1m) + thin door panel (5cm, brown)
- **Windows:** sill (0→0.9m) + glass pane (0.9→2.1m, 4cm, blue) + lintel (2.1→2.8m)
- All meshes share a single global coordinate origin (centroid of all detections)

**Realistic heights:**
| Element | Height |
|---|---|
| Wall | 2.8m |
| Door | 2.1m |
| Window glass | 1.2m (sill at 0.9m) |

**Export** (`exporter.py`)
- `.gltf` — self-contained (all data base64-encoded), opens in any glTF viewer
- `.obj` + `.mtl` — standard 3D format, opens in Blender / MeshLab
- Ground plane with floorplan image as texture

**Key file:** `src/reconstruction/pipeline.py` — `ReconstructionPipeline` orchestrates extrusion + export.

---

### Phase 5 — Interior Render (optional)

Uses ControlNet (Stable Diffusion) to generate a photorealistic interior image from the segmentation map. Embeds it as the floor texture in the `.gltf`.

Runs automatically if `diffusers` is installed. Skips gracefully if not.

---

## Output Files

For each `samples/<stem>.<ext>`:

```
generated_models/<stem>/
├── <stem>_detections.png   # YOLO detections drawn on original image
├── <stem>_floorplan.png    # Clean floorplan (used as ceiling texture)
├── <stem>_render.png       # ControlNet interior render (if Phase 5 ran)
├── <stem>.gltf             # 3D model — drag to gltf-viewer.donmccurdy.com
├── <stem>.obj              # 3D model — open in Blender
└── <stem>.mtl              # Materials for OBJ
```

**View the 3D model:** drag `<stem>.gltf` to [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com)

---

## Configuration

### `endToEnd2.py` — main settings

```python
MODEL_PATH    = "models/best.pt"       # YOLO model weights
OUTPUTS_DIR   = "outputs/"             # intermediate outputs
GENERATED_DIR = "generated_models/"    # final 3D models

BACKEND = "controlnet"                 # Phase 5 backend: "controlnet" | "gan"
GAN_CHECKPOINT = None                  # path to final.pt when GAN is trained
```

### Supported input formats
PNG, JPG, JPEG, BMP, TIFF, TIF (all files in `samples/` are processed)

---

## Project Structure

```
floor2model/
├── main.py                   ← COMBINED entry point (CV + GAN)
├── cv_pipeline.py            ← CV only entry point
├── samples/                  ← input floor plan images
├── models/
│   └── best.pt               ← trained YOLO weights
├── generated_models/         ← CV pipeline outputs
├── src/
│   ├── preprocessing/        ← Phase 1: image cleanup
│   ├── segmentation/         ← Phase 2: YOLO detection
│   ├── geometry/             ← Phase 3: vectorization + scale
│   ├── detection/            ← Phase 3.5: refinement
│   └── reconstruction/       ← Phase 4: 3D extrusion + export
├── gan_part/                 ← GAN interior generation (see README_GAN.md)
├── requirements.txt
└── configs/
    └── segmentation.yaml     ← YOLO dataset config for training
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `ultralytics`, `opencv-python`, `numpy`, `torch`, `torchvision`

Optional: `diffusers`, `transformers` (for Phase 5 ControlNet render)

---

## Training a New YOLO Model

```python
from src.segmentation.trainer import SegmentationTrainer

trainer = SegmentationTrainer(
    dataset_yaml="data/dataset.yaml",
    model_size="s",   # n/s/m
    epochs=100,
)
trainer.train()
```

See `notebooks/kaggle_phase2_training.ipynb` for Kaggle training setup.

---

## Known Limitations

| Issue | Impact |
|---|---|
| Model only detects wall/door/window | No room type labels → scale uses wall thickness heuristic |
| `sklearn` not installed | PCA wall direction fails silently (fallback works) |
| RTX 5060 not supported by PyTorch stable | GPU unused, all inference on CPU |
| No roof generated | Intentional — open-top building |
