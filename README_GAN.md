# Floor2Model — GAN Interior Pipeline

Takes the CV pipeline's 3D building model and adds realistic interior textures (walls, floor, doors, windows) and furniture using a pretrained Pix2Pix GAN, with OCR-guided furniture placement.

---

## Quick Start

```bash
# Step 1: Run the CV pipeline first (generates the 3D building)
python cv_pipeline.py

# Step 2: Run the GAN interior pipeline
python -m gan_part.gan_runner

# Or run both in one command
python main.py
```

Output: `gan_output/<stem>/<stem>_interior.gltf` for each floorplan.

**View:** drag `_interior.gltf` to [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com)

---

## What It Does

```
CV output (.gltf with grey walls)
        │
        ▼
Step 1 — Parse CV model
  Read wall/door/window face groups by per-vertex color
        │
        ▼
Step 2 — OCR Room Detection
  Scan floorplan image for text labels (Bedroom, Kitchen, etc.)
  Map pixel positions → 3D metre coordinates
        │
        ▼
Step 3 — GAN Texture Generation (Pix2Pix)
  Generate one texture per surface type:
    wall → white plaster texture
    floor → oak wood texture
    door → wood grain texture
    window → frosted glass (transparent)
        │
        ▼
Step 4 — Apply Textures + Add Furniture
  UV-map textures onto correct face groups
  Place furniture based on OCR room labels
  Floor = cropped floorplan image (top-down view)
        │
        ▼
Output: _interior.gltf
  Same geometry as CV output +
  Realistic surface textures +
  Furniture meshes (sofa, bed, table, chairs, etc.)
```

---

## GAN Model

**Model:** `timbrooks/instruct-pix2pix` (HuggingFace)

This is a **Pix2Pix GAN** — a conditional adversarial network (Generator + PatchGAN Discriminator) that translates one image to another. It takes a solid-color segmentation patch as input and generates a realistic texture image.

This is the same architecture as the custom Pix2Pix being trained on Kaggle (`gan_part/models/`). When training finishes, swap the backend:

```python
# In gan_part/gan_runner.py — change these two lines:
BACKEND        = "gan"
GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"
```

---

## OCR Room Detection

The pipeline runs OCR on the original floorplan image to find room labels (e.g. "Bedroom", "Kitchen", "Living Room"). These are matched to 3D positions for furniture placement.

**Supported OCR backends (in priority order):**
1. **EasyOCR** — better for small/rotated text: `pip install easyocr`
2. **pytesseract** — fallback: `pip install pytesseract` + install Tesseract binary

**Room labels detected:**
| Text found | Room type | Furniture placed |
|---|---|---|
| bed, bedroom, master | Bedroom | Bed + wardrobe |
| living, lounge, family | LivingRoom | Sofa + coffee table + bookshelf |
| kitchen, dining | Kitchen | Dining table + 4 chairs |
| bath, wc, toilet | Bathroom | (no furniture) |
| hall, corridor | Corridor | (no furniture) |

If no OCR labels are found, falls back to grid-based placement.

---

## Furniture

All furniture is procedural 3D geometry (box primitives) with realistic AEC dimensions:

| Item | Size (W × D × H) |
|---|---|
| Sofa | 1.6m × 0.75m × 0.85m |
| Coffee table | 0.9m × 0.45m × 0.42m |
| Dining table | 1.4m × 0.8m × 0.75m |
| Chair | 0.45m × 0.45m × 0.85m |
| Double bed | 1.5m × 2.0m × 0.55m |
| Wardrobe | 1.0m × 0.5m × 2.0m |
| Bookshelf | 0.7m × 0.25m × 1.6m |

An occupancy grid prevents furniture from overlapping each other or walls (0.3m clearance from walls).

---

## Surface Textures

| Surface | Material | Roughness | Transparency |
|---|---|---|---|
| Wall | White plaster (GAN-generated) | 0.85 | Opaque |
| Floor | Cropped floorplan image (top-down) | — | Opaque |
| Door | Wood grain (GAN-generated) | 0.70 | Opaque |
| Window | Frosted glass (35% opacity) | 0.05 | Semi-transparent |

The floor uses the actual floorplan image cropped to wall extents — so when you look down into the model you see the real floorplan layout.

---

## Output Files

```
gan_output/<stem>/
└── <stem>_interior.gltf    ← textured 3D model with furniture
```

The CV outputs in `generated_models/` are **not modified**.

---

## Project Structure

```
gan_part/
├── gan_runner.py                  ← MAIN ENTRY POINT (run this)
├── models/
│   ├── generator.py               ← UNet Generator (8-level)
│   ├── discriminator.py           ← PatchGAN Discriminator (70×70)
│   └── losses.py                  ← GAN + L1 + Perceptual losses
├── training/
│   ├── trainer.py                 ← Pix2Pix training loop
│   ├── config.py                  ← Hyperparameters
│   └── callbacks.py               ← Checkpoint + sample saving
├── inference/
│   ├── texture_generator.py       ← Pix2Pix GAN texture generation
│   ├── gltf_texturer.py           ← Reads CV .gltf, applies textures + furniture
│   ├── furniture_placer.py        ← Procedural furniture meshes + placement
│   ├── ocr_room_detector.py       ← OCR room label detection
│   ├── room_renderer.py           ← Unified renderer (controlnet / gan)
│   ├── controlnet_pipeline.py     ← ControlNet inference wrapper
│   ├── gan_pipeline.py            ← Custom GAN inference wrapper
│   ├── depth_pipeline.py          ← MiDaS depth estimation
│   └── interior_mesh.py           ← Depth map → 3D mesh
├── data/
│   ├── room_class_map.py          ← ADE20K color mapping
│   ├── preprocessor.py            ← Seg map building + normalization
│   ├── ade20k_loader.py           ← ADE20K dataset loader
│   └── lsun_loader.py             ← LSUN bedroom loader
├── integration/
│   └── floor2model_bridge.py      ← Bridge to CV pipeline
├── app/
│   └── gradio_app.py              ← Interactive web UI
├── notebooks/
│   └── kaggle_training.ipynb      ← Kaggle training notebook
└── requirements.txt
```

---

## Training Your Own GAN (Kaggle)

1. Upload `gan_part/` folder to Kaggle
2. Add `ADEChallengeData2016` dataset to inputs
3. Open `notebooks/kaggle_training.ipynb` and run all cells
4. Download `final.pt` from Kaggle outputs
5. Place it at `gan_part/checkpoints/final.pt`
6. Update `gan_runner.py`:

```python
BACKEND        = "gan"
GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"
```

**Training config** (`gan_part/training/config.py`):
- Architecture: UNet Generator + PatchGAN Discriminator
- Loss: LSGAN adversarial + L1 (λ=100) + Perceptual VGG16 (λ=10)
- Optimizer: AdamW, lr=2e-4, β1=0.5
- Schedule: 50 epochs constant + 50 epochs linear decay
- Input: 256×256 ADE20K-colored segmentation maps
- Output: 256×256 photorealistic room textures

---

## Interactive Gradio App

```bash
python -m gan_part.app.gradio_app
```

Opens at `http://localhost:7860` — upload a segmentation map, pick a room type, generate a texture.

---

## Dependencies

```bash
pip install -r gan_part/requirements.txt
```

Key packages: `diffusers`, `transformers`, `torch`, `gradio`

Optional: `easyocr` (better OCR), `timm` (MiDaS depth)

---

## Swap to Your Trained GAN

When `final.pt` is ready — **one change** in `gan_part/gan_runner.py`:

```python
# Before (using pretrained Pix2Pix from HuggingFace):
BACKEND        = "controlnet"
GAN_CHECKPOINT = None

# After (using your trained model):
BACKEND        = "gan"
GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"
```

Everything else stays the same. Your GAN replaces the pretrained model in Step 3.
