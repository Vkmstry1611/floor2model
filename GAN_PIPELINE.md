# GAN Pipeline — Technical Documentation

## Project Overview

The GAN pipeline is the second stage of the **Floor2Model** project. It takes the 3D building model produced by the CV pipeline and generates a **photorealistic interior** — applying AI-generated textures to walls, floors, doors, and windows, and placing procedural furniture inside the building.

---

## Full Pipeline Flow

```
CV Output (generated_models/<stem>/<stem>.gltf)
        │
        ▼
[Step 1] Parse CV .gltf
  Read vertices, faces, per-vertex colors
  Classify faces by color → wall / door / window / floor
        │
        ▼
[Step 2] Texture Generation (GAN / ControlNet)
  For each surface type → generate 512×512 texture image
  wall  → white plaster texture
  floor → oak hardwood texture
  door  → wood grain texture
  window → frosted glass texture
        │
        ▼
[Step 3] Apply Textures to 3D Geometry
  UV-map each texture onto its face group
  Planar projection per surface orientation
        │
        ▼
[Step 4] Furniture Placement
  Compute usable floor area (shrink by wall margin)
  Place sofa, coffee table, TV unit, dining set, bed, wardrobe, bookshelf
        │
        ▼
GAN Output (gan_output/<stem>/<stem>_interior.gltf)
  Same walls/doors/windows as CV
  + GAN-generated surface textures
  + Procedural furniture meshes
```

---

## Entry Points

| Script | Purpose |
|---|---|
| `python cv_pipeline.py` | Run CV only — all images in `samples/` |
| `python gan_pipeline.py` | Run GAN only — all stems in `generated_models/` |
| `python main.py` | Run both CV + GAN sequentially |

---

## Module Structure

```
gan_part/
├── models/
│   ├── generator.py        — Pix2Pix UNet Generator (54M params)
│   ├── discriminator.py    — PatchGAN Discriminator (2.8M params)
│   └── losses.py           — GAN + L1 + Perceptual losses
├── training/
│   ├── config.py           — TrainingConfig dataclass + presets
│   ├── trainer.py          — Pix2PixTrainer training loop
│   └── callbacks.py        — Checkpoint + sample image callbacks
├── data/
│   ├── room_class_map.py   — ADE20K color mapping
│   ├── preprocessor.py     — Image normalization utilities
│   ├── ade20k_loader.py    — ADE20K dataset loader
│   └── lsun_loader.py      — LSUN bedroom dataset loader
├── inference/
│   ├── texture_generator.py — Surface texture generation (ControlNet / procedural)
│   ├── gltf_texturer.py    — CV .gltf → textured interior .gltf
│   ├── furniture_placer.py — Procedural furniture mesh generation
│   ├── room_renderer.py    — Unified render API (GAN or ControlNet)
│   ├── controlnet_pipeline.py — ControlNet inference wrapper
│   ├── gan_pipeline.py     — Pix2Pix inference wrapper
│   └── prompt_engine.py    — Text prompt builder per room type
├── integration/
│   └── floor2model_bridge.py — Bridge to CV SegmentationResult
├── notebooks/
│   └── kaggle_training.ipynb — Kaggle training notebook
├── app/
│   └── gradio_app.py       — Interactive Gradio web UI
├── gan_runner.py           — Standalone GAN runner (legacy)
└── requirements.txt
```

---

## Part 1 — GAN Model Architecture

### Generator — `gan_part/models/generator.py`

**Type:** Pix2Pix UNet (conditional image-to-image translation)

**Task:** ADE20K segmentation map (3ch RGB) → photorealistic room photo (3ch RGB)

**Architecture:** 8-level encoder-decoder with skip connections

```
Input: (B, 3, 256, 256) — normalized seg map

Encoder (downsampling):
  E1: 3   → 64   (256→128, no BatchNorm)
  E2: 64  → 128  (128→64)
  E3: 128 → 256  (64→32)
  E4: 256 → 512  (32→16)
  E5: 512 → 512  (16→8)
  E6: 512 → 512  (8→4)
  E7: 512 → 512  (4→2)
  E8: 512 → 512  (2→1)  ← bottleneck

Decoder (upsampling + skip connections):
  D1: 512  → 512  (1→2,   dropout=True)
  D2: 1024 → 512  (2→4,   dropout=True)
  D3: 1024 → 512  (4→8,   dropout=True)
  D4: 1024 → 512  (8→16)
  D5: 1024 → 256  (16→32)
  D6: 512  → 128  (32→64)
  D7: 256  → 64   (64→128)
  D8: 128  → 3    (128→256, Tanh activation)

Output: (B, 3, 256, 256) — normalized image in [-1, 1]
```

**Building blocks:**
- `EncoderBlock`: Conv2d(4×4, stride=2) → BatchNorm → LeakyReLU(0.2)
- `DecoderBlock`: ConvTranspose2d(4×4, stride=2) → BatchNorm → [Dropout(0.5)] → ReLU
- Skip connections: encoder feature maps concatenated to decoder inputs

**Parameters:** ~54M trainable

**Weight initialization:** Gaussian N(0, 0.02) for Conv/ConvTranspose, N(1, 0.02) for BatchNorm

---

### Discriminator — `gan_part/models/discriminator.py`

**Type:** PatchGAN (70×70 receptive field)

**Task:** Given (seg_map, image) pair → predict real/fake per patch

**Architecture:** 4-layer CNN

```
Input: (B, 6, 256, 256) — concatenated seg_map(3ch) + image(3ch)

Layer 1: Conv(4×4, stride=2) → LeakyReLU(0.2)          [no BN]
Layer 2: Conv(4×4, stride=2) → BatchNorm → LeakyReLU
Layer 3: Conv(4×4, stride=2) → BatchNorm → LeakyReLU
Layer 4: Conv(4×4, stride=1) → BatchNorm → LeakyReLU
Output:  Conv(4×4, stride=1) → (B, 1, 30, 30)          [no activation]
```

**Parameters:** ~2.8M trainable

**Why PatchGAN:** Classifying 70×70 patches forces the generator to produce locally realistic textures rather than globally plausible but locally blurry images.

---

## Part 2 — Loss Functions

### `gan_part/models/losses.py`

**Combined generator loss:**
```
L_G = L_GAN(fake) + λ_L1 × L_L1 + λ_perc × L_Perceptual
```

#### GAN Loss (`GANLoss`)
Supports three modes:
| Mode | Formula | Notes |
|---|---|---|
| `lsgan` | MSE(D(fake), 1) | Default — more stable training |
| `vanilla` | BCE(D(fake), 1) | Original pix2pix |
| `wgan` | -D(fake).mean() | Wasserstein distance |

#### L1 Loss (`L1Loss`)
Pixel-wise reconstruction: `||G(x) - y||₁`
- Weight: λ_L1 = 100 (per original paper)
- Prevents mode collapse, ensures output matches ground truth structure

#### Perceptual Loss (`PerceptualLoss`)
VGG16 feature matching at layers relu1_2, relu2_2, relu3_3, relu4_3
- Weight: λ_perc = 10
- Produces sharper textures than L1 alone
- VGG16 is frozen (pretrained ImageNet weights)
- Input normalized to ImageNet mean/std

#### Discriminator Loss
```
L_D = 0.5 × [L_GAN(real, is_real=True) + L_GAN(fake, is_real=False)]
```

---

## Part 3 — Training

### Configuration — `gan_part/training/config.py`

**`TrainingConfig` dataclass — key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `data_root` | `data/ade20k` | ADE20K dataset path |
| `image_size` | 256 | Training resolution |
| `num_epochs` | 100 | Total epochs |
| `batch_size` | 4 | Samples per batch |
| `lr_g` | 2e-4 | Generator learning rate |
| `lr_d` | 2e-4 | Discriminator learning rate |
| `beta1` | 0.5 | Adam β1 (standard GAN value) |
| `beta2` | 0.999 | Adam β2 |
| `gan_mode` | `lsgan` | Loss type |
| `lambda_l1` | 100.0 | L1 loss weight |
| `lambda_percep` | 10.0 | Perceptual loss weight |
| `decay_start` | 50 | Epoch to start LR linear decay |
| `mixed_precision` | True | fp16 on CUDA |

**Preset configs:**
- `kaggle_config()` — batch=8, fp16, for Kaggle T4/P100
- `local_test_config()` — batch=1, no perceptual loss, for CPU testing

### Training Loop — `gan_part/training/trainer.py`

**`Pix2PixTrainer` — per batch:**

```
1. Train Discriminator:
   fake = G(seg)                          # generate fake
   real_pred = D(seg, photo)              # discriminate real
   fake_pred = D(seg, fake.detach())      # discriminate fake
   d_loss = 0.5 × (L_D_real + L_D_fake)  # discriminator loss
   d_loss.backward(); opt_D.step()

2. Train Generator:
   fake = G(seg)                          # generate again
   fake_pred = D(seg, fake)               # discriminate (no detach)
   g_loss = L_GAN + λ_L1×L_L1 + λ_p×L_perc
   g_loss.backward(); opt_G.step()
```

**LR Schedule:** Constant for first `decay_start` epochs, then linear decay to 0.

**Mixed precision:** `torch.cuda.amp.GradScaler` on CUDA.

### Callbacks — `gan_part/training/callbacks.py`

- `CheckpointCallback` — saves `{epoch}.pt` every N epochs with G, D, opt_G, opt_D state dicts
- `SampleCallback` — saves side-by-side comparison grid (seg | real | generated) every N steps

### Training on Kaggle

```
1. Upload gan_part/ folder to Kaggle
2. Add ADEChallengeData2016 dataset as input
3. Open notebooks/kaggle_training.ipynb
4. Run all cells
5. Download outputs/training/checkpoints/final.pt
6. Place at gan_part/checkpoints/final.pt
```

---

## Part 4 — Inference Pipeline

### Texture Generator — `gan_part/inference/texture_generator.py`

**`TextureGenerator`** — generates one 512×512 texture per surface type.

**Two backends:**

**ControlNet (current default):**
- Model: `lllyasviel/sd-controlnet-seg` + `runwayml/stable-diffusion-v1-5`
- Input: solid-color ADE20K seg map (e.g. grey for wall, cyan for floor)
- Prompt per surface type:
  - wall: `"seamless tileable white plaster wall texture..."`
  - floor: `"seamless tileable light oak hardwood floor texture..."`
  - door: `"seamless tileable wooden door texture..."`
  - window: `"seamless tileable frosted glass texture..."`
- 20 diffusion steps, guidance_scale=7.5
- ~1-2 min per texture on CPU

**Procedural fallback (no AI):**
- Used when ControlNet is unavailable
- `_plaster_texture()` — Gaussian noise on base color + blur
- `_wood_texture()` — sinusoidal grain lines + noise
- `_glass_texture()` — light blue noise + blur
- Instant, deterministic

**Switching to your trained GAN:**
```python
# In gan_pipeline.py, change:
BACKEND = "gan"
GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"
```

---

### glTF Texturer — `gan_part/inference/gltf_texturer.py`

**`apply_textures(cv_gltf_path, texture_generator, output_path)`**

**Step 1 — Parse CV .gltf:**
- Decode base64 buffer
- Find building mesh (skip `InteriorFloor`, `FloorPlan` planes)
- Read vertices (N×3), per-vertex colors (N×3), triangle indices

**Step 2 — Classify faces by color:**
Each triangle's 3 vertex colors are averaged → nearest surface type:

| Color (RGB 0-1) | Surface |
|---|---|
| (0.25-0.40, 0.25-0.40, 0.35-0.50) | wall |
| (0.65, 0.45, 0.25) | door |
| (0.60, 0.80, 0.95) | window |
| (0.60, 0.55, 0.50) | floor |
| (0.85, 0.85, 0.80) | ceiling |

**Step 3 — UV mapping (`_planar_uv`):**
- Floor/ceiling → top-down projection (U=X, V=Y)
- Horizontal walls → U=X, V=Z
- Vertical walls → U=Y, V=Z

**Step 4 — Build output .gltf:**
- Floor quad with tiled texture (4× repeat for realistic scale)
- Per-surface textured mesh primitives
- Furniture meshes with per-vertex colors
- All data base64-encoded in self-contained .gltf

**PBR material properties:**
| Surface | Metallic | Roughness |
|---|---|---|
| Wall | 0.0 | 0.85 |
| Floor | 0.0 | 0.85 |
| Door | 0.0 | 0.70 |
| Window | 0.05 | 0.20 |
| Furniture | 0.0 | 0.85 |

---

### Furniture Placer — `gan_part/inference/furniture_placer.py`

**`place_furniture(x_min, x_max, y_min, y_max)`**

Generates procedural furniture meshes placed inside the building footprint.

**Wall clearance:** `WALL_MARGIN = 0.35m` — all furniture stays this far from walls.

**Placement logic (based on usable floor area):**

| Condition | Furniture placed |
|---|---|
| Always | Sofa (against bottom wall) |
| Always | Coffee table (in front of sofa) |
| If space allows | TV unit (against top wall) |
| If width > 3.5m | Dining table + 4 chairs (right side) |
| If height > 3.5m | Bed + wardrobe (top-left corner) |
| If no overlap | Bookshelf (left wall) |

**Furniture pieces and dimensions:**

| Piece | Width | Depth | Height | Material |
|---|---|---|---|---|
| Sofa | 1.8m (scaled) | 0.85m | 0.92m | Grey fabric + beige cushions |
| Coffee table | 0.9m | 0.45m | 0.42m | Light oak |
| TV unit | 1.2m | 0.35m | 0.45m | Dark wood + black screen |
| Dining table | 1.2m | 0.75m | 0.76m | Light oak |
| Chair | 0.40m | 0.40m | 0.82m | Light oak |
| Bed | 1.4m | 1.9m | 0.47m | Wood frame + white linen |
| Wardrobe | 1.0m | 0.50m | 2.0m | Light oak + dark doors |
| Bookshelf | 0.70m | 0.25m | 1.70m | Dark wood + coloured books |

Each piece is built from multiple `_box()` primitives merged into one mesh with per-vertex colors.

---

## Part 5 — Data Layer

### ADE20K Color Mapping — `gan_part/data/room_class_map.py`

Maps floor2model class names to ADE20K protocol colors (used by ControlNet):

| floor2model class | ADE20K ID | RGB Color |
|---|---|---|
| wall / OuterWall / InnerWall | 3 | (120, 120, 120) grey |
| floor | 4 | (6, 230, 230) cyan |
| window / Window | 8 | (255, 255, 153) yellow |
| door / Door | 14 | (150, 5, 61) dark red |
| Kitchen | 25 | (255, 179, 240) pink |
| LivingRoom | 66 | (61, 230, 250) light blue |
| Bedroom | 7 | (4, 200, 3) green |
| Bathroom | 64 | (0, 102, 200) blue |
| Corridor | 98 | (255, 6, 51) red |

### Preprocessor — `gan_part/data/preprocessor.py`

- `build_seg_map_from_elements()` — builds ADE20K color map from YOLO detections
- `normalize()` — uint8 [0,255] → float32 [-1,1]
- `denormalize()` — float32 [-1,1] → uint8 [0,255]
- `prepare_seg_map()` — resize → normalize → tensor (for GAN inference)
- `make_comparison_grid()` — side-by-side [seg | real | generated] for training visualization

---

## Part 6 — Output Files

For each input image `samples/<stem>.jpg`:

```
generated_models/<stem>/          ← CV Pipeline output
├── <stem>_detections.png         YOLO detection overlay
├── <stem>_floorplan.png          clean floorplan texture
├── <stem>.gltf                   3D building (walls/doors/windows, grey)
└── <stem>.obj + .mtl             same in OBJ format

gan_output/<stem>/                ← GAN Pipeline output
└── <stem>_interior.gltf          3D building + textures + furniture
```

**How to view:**
- Drag any `.gltf` to → https://gltf-viewer.donmccurdy.com
- Open `.obj` in Blender, MeshLab, or Windows 3D Viewer

---

## Part 7 — Swapping to Your Trained GAN

When `final.pt` is ready from Kaggle training:

**1. Place checkpoint:**
```
gan_part/checkpoints/final.pt
```

**2. Change one line in `gan_pipeline.py`:**
```python
# Before (ControlNet):
BACKEND        = "controlnet"
GAN_CHECKPOINT = None

# After (your GAN):
BACKEND        = "gan"
GAN_CHECKPOINT = "gan_part/checkpoints/final.pt"
```

**3. Run:**
```bash
python gan_pipeline.py
```

The rest of the pipeline (texture application, furniture, glTF export) is identical — only the texture image generation changes.

---

## Part 8 — Dependencies

```
# Core ML
torch>=2.4.0
torchvision>=0.19.0
numpy<2.0

# Generative models
diffusers==0.30.3
transformers==4.44.0
accelerate
huggingface_hub>=0.23.0,<0.25.0

# Image processing
opencv-python-headless
pillow
scipy

# UI
gradio

# Utils
tqdm
matplotlib
timm
```

Install:
```bash
pip install -r gan_part/requirements.txt
```

---

## Part 9 — Known Limitations

| Issue | Impact | Status |
|---|---|---|
| ControlNet generates perspective photos, not tileable textures | Textures may not tile perfectly | Acceptable for demo |
| Current YOLO model has no room type labels | All rooms get generic furniture | Needs room-aware model |
| Furniture placement is generic (no room type awareness) | Bed may appear in living room | Improves when room labels added |
| RTX 5060 not supported by PyTorch stable | GPU acceleration unavailable | Use Kaggle or wait for PyTorch update |
| ControlNet ~2 min/texture on CPU | Full pipeline ~10 min/floorplan | Acceptable for demo |
| sklearn not installed | PCA-based wall direction fails in refinement | Non-critical, fallback works |

---

## Part 10 — Gradio App

Interactive web UI for testing:

```bash
python -m gan_part.app.gradio_app
```

Opens at `http://localhost:7860`

Features:
- Upload any segmentation map
- Select room type (Bedroom, LivingRoom, Kitchen, Bathroom, Corridor, Garage)
- Enter style prompt (e.g. "minimalist wooden style")
- Choose backend (controlnet or gan)
- View generated interior image

`share=True` in the code also generates a temporary public URL.
