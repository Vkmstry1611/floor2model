# GAN Part — Summary of Development

This document provides a comprehensive overview of the `gan_part` module, which was developed to add photorealistic room interior generation to the `floor2model` project.

## Project Goal
To enable a two-stage pipeline where 2D floor plans are first segmented (Phase 2 of existing project) and then rendered into realistic 3D-like interior scenes based on room types and user prompts.

---

## Architecture Overview

The system uses a **Dual-Track Backend** designed to work both with and without specialized hardware (GPUs).

### 1. ControlNet (State-of-the-Art / Zero-Shot)
- **Backend**: Stable Diffusion 1.5 + ControlNet (`sd-controlnet-seg`).
- **Input**: ADE20K color-coded segmentation map + Text Prompt.
- **Strength**: Immediate, high-quality, photorealistic results without any training.
- **Hardware**: Runs on CPU (slow) or GPU (fast).

### 2. Pix2Pix GAN (Academic / Custom Training)
- **Backend**: Custom UNet Generator + PatchGAN Discriminator.
- **Training**: Optimized for Kaggle GPUs (T4/P100) using the ADE20K and LSUN datasets.
- **Input**: 3-channel segmentation map (256x256).
- **Strength**: Light-weight inference, fully controllable architecture, no dependency on external weights once trained.

---

## Components Implemented

### Data Layer (`gan_part/data/`)
- `room_class_map.py`: Maps floor plan categories to the official **ADE20K protocol** (ids, colors, and prompts).
- `ade20k_loader.py`: Handles automatic download and parsing of the MIT ADE20K dataset for multi-class training.
- `lsun_loader.py`: Self-supervised dataset builder that uses **UPerNet** to generate ground-truth labels for raw bedroom photos.
- `preprocessor.py`: Unified pipeline for resizing (256x256), normalization ([-1, 1]), and pixel-color conversion.

### Model Architecture (`gan_part/models/`)
- `generator.py`: 8-level **UNet** with skip connections and symmetric encoder/decoder blocks.
- `discriminator.py`: **PatchGAN** (70x70) that forces the generator to focus on local texture consistency.
- `losses.py`: Multi-objective loss function combining **Adversarial (LSGAN)**, **L1 (Reconstruction)**, and **Perceptual (VGG16)** losses.

### Inference & Logic (`gan_part/inference/`)
- `room_renderer.py`: A unified API that allows switching between `gan` and `controlnet` backends seamlessly.
- `prompt_engine.py`: Automatically generates descriptive "base prompts" based on the detected room type (e.g., "cozy bedroom" vs "modern kitchen").
- `gan_pipeline.py` & `controlnet_pipeline.py`: Low-level wrappers for model loading and generation.

### Integration & UI (`gan_part/app/` & `gan_part/integration/`)
- `gradio_app.py`: A local web GUI for uploading maps and viewing renders.
- `floor2model_bridge.py`: Automated bridge that takes a `SegmentationResult` from the parent project, crops individual rooms, and renders them in batch.

---

## Usage Instructions

### 1. Local Testing (No GPU required)
```bash
cd gan_part
pip install -r requirements.txt
python -m gan_part.app.gradio_app
```
*Note: Using the `controlnet` backend on CPU will take 1-3 minutes per image.*

### 2. Training on Kaggle
1. Upload the `gan_part` folder to your Kaggle environment.
2. Add the `ADEChallengeData2016` dataset to your inputs.
3. Open `notebooks/kaggle_training.ipynb` and run all cells.
4. Download the generated `final.pt` and place it in the `checkpoints/` folder.

---

## Next Steps
- [ ] **Collect local outputs**: Use saved `.json` or `.png` files from Phase 2 to test the `Floor2ModelBridge`.
- [ ] **Fine-tuning**: Perform a short fine-tuning run on LSUN for ultra-detailed bedrooms.
- [ ] **3D Projection**: (Future) Project the generated 2D renders back onto the 3D meshes from Phase 4.
