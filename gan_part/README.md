# gan_part — Realistic Room Scenery Generation

This module is the second stage of the `floor2model` pipeline. It converts 2D floor plan segmentation maps into photorealistic room interior renderings.

## Features
- **Dual Backend**: Use a home-grown **Pix2Pix GAN** (trained from scratch) or a zero-shot **ControlNet** (Stable Diffusion).
- **Promptable**: Control the visual style (e.g., "minimalist", "cozy", "industrial") via text prompts.
- **ADE20K Protocol**: Uses standard semantic color mapping to ensure high-quality spatial awareness.
- **Kaggle Optimized**: Includes configurations and notebooks for training on Kaggle GPUs.

## Installation
```bash
cd gan_part
pip install -r requirements.txt
```

## Usage

### 1. Interactive App (Gradio)
Launch a local UI to upload segmentation maps and render rooms.
```bash
python -m gan_part.app.gradio_app
```

### 2. Training (Kaggle)
1. Upload the `gan_part` folder to Kaggle.
2. Open `notebooks/kaggle_training.ipynb`.
3. Select an ADE20K dataset and start training.

### 3. Inference API
```python
from gan_part.inference.room_renderer import RoomRenderer

renderer = RoomRenderer(backend="controlnet") # or "gan" with checkpoint
image = renderer.render(
    seg_map=my_numpy_mask, 
    room_label="Bedroom", 
    user_prompt="cozy wooden style"
)
image.save("result.png")
```

## Structure
- `data/`: Dataset loaders (ADE20K, LSUN) and preprocessing.
- `models/`: Pix2Pix Generator, Discriminator, and losses.
- `training/`: Training loops and configs.
- `inference/`: Inference pipelines and prompt engine.
- `app/`: Gradio interface.
- `integration/`: Bridge to the main `floor2model` project.
