"""
gradio_app.py
-------------
Interactive Gradio demo for the GAN room rendering system.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path so we can import everything correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from PIL import Image

from gan_part.inference.room_renderer import RoomRenderer

# Global renderer state
RENDERER = None

def get_renderer(backend, device):
    global RENDERER
    if RENDERER is None or RENDERER.backend != backend:
        # Defaulting to cpu as user has no GPU
        RENDERER = RoomRenderer(backend=backend, device=device)
    return RENDERER

def render_interface(image, room_type, user_prompt, backend):
    """Gradio wrapper for rendering."""
    try:
        # Convert to numpy
        seg_map = np.array(image)
        
        # Use CPU by default as per user info
        renderer = get_renderer(backend, "cpu")
        
        result = renderer.render(
            seg_map, 
            room_type, 
            user_prompt=user_prompt
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Define the UI
with gr.Blocks(title="Floor2Model GAN Interior Renderer") as demo:
    gr.Markdown("# 🏠 Floor2Model: GAN Room Interior Renderer")
    gr.Markdown(
        "Upload a **segmentation map** (ADE20K colors) or use the previous project output "
        "to generate a realistic room interior."
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Segmentation Map (ADE20K Protocol)")
            room_type = gr.Dropdown(
                choices=["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Corridor", "Garage"],
                value="Bedroom",
                label="Room Category"
            )
            prompt = gr.Textbox(
                label="Style Prompt (optional)", 
                placeholder="e.g. minimalist wooden style, warm lighting..."
            )
            backend = gr.Radio(
                choices=["controlnet", "gan"], 
                value="controlnet", 
                label="Model Backend (ControlNet is recommended for quality)"
            )
            render_btn = gr.Button("Generate Room Interior", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Generated Result")
            
    render_btn.click(
        fn=render_interface,
        inputs=[input_img, room_type, prompt, backend],
        outputs=[output_img]
    )
    
    gr.Examples(
        examples=[
            # We would add actual file paths here if we had sample seg maps
        ],
        inputs=[input_img]
    )

if __name__ == "__main__":
    demo.launch(share=True)
