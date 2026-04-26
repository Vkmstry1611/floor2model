"""
floor2model_bridge.py
---------------------
Bridges the floor2model segmentation output with the GAN rendering pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image

from ..data.room_class_map import is_renderable
from ..data.preprocessor import build_seg_map_from_elements, crop_room_seg_map
from ..inference.room_renderer import RoomRenderer

class Floor2ModelBridge:
    """
    Takes floor2model SegmentationResult and renders all rooms.
    """

    def __init__(self, renderer: RoomRenderer):
        self.renderer = renderer

    def process_floorplan(
        self, 
        seg_result, 
        user_prompts: dict[str, str] | None = None
    ) -> dict[str, Image.Image]:
        """
        Process a full floor plan and return a dictionary of room images.

        Args:
            seg_result:   A SegmentationResult object from floor2model.
            user_prompts: Optional dict mapping room class names to prompts.

        Returns:
            Dict mapping room ID (or generic name) to PIL Image.
        """
        user_prompts = user_prompts or {}
        
        # 1. Build full-image segmentation map in ADE20K color protocol
        full_seg_map = build_seg_map_from_elements(
            seg_result.elements, 
            seg_result.image_shape
        )
        
        results = {}
        
        # 2. Extract and render each room
        for i, room in enumerate(seg_result.rooms):
            if not is_renderable(room.class_name):
                continue
                
            # Crop the room's specific segmentation region
            room_seg = crop_room_seg_map(full_seg_map, room.bbox)
            
            # Fetch custom prompt if provided
            room_prompt = user_prompts.get(room.class_name)
            
            print(f"Rendering {room.class_name} #{i}...")
            rendered = self.renderer.render(
                room_seg, 
                room.class_name, 
                user_prompt=room_prompt
            )
            
            results[f"{room.class_name}_{i}"] = rendered
            
        return results
