"""
Detection refinement module for improving door and window detection.

This module provides geometry-based refinement of YOLO segmentation results,
using wall structure analysis and computer vision techniques to improve
detection accuracy for doors and windows in floor plans.
"""

from .refinement import (
    RefinementConfig,
    DetectionRefiner,
    refine_detections,
)

__all__ = [
    'RefinementConfig',
    'DetectionRefiner',
    'refine_detections',
]
