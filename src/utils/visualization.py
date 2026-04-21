"""
Visualization utilities for detection validation.

This module provides functions to visualize detection results,
allowing visual inspection of walls, rooms, doors, and windows.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2


def visualize_detections(
    image: np.ndarray,
    detections: Dict[str, List[np.ndarray]],
    output_path: Optional[str] = None,
    show_labels: bool = True,
    room_alpha: float = 0.3
) -> np.ndarray:
    """
    Visualize detection results with color-coded overlays.
    
    This function draws walls, rooms, doors, and windows on the input image
    with different colors and styles for easy visual inspection.
    
    Parameters
    ----------
    image : np.ndarray
        Original or preprocessed image (grayscale or BGR)
    detections : Dict[str, List[np.ndarray]]
        Dictionary containing detection results with keys:
        - "walls": List of wall polygons (each Nx2 numpy array)
        - "rooms": List of room polygons (each Nx2 numpy array)
        - "doors": List of door polygons (each Nx2 numpy array)
        - "windows": List of window polygons (each Nx2 numpy array)
    output_path : str, optional
        If provided, save the annotated image to this path
    show_labels : bool, default=True
        Whether to add text labels for doors and windows
    room_alpha : float, default=0.3
        Transparency for room fill (0.0 = transparent, 1.0 = opaque)
    
    Returns
    -------
    np.ndarray
        Annotated image with detection overlays
    
    Examples
    --------
    >>> detections = {
    ...     "walls": [wall_polygon1, wall_polygon2],
    ...     "rooms": [room_polygon1, room_polygon2],
    ...     "doors": [door_polygon1],
    ...     "windows": [window_polygon1, window_polygon2]
    ... }
    >>> annotated = visualize_detections(image, detections, "output.png")
    """
    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        viz_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        viz_image = image.copy()
    
    # Ensure image is uint8
    if viz_image.dtype != np.uint8:
        viz_image = (viz_image * 255).astype(np.uint8) if viz_image.max() <= 1.0 else viz_image.astype(np.uint8)
    
    # Define colors (BGR format)
    colors = {
        "walls": (0, 255, 0),      # Green
        "rooms": (144, 238, 144),  # Light green
        "doors": (255, 0, 0),      # Blue
        "windows": (0, 0, 255)     # Red
    }
    
    # Define thicknesses
    thicknesses = {
        "walls": 2,
        "rooms": -1,  # Filled
        "doors": 2,
        "windows": 2
    }
    
    # Draw rooms first (with transparency)
    if "rooms" in detections and detections["rooms"]:
        rooms_overlay = viz_image.copy()
        for room_polygon in detections["rooms"]:
            if room_polygon is None or len(room_polygon) == 0:
                continue
            
            # Convert to integer coordinates
            points = _ensure_int_array(room_polygon)
            
            # Draw filled polygon
            cv2.fillPoly(rooms_overlay, [points], colors["rooms"])
        
        # Blend with original image for transparency
        cv2.addWeighted(rooms_overlay, room_alpha, viz_image, 1 - room_alpha, 0, viz_image)
    
    # Draw walls
    if "walls" in detections and detections["walls"]:
        for wall_polygon in detections["walls"]:
            if wall_polygon is None or len(wall_polygon) == 0:
                continue
            
            points = _ensure_int_array(wall_polygon)
            cv2.polylines(viz_image, [points], isClosed=True, 
                         color=colors["walls"], thickness=thicknesses["walls"])
    
    # Draw doors with labels
    if "doors" in detections and detections["doors"]:
        for i, door_polygon in enumerate(detections["doors"]):
            if door_polygon is None or len(door_polygon) == 0:
                continue
            
            points = _ensure_int_array(door_polygon)
            cv2.polylines(viz_image, [points], isClosed=True,
                         color=colors["doors"], thickness=thicknesses["doors"])
            
            # Add label
            if show_labels:
                centroid = _compute_centroid(points)
                _draw_label(viz_image, f"Door {i+1}", centroid, colors["doors"])
    
    # Draw windows with labels
    if "windows" in detections and detections["windows"]:
        for i, window_polygon in enumerate(detections["windows"]):
            if window_polygon is None or len(window_polygon) == 0:
                continue
            
            points = _ensure_int_array(window_polygon)
            cv2.polylines(viz_image, [points], isClosed=True,
                         color=colors["windows"], thickness=thicknesses["windows"])
            
            # Add label
            if show_labels:
                centroid = _compute_centroid(points)
                _draw_label(viz_image, f"Win {i+1}", centroid, colors["windows"])
    
    # Add legend
    viz_image = _add_legend(viz_image, colors)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, viz_image)
        print(f"Visualization saved to: {output_path}")
    
    return viz_image


def visualize_comparison(
    image: np.ndarray,
    detections_before: Dict[str, List[np.ndarray]],
    detections_after: Dict[str, List[np.ndarray]],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of detections before and after refinement.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    detections_before : Dict[str, List[np.ndarray]]
        Detections before refinement (YOLO)
    detections_after : Dict[str, List[np.ndarray]]
        Detections after refinement
    output_path : str, optional
        If provided, save the comparison image
    
    Returns
    -------
    np.ndarray
        Side-by-side comparison image
    """
    # Create visualizations
    viz_before = visualize_detections(image, detections_before, show_labels=False)
    viz_after = visualize_detections(image, detections_after, show_labels=True)
    
    # Add titles
    viz_before = _add_title(viz_before, "Before Refinement (YOLO)")
    viz_after = _add_title(viz_after, "After Refinement (Geometry)")
    
    # Concatenate horizontally
    comparison = np.hstack([viz_before, viz_after])
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, comparison)
        print(f"Comparison saved to: {output_path}")
    
    return comparison


def visualize_vectorization_result(
    image: np.ndarray,
    vectorization_result,
    output_path: Optional[str] = None,
    show_labels: bool = True
) -> np.ndarray:
    """
    Visualize a VectorizationResult object.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    vectorization_result : VectorizationResult
        VectorizationResult object from wall_vectorizer
    output_path : str, optional
        If provided, save the visualization
    show_labels : bool, default=True
        Whether to show labels
    
    Returns
    -------
    np.ndarray
        Annotated image
    """
    # Convert VectorizationResult to detections dict
    detections = {
        "walls": [np.array(w.points) for w in vectorization_result.walls],
        "rooms": [np.array(r.points) for r in vectorization_result.rooms],
        "doors": [np.array(d.points) for d in vectorization_result.doors],
        "windows": [np.array(w.points) for w in vectorization_result.windows]
    }
    
    return visualize_detections(image, detections, output_path, show_labels)


# ── Helper Functions ──────────────────────────────────────────────────────────

def _ensure_int_array(polygon: Union[np.ndarray, List]) -> np.ndarray:
    """Convert polygon to integer numpy array."""
    if isinstance(polygon, list):
        polygon = np.array(polygon)
    
    # Ensure 2D array
    if len(polygon.shape) == 1:
        polygon = polygon.reshape(-1, 2)
    
    return polygon.astype(np.int32)


def _compute_centroid(points: np.ndarray) -> Tuple[int, int]:
    """Compute centroid of a polygon."""
    centroid = np.mean(points, axis=0)
    return (int(centroid[0]), int(centroid[1]))


def _draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.4,
    thickness: int = 1
):
    """Draw text label with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    x, y = position
    padding = 2
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (255, 255, 255),
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def _add_legend(
    image: np.ndarray,
    colors: Dict[str, Tuple[int, int, int]],
    position: str = "top-right"
) -> np.ndarray:
    """Add color legend to image."""
    legend_items = [
        ("Walls", colors["walls"]),
        ("Rooms", colors["rooms"]),
        ("Doors", colors["doors"]),
        ("Windows", colors["windows"])
    ]
    
    # Legend dimensions
    item_height = 25
    item_width = 120
    padding = 10
    legend_height = len(legend_items) * item_height + 2 * padding
    legend_width = item_width + 2 * padding
    
    # Determine position
    h, w = image.shape[:2]
    if position == "top-right":
        x_start = w - legend_width - 10
        y_start = 10
    elif position == "top-left":
        x_start = 10
        y_start = 10
    elif position == "bottom-right":
        x_start = w - legend_width - 10
        y_start = h - legend_height - 10
    else:  # bottom-left
        x_start = 10
        y_start = h - legend_height - 10
    
    # Draw legend background
    cv2.rectangle(
        image,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (255, 255, 255),
        -1
    )
    cv2.rectangle(
        image,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (0, 0, 0),
        1
    )
    
    # Draw legend items
    for i, (label, color) in enumerate(legend_items):
        y = y_start + padding + i * item_height + item_height // 2
        
        # Draw color box
        box_size = 15
        cv2.rectangle(
            image,
            (x_start + padding, y - box_size // 2),
            (x_start + padding + box_size, y + box_size // 2),
            color,
            -1
        )
        cv2.rectangle(
            image,
            (x_start + padding, y - box_size // 2),
            (x_start + padding + box_size, y + box_size // 2),
            (0, 0, 0),
            1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x_start + padding + box_size + 10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    return image


def _add_title(
    image: np.ndarray,
    title: str,
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """Add title to top of image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        title, font, font_scale, thickness
    )
    
    # Create space for title
    title_height = text_height + baseline + 20
    titled_image = np.ones((image.shape[0] + title_height, image.shape[1], 3), dtype=np.uint8) * 255
    titled_image[title_height:, :] = image
    
    # Draw title
    x = (image.shape[1] - text_width) // 2
    y = text_height + 10
    cv2.putText(
        titled_image,
        title,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )
    
    return titled_image


def create_detection_report(
    image: np.ndarray,
    detections: Dict[str, List[np.ndarray]],
    output_path: str,
    title: str = "Detection Report"
):
    """
    Create a comprehensive detection report with statistics.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    detections : Dict[str, List[np.ndarray]]
        Detection results
    output_path : str
        Path to save the report image
    title : str, default="Detection Report"
        Report title
    """
    # Create visualization
    viz = visualize_detections(image, detections, show_labels=True)
    
    # Add title
    viz = _add_title(viz, title)
    
    # Add statistics panel
    stats_text = [
        f"Walls: {len(detections.get('walls', []))}",
        f"Rooms: {len(detections.get('rooms', []))}",
        f"Doors: {len(detections.get('doors', []))}",
        f"Windows: {len(detections.get('windows', []))}"
    ]
    
    # Draw statistics
    y_offset = 50
    for i, text in enumerate(stats_text):
        cv2.putText(
            viz,
            text,
            (10, y_offset + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
    
    # Save report
    cv2.imwrite(output_path, viz)
    print(f"Detection report saved to: {output_path}")
