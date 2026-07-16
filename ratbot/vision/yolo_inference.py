"""
Shared YOLO inference utilities for consistent inference across different scripts.
This module provides a unified interface for running YOLO inference with standard parameters.
"""

from typing import Union, Optional, List
import numpy as np
from pathlib import Path


def run_inference(
    model,
    source: Union[str, Path, np.ndarray],
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: Optional[str] = None,
    agnostic_nms: bool = False,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    verbose: bool = False
):
    """
    Run YOLO inference with consistent parameters.

    Args:
        model: YOLO model instance
        source: Input source (file path, Path object, or numpy array)
        conf: Confidence threshold (default: 0.25)
        iou: IoU threshold for NMS (default: 0.45)
        imgsz: Inference image size in pixels (default: 640)
        device: Device to use - '0' for GPU, 'cpu' for CPU, None for auto-detect (default: None)
        agnostic_nms: Class-agnostic NMS (default: False)
        max_det: Maximum number of detections per image (default: 300)
        classes: Filter by class indices, e.g., [0, 1, 2] (default: None)
        verbose: Print verbose output (default: False)

    Returns:
        YOLO results object(s)

    Examples:
        >>> model = YOLO('yolo11n.pt')
        >>> # Inference on image file (auto-detect device)
        >>> results = run_inference(model, 'image.jpg', conf=0.5, imgsz=640)
        >>>
        >>> # Inference on numpy array (frame from camera)
        >>> import cv2
        >>> frame = cv2.imread('image.jpg')
        >>> results = run_inference(model, frame, conf=0.75, imgsz=1024)
        >>>
        >>> # Force CPU usage
        >>> results = run_inference(model, frame, device='cpu')
    """
    # Build kwargs, only include device if explicitly specified
    kwargs = {
        'conf': conf,
        'iou': iou,
        'imgsz': imgsz,
        'agnostic_nms': agnostic_nms,
        'max_det': max_det,
        'classes': classes,
        'verbose': verbose
    }

    # Only pass device if explicitly specified, otherwise let YOLO auto-detect
    if device is not None:
        kwargs['device'] = device

    return model(source, **kwargs)


def extract_detections(results, model, target_class: Optional[str] = None):
    """
    Extract detection information from YOLO results.

    Args:
        results: YOLO results object from inference
        model: YOLO model instance (needed for class names)
        target_class: Optional class name to filter (e.g., 'rat')

    Returns:
        List of dictionaries containing detection info:
        [
            {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy)
            },
            ...
        ]

    Examples:
        >>> results = run_inference(model, 'image.jpg')
        >>> detections = extract_detections(results, model, target_class='rat')
        >>> for det in detections:
        ...     print(f"Found {det['class_name']} at {det['center']} with conf {det['confidence']:.2f}")
    """
    detections = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls_id] if cls_id < len(model.names) else f"Class_{cls_id}"

            # Filter by target class if specified
            if target_class and target_class.lower() not in class_name.lower():
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = (int(x1), int(y1), int(x2), int(y2))

            # Calculate center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center = (center_x, center_y)

            detections.append({
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': bbox,
                'center': center
            })

    return detections
