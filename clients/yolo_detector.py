"""
YOLO detector module for object detection and region cropping.
Uses Ultralytics YOLOv8 to produce bounding boxes for LLM labeling.
"""

import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

from pathlib import Path
from typing import List
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


# =============================================================================
# YOLO CONFIGURATION PARAMETERS
# Adjust these values to tune detection behavior
# =============================================================================

# Confidence threshold for detections (0.0 - 1.0)
# Lower = more detections (higher recall), Higher = fewer detections (higher precision)
CONF_THRESHOLD = 0.25

# IoU threshold for Non-Maximum Suppression
# Controls how much overlap is allowed before boxes are merged
# Using YOLO's default NMS when set; explicitly pass to override
IOU_THRESHOLD = 0.45

# Maximum number of detections to return
# Limits results to top K highest-confidence detections
MAX_DETECTIONS = 15

# Crop padding as percentage of bounding box dimensions (0.0 - 1.0)
# Adds extra margin around detected objects when cropping
CROP_PADDING_PCT = 0.10

# =============================================================================


class YOLODetector:
    """
    YOLO-based object detector for generating region proposals.

    Uses YOLOv8 to detect objects and crop regions for LLM labeling.
    The detector returns cropped images with bounding box metadata.

    Attributes:
        model: Loaded YOLO model instance
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections to return
        crop_padding: Padding percentage for crops
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        crop_padding: float = CROP_PADDING_PCT
    ):
        """
        Initialize YOLO detector with model and parameters.

        Args:
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Confidence threshold (default 0.25)
            iou_threshold: IoU threshold for NMS (default 0.45)
            max_detections: Max detections to return (default 15)
            crop_padding: Padding percentage for crops (default 0.10)
        """
        # Find model in repo root or use absolute path
        if not Path(model_path).exists():
            repo_root = Path(__file__).parent.parent
            model_path = str(repo_root / model_path)

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.crop_padding = crop_padding

    def detect(self, image_path: str) -> List[dict]:
        """
        Detect objects and return cropped regions.

        Pipeline: Load image -> Run YOLO -> Crop detections -> Return crops

        Args:
            image_path: Path to the image file

        Returns:
            List of dicts containing:
                - bbox: {x, y, width, height} coordinates
                - image_bytes: PNG bytes of cropped region
                - confidence: Detection confidence score
        """
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run YOLO detection
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                width, height = x2 - x1, y2 - y1

                # Add padding to capture full object
                pad_x = int(width * self.crop_padding)
                pad_y = int(height * self.crop_padding)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(image.width, x2 + pad_x)
                y2 = min(image.height, y2 + pad_y)

                # Crop the region
                crop = image.crop((x1, y1, x2, y2))

                # Convert to bytes
                buffer = BytesIO()
                crop.save(buffer, format="PNG")

                detections.append({
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    "image_bytes": buffer.getvalue(),
                    "confidence": conf
                })

        return detections

    def get_full_image_fallback(self, image_path: str) -> List[dict]:
        """
        Return full image as a single detection region.
        Used when YOLO finds no detections and fallback is enabled.

        Args:
            image_path: Path to the image file

        Returns:
            List with single detection covering full image
        """
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        return [{
            "bbox": {
                "x": 0,
                "y": 0,
                "width": image.width,
                "height": image.height
            },
            "image_bytes": buffer.getvalue(),
            "confidence": 1.0
        }]
