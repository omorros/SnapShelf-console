"""
Class-agnostic YOLO detector for structural pre-processing (Pipeline B).
Uses standard YOLOv8 (COCO-trained) with class labels ignored.
All detections treated as generic "object" regions for LLM classification.
"""

import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

from pathlib import Path
from typing import List
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION (matching YOLO-World for fair comparison)
# =============================================================================

# Confidence threshold (lower = more detections, higher recall)
CONF_THRESHOLD = 0.15

# IoU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.45

# Maximum detections per image
MAX_DETECTIONS = 20

# Crop padding as percentage (captures context around objects)
CROP_PADDING_PCT = 0.10

# =============================================================================


class YOLODetectorAgnostic:
    """
    Class-agnostic YOLOv8 detector for region proposals.

    Uses standard COCO-trained YOLOv8 but ignores class labels.
    Returns cropped regions for LLM classification (structural pre-processing only).
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
        Initialize class-agnostic YOLOv8 detector.

        Args:
            model_path: YOLOv8 model weights (auto-downloads if missing)
            conf_threshold: Detection confidence threshold (default 0.15)
            iou_threshold: NMS IoU threshold (default 0.45)
            max_detections: Max detections to return (default 20)
            crop_padding: Padding percentage for crops (default 0.10)
        """
        # Resolve model path
        if Path(model_path).exists():
            resolved_path = model_path
        else:
            repo_root = Path(__file__).parent.parent
            repo_path = repo_root / model_path
            if repo_path.exists():
                resolved_path = str(repo_path)
            else:
                # Let Ultralytics auto-download
                resolved_path = model_path

        # Load standard YOLOv8 model (COCO-trained, 80 classes)
        self.model = YOLO(resolved_path)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.crop_padding = crop_padding

    def detect(self, image_path: str) -> List[dict]:
        """
        Detect objects and return cropped regions (class labels ignored).

        Args:
            image_path: Path to the image file

        Returns:
            List of dicts containing:
                - bbox: {x, y, width, height}
                - image_bytes: PNG bytes of cropped region
                - confidence: Detection confidence
                - prompt_match: Always "object" (class-agnostic)
        """
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run YOLOv8 detection
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

                # Class label intentionally ignored (class-agnostic)
                # All detections treated as generic "object"

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
                    "confidence": conf,
                    "prompt_match": "object"  # Class-agnostic: always "object"
                })

        return detections
