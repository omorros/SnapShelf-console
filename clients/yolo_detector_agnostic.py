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

# Maximum detections per image (reduced to limit noise)
MAX_DETECTIONS = 8

# Crop padding as percentage (captures context around objects)
CROP_PADDING_PCT = 0.10

# =============================================================================
# GEOMETRIC FILTERS (non-semantic noise reduction)
# =============================================================================

# Minimum bounding box area as percentage of image area
# Filters out tiny detections (noise, distant objects)
MIN_BBOX_AREA_PCT = 0.02  # 2% of image area

# Aspect ratio constraints (width/height)
# Filters out extremely elongated detections (edges, lines)
MIN_ASPECT_RATIO = 0.2   # Not too tall/narrow
MAX_ASPECT_RATIO = 5.0   # Not too wide/flat

# =============================================================================


class YOLODetectorAgnostic:
    """
    Class-agnostic YOLOv8 detector for region proposals.

    Uses standard COCO-trained YOLOv8 but ignores class labels.
    Returns cropped regions for LLM classification (structural pre-processing only).
    Applies geometric filtering to reduce noise (no semantic reasoning).
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        crop_padding: float = CROP_PADDING_PCT,
        min_bbox_area_pct: float = MIN_BBOX_AREA_PCT,
        min_aspect_ratio: float = MIN_ASPECT_RATIO,
        max_aspect_ratio: float = MAX_ASPECT_RATIO
    ):
        """
        Initialize class-agnostic YOLOv8 detector.

        Args:
            model_path: YOLOv8 model weights (auto-downloads if missing)
            conf_threshold: Detection confidence threshold (default 0.15)
            iou_threshold: NMS IoU threshold (default 0.45)
            max_detections: Max detections to return (default 8)
            crop_padding: Padding percentage for crops (default 0.10)
            min_bbox_area_pct: Minimum bbox area as % of image (default 0.02)
            min_aspect_ratio: Minimum width/height ratio (default 0.2)
            max_aspect_ratio: Maximum width/height ratio (default 5.0)
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

        # Geometric filter parameters (non-semantic noise reduction)
        self.min_bbox_area_pct = min_bbox_area_pct
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def _passes_geometric_filter(
        self, width: int, height: int, image_area: int
    ) -> bool:
        """
        Check if detection passes geometric filters (non-semantic).

        Args:
            width: Bounding box width
            height: Bounding box height
            image_area: Total image area (width * height)

        Returns:
            True if detection passes all geometric constraints
        """
        # Filter 1: Minimum area (removes tiny/noise detections)
        bbox_area = width * height
        area_pct = bbox_area / image_area
        if area_pct < self.min_bbox_area_pct:
            return False

        # Filter 2: Aspect ratio (removes elongated/edge detections)
        if height == 0:
            return False
        aspect_ratio = width / height
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False

        return True

    def detect(self, image_path: str) -> List[dict]:
        """
        Detect objects and return cropped regions (class labels ignored).

        Applies geometric filtering to reduce noise from non-food detections.
        NO semantic reasoning - only structural/geometric constraints.

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

        image_area = image.width * image.height

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

                # Apply geometric filter (non-semantic noise reduction)
                if not self._passes_geometric_filter(width, height, image_area):
                    continue

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
