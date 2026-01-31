"""
Class-agnostic YOLO detector for structural pre-processing (Pipeline B).

Uses standard YOLOv8 (COCO-trained) with class labels IGNORED.
All detections treated as generic "object" regions for LLM classification.

EXPERIMENTAL DESIGN NOTE:
This detector provides STRUCTURAL pre-processing only:
- Detects any object that YOLO was trained on (80 COCO classes)
- Class labels are intentionally discarded - only bounding boxes used
- Tests hypothesis: "Does ANY region proposal help the LLM?"

Compare with Pipeline C (YOLO-World) which uses SEMANTIC pre-processing:
- Detects only food-related objects via text prompts
- Tests hypothesis: "Does FOOD-SPECIFIC region proposal help more?"
"""

import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

from config import CONFIG, get_logger, Timer, PipelineLog


# =============================================================================
# YOLO DETECTOR (Singleton pattern for timing fairness)
# =============================================================================

class YOLODetectorAgnostic:
    """
    Class-agnostic YOLOv8 detector for region proposals.

    Uses standard COCO-trained YOLOv8 but ignores class labels.
    Returns cropped regions for LLM classification (structural pre-processing only).
    Applies geometric filtering to reduce noise (no semantic reasoning).

    Uses singleton pattern to avoid model loading during timed pipeline runs.
    """

    _instance: Optional["YOLODetectorAgnostic"] = None
    _initialized: bool = False

    def __new__(cls) -> "YOLODetectorAgnostic":
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize YOLOv8 model (only once)."""
        if YOLODetectorAgnostic._initialized:
            return

        model_path = CONFIG.yolo_standard_model

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
        self.model_path = resolved_path

        # Store config values
        self.conf_threshold = CONFIG.yolo_conf_threshold
        self.iou_threshold = CONFIG.yolo_iou_threshold
        self.max_detections = CONFIG.yolo_max_detections
        self.crop_padding = CONFIG.yolo_crop_padding

        # Geometric filter parameters (non-semantic noise reduction)
        self.min_bbox_area_pct = CONFIG.min_bbox_area_pct
        self.min_aspect_ratio = CONFIG.min_aspect_ratio
        self.max_aspect_ratio = CONFIG.max_aspect_ratio

        YOLODetectorAgnostic._initialized = True

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

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
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
        logger = get_logger()
        image_name = Path(image_path).name

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_area = image.width * image.height

        # Run YOLOv8 detection
        with Timer("yolo_inference") as timer:
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )

        detections = []
        filtered_count = 0

        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Class label intentionally ignored (class-agnostic)
                # All detections treated as generic "object"

                width, height = x2 - x1, y2 - y1

                # Check geometric filter
                passes_filter = self._passes_geometric_filter(width, height, image_area)

                # Log every detection (for analysis)
                logger.log_detection(
                    pipeline="yolo",
                    image=image_name,
                    bbox={"x": x1, "y": y1, "width": width, "height": height},
                    confidence=conf,
                    passed_filter=passes_filter,
                    prompt_match="object"
                )

                if not passes_filter:
                    filtered_count += 1
                    continue

                # Add padding to capture full object
                pad_x = int(width * self.crop_padding)
                pad_y = int(height * self.crop_padding)
                x1_padded = max(0, x1 - pad_x)
                y1_padded = max(0, y1 - pad_y)
                x2_padded = min(image.width, x2 + pad_x)
                y2_padded = min(image.height, y2 + pad_y)

                # Crop the region
                crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))

                # Convert to bytes
                buffer = BytesIO()
                crop.save(buffer, format="PNG")

                detections.append({
                    "bbox": {
                        "x": x1_padded,
                        "y": y1_padded,
                        "width": x2_padded - x1_padded,
                        "height": y2_padded - y1_padded
                    },
                    "image_bytes": buffer.getvalue(),
                    "confidence": conf,
                    "prompt_match": "object"  # Class-agnostic: always "object"
                })

        # Log summary
        logger.log(PipelineLog(
            pipeline="yolo",
            image=image_name,
            step="detection_complete",
            duration_ms=timer.duration_ms,
            details={
                "total_raw_detections": len(results[0].boxes) if results else 0,
                "filtered_out": filtered_count,
                "final_detections": len(detections)
            }
        ))

        return detections


# =============================================================================
# MODULE-LEVEL ACCESS (for timing fairness)
# =============================================================================

def get_yolo_agnostic_detector() -> YOLODetectorAgnostic:
    """
    Get the singleton YOLO detector instance.
    Use this to ensure model loading happens before timing starts.
    """
    return YOLODetectorAgnostic()


def warmup_yolo_agnostic():
    """
    Warm up the YOLO detector by loading the model.
    Call this before timing measurements to exclude model loading time.
    """
    _ = get_yolo_agnostic_detector()
