"""
Pipeline C: YOLO + CNN (objectness YOLO -> crops -> CNN classifier -> inventory).

Image -> 1-class objectness YOLO -> crops -> CNN winner from Exp 1 -> inventory.
No LLM calls. Detection is separated from classification.
"""

from pathlib import Path
from typing import Dict

from clients.yolo_objectness import detect, crop_detections
from clients.cnn_classifier import create_cnn_classifier
from pipelines.output import PipelineResult, make_result
from config import Timer


def run(image_path: str) -> PipelineResult:
    """
    Execute YOLO + CNN pipeline.

    Args:
        image_path: Path to image file.

    Returns:
        PipelineResult with inventory and metadata.
    """
    image_name = Path(image_path).name
    classifier = create_cnn_classifier()

    with Timer("yolo_cnn_pipeline") as total:
        # Step 1: Objectness detection
        with Timer("detection") as det_timer:
            detections = detect(image_path)

        # Step 2: Crop detected regions
        crops = crop_detections(image_path, detections)

        # Step 3: CNN classification
        with Timer("classification") as cls_timer:
            labels = classifier.predict_batch(crops)

        # Step 4: Build inventory from labels
        inventory: Dict[str, int] = {}
        for label in labels:
            inventory[label] = inventory.get(label, 0) + 1

    return make_result(
        inventory=inventory,
        pipeline="yolo-cnn",
        image=image_name,
        runtime_ms=total.duration_ms,
        timing_breakdown={
            "detection_ms": round(det_timer.duration_ms, 2),
            "classification_ms": round(cls_timer.duration_ms, 2),
            "total_ms": round(total.duration_ms, 2),
        },
        detections_count=len(detections),
    )
