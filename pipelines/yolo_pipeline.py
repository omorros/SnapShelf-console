"""
Pipeline B: YOLO end-to-end (14-class YOLO -> inventory).

Image -> 14-class YOLO -> boxes + labels -> inventory.
No LLM calls. Pure detection-based counting.
"""

from pathlib import Path

from clients.yolo_detector import detect, detections_to_inventory
from pipelines.output import PipelineResult, make_result
from config import Timer


def run(image_path: str) -> PipelineResult:
    """
    Execute YOLO end-to-end pipeline.

    Args:
        image_path: Path to image file.

    Returns:
        PipelineResult with inventory and metadata.
    """
    image_name = Path(image_path).name

    with Timer("yolo_pipeline") as total:
        with Timer("detection") as det_timer:
            detections = detect(image_path)
        inventory = detections_to_inventory(detections)

    return make_result(
        inventory=inventory,
        pipeline="yolo-14",
        image=image_name,
        runtime_ms=total.duration_ms,
        timing_breakdown={
            "detection_ms": round(det_timer.duration_ms, 2),
            "total_ms": round(total.duration_ms, 2),
        },
        detections_count=len(detections),
    )
