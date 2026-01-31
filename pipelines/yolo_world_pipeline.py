"""
Pipeline C: YOLO-World + LLM hybrid pipeline.

YOLO-World proposes regions with SEMANTIC food prompts, LLM identifies per crop.

EXPERIMENTAL DESIGN:
- Tests hypothesis: "Does FOOD-SPECIFIC region proposal help more than blind detection?"
- Uses fixed prompts: ["food", "fruit", "vegetable", "packaged food"]
- Only detects objects matching these food-related categories
- Compare with Pipeline A (no pre-processing) and B (blind pre-processing)

FAIRNESS NOTES:
- Prompts are FIXED and NEVER change per image (critical for validity)
- Fallback is DISABLED for experimental validity
- If YOLO-World detects nothing, returns empty result (no full-image LLM call)
- Geometric filters identical to Pipeline B
- All other parameters (conf, iou, max_det, padding) identical to Pipeline B
"""

from pathlib import Path
from typing import List

from clients.yolo_detector import get_yolo_world_detector
from clients.llm_client import get_llm_client
from pipelines.output import PipelineResult, ItemResult, make_result
from config import get_logger, Timer, PipelineLog


def run(image_path: str) -> PipelineResult:
    """
    Execute YOLO-World + LLM hybrid pipeline (Pipeline C).

    Pipeline:
        1. YOLO-World proposes regions using semantic prompts
        2. LLM identifies food in each crop
        3. Results aggregated (deduplicated by name)

    TIMING NOTE:
    - YOLO-World and LLM models are pre-initialized (singletons) - init time excluded
    - Only inference time is measured
    - Per-component timing breakdown included for analysis

    Fallback is DISABLED for experimental fairness.
    If YOLO-World detects nothing, returns empty result.

    Args:
        image_path: Path to image file

    Returns:
        PipelineResult with detected items and metadata
    """
    logger = get_logger()
    image_name = Path(image_path).name

    # Get pre-initialized clients (singletons - no init overhead)
    detector = get_yolo_world_detector()
    llm = get_llm_client()

    # Timing breakdown
    detection_time = 0.0
    llm_times: List[float] = []

    # Step 1: YOLO-World region proposals (timed separately)
    with Timer("detection") as det_timer:
        detections = detector.detect(image_path)
    detection_time = det_timer.duration_ms
    detections_count = len(detections)

    # Handle no detections (strict mode: return empty, NO FALLBACK)
    if not detections:
        logger.log(PipelineLog(
            pipeline="yolo-world",
            image=image_name,
            step="pipeline_complete",
            duration_ms=detection_time,
            details={
                "detection_ms": detection_time,
                "detections_count": 0,
                "llm_calls": 0,
                "items_detected": 0,
                "fallback_used": False
            }
        ))

        return make_result(
            items=[],
            pipeline="yolo-world",
            image=image_name,
            runtime_ms=detection_time,
            timing_breakdown={
                "detection_ms": round(detection_time, 2),
                "llm_total_ms": 0,
                "llm_calls": 0
            },
            fallback_used=False,
            detections_count=0
        )

    # Start timing for LLM phase
    with Timer("pipeline_c_total") as total_timer:

        # Step 2: LLM per crop
        raw_items: List[ItemResult] = []

        for i, detection in enumerate(detections):
            result, llm_time = llm.identify_single(
                detection["image_bytes"],
                pipeline="yolo-world",
                image_name=image_name,
                crop_index=i
            )
            llm_times.append(llm_time)

            if result is not None:
                raw_items.append(result)

        # Step 3: Aggregate (deduplicate by name)
        items = _deduplicate(raw_items)

    # Calculate timing stats
    total_llm_time = sum(llm_times)
    avg_llm_time = total_llm_time / len(llm_times) if llm_times else 0
    total_runtime = detection_time + total_timer.duration_ms  # Detection + LLM

    # Log pipeline execution
    logger.log(PipelineLog(
        pipeline="yolo-world",
        image=image_name,
        step="pipeline_complete",
        duration_ms=total_runtime,
        details={
            "detection_ms": detection_time,
            "detections_count": detections_count,
            "llm_calls": len(llm_times),
            "llm_total_ms": total_llm_time,
            "llm_avg_ms": avg_llm_time,
            "raw_items": len(raw_items),
            "items_after_dedup": len(items),
            "fallback_used": False
        }
    ))

    return make_result(
        items=items,
        pipeline="yolo-world",
        image=image_name,
        runtime_ms=total_runtime,
        timing_breakdown={
            "detection_ms": round(detection_time, 2),
            "llm_total_ms": round(total_llm_time, 2),
            "llm_avg_ms": round(avg_llm_time, 2),
            "llm_calls": len(llm_times)
        },
        fallback_used=False,
        detections_count=detections_count
    )


def _deduplicate(items: List[ItemResult]) -> List[ItemResult]:
    """
    Deduplicate items by name (case-insensitive, already normalized).

    NOTE: This deduplication is necessary because multiple YOLO-World crops
    may contain the same food item (e.g., overlapping detections).
    Pipeline A does not need this as the LLM sees the full image once.
    """
    seen = {}
    for item in items:
        key = item["name"].lower()
        if key not in seen:
            seen[key] = item
    return list(seen.values())
