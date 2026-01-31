"""
Pipeline B: Class-agnostic YOLO + LLM hybrid pipeline.

Standard YOLOv8 proposes regions (class labels IGNORED), LLM identifies per crop.
Structural pre-processing only (no semantic prompts).

EXPERIMENTAL DESIGN:
- Tests hypothesis: "Does ANY region proposal help the LLM?"
- YOLO detects objects from 80 COCO classes but labels are discarded
- All detections treated as generic "object" regions
- Compare with Pipeline A (no pre-processing) and C (semantic pre-processing)

FAIRNESS NOTE:
- Fallback is DISABLED for experimental validity
- If YOLO detects nothing, returns empty result (no full-image LLM call)
- Geometric filters identical to Pipeline C
"""

from pathlib import Path
from typing import List

from clients.yolo_detector_agnostic import get_yolo_agnostic_detector
from clients.llm_client import get_llm_client
from pipelines.output import PipelineResult, ItemResult, make_result
from config import get_logger, Timer, PipelineLog


def run(image_path: str) -> PipelineResult:
    """
    Execute class-agnostic YOLO + LLM hybrid pipeline (Pipeline B).

    Pipeline:
        1. YOLOv8 proposes regions (class labels ignored, geometric filtering only)
        2. LLM identifies food in each crop
        3. Results aggregated (deduplicated by name)

    TIMING NOTE:
    - YOLO and LLM models are pre-initialized (singletons) - init time excluded
    - Only inference time is measured
    - Per-component timing breakdown included for analysis

    Fallback is DISABLED for experimental fairness.
    If YOLO detects nothing, returns empty result.

    Args:
        image_path: Path to image file

    Returns:
        PipelineResult with detected items and metadata
    """
    logger = get_logger()
    image_name = Path(image_path).name

    # Get pre-initialized clients (singletons - no init overhead)
    detector = get_yolo_agnostic_detector()
    llm = get_llm_client()

    # Timing breakdown
    detection_time = 0.0
    llm_times: List[float] = []

    # Step 1: Class-agnostic YOLO region proposals (timed separately)
    with Timer("detection") as det_timer:
        detections = detector.detect(image_path)
    detection_time = det_timer.duration_ms
    detections_count = len(detections)

    # Handle no detections (strict mode: return empty, NO FALLBACK)
    if not detections:
        logger.log(PipelineLog(
            pipeline="yolo",
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
            pipeline="yolo",
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
    with Timer("pipeline_b_total") as total_timer:

        # Step 2: LLM per crop
        raw_items: List[ItemResult] = []

        for i, detection in enumerate(detections):
            result, llm_time = llm.identify_single(
                detection["image_bytes"],
                pipeline="yolo",
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
        pipeline="yolo",
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
        pipeline="yolo",
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

    NOTE: This deduplication is necessary because multiple YOLO crops
    may contain the same food item (e.g., overlapping detections).
    Pipeline A does not need this as the LLM sees the full image once.
    """
    seen = {}
    for item in items:
        key = item["name"].lower()
        if key not in seen:
            seen[key] = item
    return list(seen.values())
