"""
System B: YOLO + LLM hybrid pipeline.
YOLO proposes regions, LLM identifies per crop, results aggregated.
"""

import time
from pathlib import Path
from typing import List

from clients.yolo_detector import YOLODetector
from clients.llm_client import LLMClient
from pipelines.output import PipelineResult, ItemResult, make_result


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fallback behavior when YOLO finds no detections:
# - True: Use full image as single region (can contaminate comparison)
# - False: Return empty items list (strict mode for fair comparison)
USE_FALLBACK = True


def run(image_path: str, use_fallback: bool = USE_FALLBACK) -> PipelineResult:
    """
    Execute YOLO-LLM hybrid pipeline (System B).

    Pipeline:
        1. YOLO proposes regions (class labels ignored)
        2. LLM identifies food in each crop
        3. Results aggregated (deduplicated by name)

    Args:
        image_path: Path to image file
        use_fallback: If True, use full image when YOLO finds nothing

    Returns:
        PipelineResult with detected items and metadata
    """
    start_time = time.perf_counter()
    fallback_used = False

    # Step 1: YOLO region proposals
    detector = YOLODetector()
    detections = detector.detect(image_path)

    # Handle no detections
    if not detections:
        if use_fallback:
            # Fallback: use full image (logged for transparency)
            detections = detector.get_full_image_fallback(image_path)
            fallback_used = True
        else:
            # Strict mode: return empty
            runtime_ms = (time.perf_counter() - start_time) * 1000
            return make_result(
                items=[],
                pipeline="yolo-llm",
                image=Path(image_path).name,
                runtime_ms=runtime_ms,
                fallback_used=False
            )

    # Step 2: LLM per crop
    llm = LLMClient()
    raw_items: List[ItemResult] = []

    for detection in detections:
        result = llm.identify_single(detection["image_bytes"])
        if result is not None:
            raw_items.append(result)

    # Step 3: Aggregate (deduplicate by name)
    items = _deduplicate(raw_items)

    runtime_ms = (time.perf_counter() - start_time) * 1000

    return make_result(
        items=items,
        pipeline="yolo-llm",
        image=Path(image_path).name,
        runtime_ms=runtime_ms,
        fallback_used=fallback_used
    )


def _deduplicate(items: List[ItemResult]) -> List[ItemResult]:
    """Deduplicate items by name (case-insensitive, already normalized)."""
    seen = {}
    for item in items:
        key = item["name"].lower()
        if key not in seen:
            seen[key] = item
    return list(seen.values())
