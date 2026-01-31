"""
Pipeline C: YOLO-World + LLM hybrid pipeline.
YOLO-World proposes regions with semantic prompts, LLM identifies per crop.
"""

import time
from pathlib import Path
from typing import List

from clients.yolo_detector import YOLODetector
from clients.llm_client import LLMClient
from pipelines.output import PipelineResult, ItemResult, make_result


def run(image_path: str) -> PipelineResult:
    """
    Execute YOLO-World + LLM hybrid pipeline (Pipeline C).

    Pipeline:
        1. YOLO-World proposes regions using semantic prompts
        2. LLM identifies food in each crop
        3. Results aggregated (deduplicated by name)

    Fallback is DISABLED for experimental fairness.
    If YOLO-World detects nothing, returns empty result.

    Args:
        image_path: Path to image file

    Returns:
        PipelineResult with detected items and metadata
    """
    start_time = time.perf_counter()

    # Step 1: YOLO-World region proposals (semantic prompts)
    detector = YOLODetector()
    detections = detector.detect(image_path)
    detections_count = len(detections)

    # Handle no detections (strict mode: return empty, NO fallback)
    if not detections:
        runtime_ms = (time.perf_counter() - start_time) * 1000
        return make_result(
            items=[],
            pipeline="yolo-world",
            image=Path(image_path).name,
            runtime_ms=runtime_ms,
            fallback_used=False,
            detections_count=0
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
        pipeline="yolo-world",
        image=Path(image_path).name,
        runtime_ms=runtime_ms,
        fallback_used=False,
        detections_count=detections_count
    )


def _deduplicate(items: List[ItemResult]) -> List[ItemResult]:
    """Deduplicate items by name (case-insensitive, already normalized)."""
    seen = {}
    for item in items:
        key = item["name"].lower()
        if key not in seen:
            seen[key] = item
    return list(seen.values())
