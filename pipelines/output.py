"""
Shared output schema for pipeline results.

FROZEN SCHEMA for fair research comparison.
All pipelines output Inventory = Dict[str, int] (class -> count).
"""

from typing import Dict, TypedDict, Optional, Any


# Universal output type: class name -> count
Inventory = Dict[str, int]  # e.g. {"apple": 3, "banana": 1, "tomato": 2}


class TimingBreakdown(TypedDict, total=False):
    """Detailed timing information for analysis."""
    total_ms: float
    detection_ms: float          # YOLO inference (Pipelines B/C)
    classification_ms: float     # CNN inference (Pipeline C) or VLM call (Pipeline A)
    vlm_call_ms: float           # VLM API call time (Pipeline A only)


class PipelineMeta(TypedDict, total=False):
    """Pipeline execution metadata."""
    pipeline: str             # "vlm" | "yolo-14" | "yolo-cnn"
    image: str                # Filename
    runtime_ms: float         # Total execution time in milliseconds
    timing_breakdown: TimingBreakdown
    detections_count: int     # Number of YOLO detections (Pipeline B/C)


class PipelineResult(TypedDict):
    """Standard output schema for all pipelines."""
    inventory: Inventory
    meta: PipelineMeta


def make_result(
    inventory: Inventory,
    pipeline: str,
    image: str,
    runtime_ms: float,
    timing_breakdown: Optional[Dict[str, Any]] = None,
    detections_count: Optional[int] = None
) -> PipelineResult:
    """
    Create standardized pipeline result.

    Args:
        inventory: Dict mapping class name -> count.
        pipeline: "vlm" | "yolo-14" | "yolo-cnn"
        image: Image filename.
        runtime_ms: Total execution time in milliseconds.
        timing_breakdown: Detailed timing for analysis.
        detections_count: Number of YOLO detections (Pipeline B/C).

    Returns:
        PipelineResult dict.
    """
    meta: PipelineMeta = {
        "pipeline": pipeline,
        "image": image,
        "runtime_ms": round(runtime_ms, 2),
    }

    if timing_breakdown:
        meta["timing_breakdown"] = timing_breakdown

    if detections_count is not None:
        meta["detections_count"] = detections_count

    return {
        "inventory": inventory,
        "meta": meta
    }
