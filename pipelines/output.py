"""
Shared output schema for pipeline results.

FROZEN SCHEMA for fair research comparison.
All pipelines must output in this exact format for valid comparison.
"""

from typing import List, TypedDict, Optional, Dict, Any


class ItemResult(TypedDict):
    """Single detected food item."""
    name: str  # Generic name (lowercase, normalized)
    state: str  # fresh | packaged | cooked | unknown


class TimingBreakdown(TypedDict, total=False):
    """Detailed timing information for analysis."""
    image_load_ms: float  # Time to load and convert image (Pipeline A)
    detection_ms: float  # Time for YOLO detection (Pipeline B/C)
    llm_inference_ms: float  # Total LLM time (Pipeline A - single call)
    llm_total_ms: float  # Total LLM time (Pipeline B/C - all crops)
    llm_avg_ms: float  # Average LLM time per crop (Pipeline B/C)
    llm_calls: int  # Number of LLM API calls


class PipelineMeta(TypedDict, total=False):
    """Pipeline execution metadata."""
    pipeline: str  # "llm" | "yolo" | "yolo-world"
    image: str  # Filename
    runtime_ms: float  # Total execution time in milliseconds
    timing_breakdown: TimingBreakdown  # Detailed timing for analysis
    fallback_used: bool  # True if YOLO fallback was triggered (Pipeline B/C only)
    detections_count: int  # Number of YOLO detections (Pipeline B/C only)


class PipelineResult(TypedDict):
    """Standard output schema for all pipelines."""
    items: List[ItemResult]
    meta: PipelineMeta


def make_result(
    items: List[ItemResult],
    pipeline: str,
    image: str,
    runtime_ms: float,
    timing_breakdown: Optional[Dict[str, Any]] = None,
    fallback_used: bool = False,
    detections_count: Optional[int] = None
) -> PipelineResult:
    """
    Create standardized pipeline result.

    Args:
        items: List of detected items
        pipeline: "llm" | "yolo" | "yolo-world"
        image: Image filename
        runtime_ms: Total execution time in milliseconds
        timing_breakdown: Detailed timing for analysis
        fallback_used: Whether YOLO fallback was triggered
        detections_count: Number of YOLO detections (Pipeline B/C only)

    Returns:
        PipelineResult dict
    """
    meta: PipelineMeta = {
        "pipeline": pipeline,
        "image": image,
        "runtime_ms": round(runtime_ms, 2),
        "fallback_used": fallback_used
    }

    # Include timing breakdown if provided
    if timing_breakdown:
        meta["timing_breakdown"] = timing_breakdown

    # Only include detections_count for YOLO pipelines
    if detections_count is not None:
        meta["detections_count"] = detections_count

    return {
        "items": items,
        "meta": meta
    }
