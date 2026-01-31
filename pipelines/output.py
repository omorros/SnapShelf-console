"""
Shared output schema for pipeline results.
Frozen schema for fair research comparison.
"""

from typing import List, TypedDict


class ItemResult(TypedDict):
    """Single detected food item."""
    name: str  # Generic name (lowercase, normalized)
    state: str  # fresh | packaged | cooked | unknown


class PipelineMeta(TypedDict, total=False):
    """Pipeline execution metadata."""
    pipeline: str  # "llm" | "yolo" | "yolo-world"
    image: str  # Filename
    runtime_ms: float  # Execution time in milliseconds
    fallback_used: bool  # True if YOLO fallback was triggered (Pipeline B/C only)
    detections_count: int  # Number of YOLO detections (Pipeline B/C only)


class PipelineResult(TypedDict):
    """Standard output schema for both systems."""
    items: List[ItemResult]
    meta: PipelineMeta


def make_result(
    items: List[ItemResult],
    pipeline: str,
    image: str,
    runtime_ms: float,
    fallback_used: bool = False,
    detections_count: int = None
) -> PipelineResult:
    """
    Create standardized pipeline result.

    Args:
        items: List of detected items
        pipeline: "llm" | "yolo" | "yolo-world"
        image: Image filename
        runtime_ms: Execution time in milliseconds
        fallback_used: Whether YOLO fallback was triggered
        detections_count: Number of YOLO detections (Pipeline B/C only)

    Returns:
        PipelineResult dict
    """
    meta = {
        "pipeline": pipeline,
        "image": image,
        "runtime_ms": round(runtime_ms, 2),
        "fallback_used": fallback_used
    }

    # Only include detections_count for YOLO pipelines
    if detections_count is not None:
        meta["detections_count"] = detections_count

    return {
        "items": items,
        "meta": meta
    }
