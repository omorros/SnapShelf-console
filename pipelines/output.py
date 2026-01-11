"""
Shared output schema for pipeline results.
Frozen schema for fair research comparison.
"""

from typing import List, TypedDict


class ItemResult(TypedDict):
    """Single detected food item."""
    name: str  # Generic name (lowercase, normalized)
    state: str  # fresh | packaged | cooked | unknown


class PipelineMeta(TypedDict):
    """Pipeline execution metadata."""
    pipeline: str  # "llm" or "yolo-llm"
    image: str  # Filename
    runtime_ms: float  # Execution time in milliseconds
    fallback_used: bool  # True if YOLO fallback was triggered (System B only)


class PipelineResult(TypedDict):
    """Standard output schema for both systems."""
    items: List[ItemResult]
    meta: PipelineMeta


def make_result(
    items: List[ItemResult],
    pipeline: str,
    image: str,
    runtime_ms: float,
    fallback_used: bool = False
) -> PipelineResult:
    """
    Create standardized pipeline result.

    Args:
        items: List of detected items
        pipeline: "llm" or "yolo-llm"
        image: Image filename
        runtime_ms: Execution time in milliseconds
        fallback_used: Whether YOLO fallback was triggered

    Returns:
        PipelineResult dict
    """
    return {
        "items": items,
        "meta": {
            "pipeline": pipeline,
            "image": image,
            "runtime_ms": round(runtime_ms, 2),
            "fallback_used": fallback_used
        }
    }
