"""
Pipeline A: VLM-only (GPT-4o-mini constrained to 14 labels).

Image -> GPT-4o-mini -> inventory.
Single API call, no detection or pre-processing.
"""

from pathlib import Path

from clients.vlm_client import identify
from pipelines.output import PipelineResult, make_result
from config import Timer


def run(image_path: str) -> PipelineResult:
    """
    Execute VLM-only pipeline.

    Args:
        image_path: Path to image file.

    Returns:
        PipelineResult with inventory and metadata.
    """
    image_name = Path(image_path).name

    with Timer("vlm_pipeline") as total:
        with Timer("vlm_call") as vlm_timer:
            inventory = identify(image_path)

    return make_result(
        inventory=inventory,
        pipeline="vlm",
        image=image_name,
        runtime_ms=total.duration_ms,
        timing_breakdown={
            "vlm_call_ms": round(vlm_timer.duration_ms, 2),
            "total_ms": round(total.duration_ms, 2),
        },
    )
