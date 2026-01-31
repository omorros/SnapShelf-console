"""
Pipeline A: LLM-only pipeline (baseline).

One image in, one structured JSON out.
No pre-processing - sends full image directly to LLM.

This serves as the BASELINE for comparison:
- Measures LLM's native multi-item detection capability
- All other pipelines should be compared against this
"""

from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import List

from clients.llm_client import get_llm_client
from pipelines.output import PipelineResult, ItemResult, make_result
from config import get_logger, Timer, PipelineLog


def run(image_path: str) -> PipelineResult:
    """
    Execute LLM-only pipeline (Pipeline A / Baseline).

    Pipeline: Full image -> LLM identifies all items -> JSON output

    TIMING NOTE:
    - LLM client is pre-initialized (singleton) - init time excluded
    - Only inference time is measured
    - Image loading time IS included (fair - all pipelines load images)

    Args:
        image_path: Path to image file

    Returns:
        PipelineResult with detected items and metadata
    """
    logger = get_logger()
    image_name = Path(image_path).name

    # Get pre-initialized LLM client (singleton - no init overhead)
    llm = get_llm_client()

    # Start timing AFTER client is ready
    with Timer("pipeline_a_total") as total_timer:
        # Load and convert image
        with Timer("image_load") as load_timer:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        # Single LLM call for all items
        items, llm_time = llm.identify_all(
            image_bytes,
            pipeline="llm",
            image_name=image_name
        )

    # Log pipeline execution
    logger.log(PipelineLog(
        pipeline="llm",
        image=image_name,
        step="pipeline_complete",
        duration_ms=total_timer.duration_ms,
        details={
            "image_load_ms": load_timer.duration_ms,
            "llm_inference_ms": llm_time,
            "items_detected": len(items)
        }
    ))

    return make_result(
        items=items,
        pipeline="llm",
        image=image_name,
        runtime_ms=total_timer.duration_ms,
        timing_breakdown={
            "image_load_ms": round(load_timer.duration_ms, 2),
            "llm_inference_ms": round(llm_time, 2)
        },
        fallback_used=False  # N/A for Pipeline A, always False
    )
