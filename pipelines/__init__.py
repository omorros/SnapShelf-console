"""
Pipeline modules for food detection comparison.

This package contains three pipelines for comparing vision pre-processing approaches:
- Pipeline A (llm_pipeline): Raw image -> LLM (baseline)
- Pipeline B (yolo_agnostic_pipeline): Class-agnostic YOLO -> crops -> LLM
- Pipeline C (yolo_world_pipeline): YOLO-World with food prompts -> crops -> LLM

EXPERIMENTAL DESIGN NOTE:
Pipelines B and C use intentionally different detection strategies:
- B: Structural pre-processing only (any detected object becomes a crop)
- C: Semantic pre-processing (only food-related detections become crops)

This tests two hypotheses:
1. Does ANY region proposal help the LLM? (A vs B)
2. Does SEMANTIC region proposal help more than blind proposal? (B vs C)
"""

from pipelines import llm_pipeline
from pipelines import yolo_agnostic_pipeline
from pipelines import yolo_world_pipeline

__all__ = ["llm_pipeline", "yolo_agnostic_pipeline", "yolo_world_pipeline"]
