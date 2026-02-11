"""
Pipeline modules for Experiment 2: End-to-End Pipeline Comparison.

Three fundamentally different end-to-end pipelines for 14-class inventory:
- Pipeline A (vlm_pipeline):      VLM-only (GPT-4o-mini constrained to 14 labels)
- Pipeline B (yolo_pipeline):     YOLO end-to-end (14-class YOLO)
- Pipeline C (yolo_cnn_pipeline): YOLO + CNN (objectness YOLO -> crops -> CNN)

Key: Pipelines B and C use NO LLM calls.
Comparison: VLM vs. pure detection vs. detect-then-classify.
"""

from pipelines import vlm_pipeline
from pipelines import yolo_pipeline
from pipelines import yolo_cnn_pipeline

__all__ = ["vlm_pipeline", "yolo_pipeline", "yolo_cnn_pipeline"]
