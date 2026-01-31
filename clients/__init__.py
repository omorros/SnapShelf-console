"""
External integration clients for SnapShelf.

Contains:
- LLMClient: OpenAI Vision API client with frozen prompts
- YOLODetector: YOLO-World open-vocabulary detector (Pipeline C)
- YOLODetectorAgnostic: Class-agnostic YOLOv8 detector (Pipeline B)
"""

from clients.llm_client import LLMClient
from clients.yolo_detector import YOLODetector
from clients.yolo_detector_agnostic import YOLODetectorAgnostic

__all__ = ["LLMClient", "YOLODetector", "YOLODetectorAgnostic"]
