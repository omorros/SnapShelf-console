"""
Scan service orchestrator.
Coordinates YOLO detection and LLM identification to process food images.
"""

from datetime import date, timedelta
from typing import List
from services.yolo_service import YOLOService
from services.llm_service import LLMService
from models.food_item import FoodItem
from storage.inventory import InventoryStorage


class ScanService:
    """
    Orchestrates the full image scanning pipeline.

    Coordinates object detection, food identification, and inventory storage.

    Attributes:
        yolo: YOLO detection service
        llm: LLM food identification service
        storage: Inventory storage manager
    """

    def __init__(self):
        """Initialize scan service with all required components."""
        self.yolo = YOLOService()
        self.llm = LLMService()
        self.storage = InventoryStorage()

    def scan_image(self, image_path: str) -> List[FoodItem]:
        """
        Execute full scanning pipeline: detect → identify → save.

        Args:
            image_path: Path to the image file to scan

        Returns:
            List of identified and saved FoodItem objects
        """
        # Detect and identify items
        food_items = self.scan_image_preview(image_path)

        # Save all identified items to inventory
        if food_items:
            self.storage.add_many(food_items)

        return food_items

    def scan_image_preview(self, image_path: str) -> List[FoodItem]:
        """
        Detect and identify items without saving to inventory.

        Args:
            image_path: Path to the image file to scan

        Returns:
            List of identified FoodItem objects (not saved)
        """
        # Step 1: YOLO object detection
        detections = self.yolo.detect(image_path)

        # Step 2: LLM food identification for each detected region
        food_items = []

        for detection in detections:
            result = self.llm.identify_food(detection["image_bytes"])

            # If LLM identifies it as food, create FoodItem
            if result:
                # Calculate expiry date from estimated days
                expiry_date = date.today() + timedelta(days=result["expiry_days"])

                item = FoodItem(
                    name=result["name"],
                    category=result["category"],
                    freshness=result["freshness"],
                    expiry_date=expiry_date,
                    unit=result.get("unit", "unit"),
                    storage_tip=result.get("storage_tip", ""),
                    confidence=result.get("confidence", 0.8)
                )
                food_items.append(item)

        return food_items

    def save_items(self, items: List[FoodItem]) -> None:
        """
        Save items to inventory.

        Args:
            items: List of FoodItem objects to save
        """
        if items:
            self.storage.add_many(items)
