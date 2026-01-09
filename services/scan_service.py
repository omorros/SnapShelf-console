"""
Scan service orchestrator.
Uses LLM vision to identify all food items in images.
"""

from datetime import date, timedelta
from typing import List
from PIL import Image
from io import BytesIO
from services.llm_service import LLMService
from models.food_item import FoodItem
from storage.inventory import InventoryStorage


class ScanService:
    """
    Orchestrates the full image scanning pipeline.

    Uses LLM to identify all food items in a single pass.

    Attributes:
        llm: LLM food identification service
        storage: Inventory storage manager
    """

    def __init__(self):
        """Initialize scan service with all required components."""
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
        Identify all food items in image without saving to inventory.

        Pipeline: Image → LLM identifies ALL foods → User review

        Args:
            image_path: Path to the image file to scan

        Returns:
            List of identified FoodItem objects (not saved)
        """
        # Load image and convert to bytes
        image = Image.open(image_path)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # LLM identifies ALL food items in one pass
        results = self.llm.identify_all_foods(image_bytes)

        # Convert results to FoodItem objects
        food_items = []
        for result in results:
            expiry_days = result.get("expiry_days", 7)
            expiry_date = date.today() + timedelta(days=expiry_days)

            item = FoodItem(
                name=result.get("name", "Unknown"),
                category=result.get("category", "Other"),
                freshness=result.get("freshness", "Fresh"),
                expiry_date=expiry_date,
                quantity=1,  # Default, user sets actual quantity
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
