"""
Food Item data model for SnapShelf inventory.
Defines the structure and methods for food items tracked in the system.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import uuid


@dataclass
class FoodItem:
    """
    Represents a food item in the inventory.

    Attributes:
        id: Unique identifier for the item
        name: Specific name of the food (e.g., 'Granny Smith Apple')
        category: Food category (Fruit, Vegetable, Dairy, etc.)
        quantity: Amount of the item
        unit: Unit of measurement (item, kg, lbs, etc.)
        expiry_date: Expected expiry date
        freshness: Current freshness state (Fresh, Good, Fair, Poor)
        storage_tip: Recommended storage instructions
        confidence: AI confidence score (0-1)
        status: Item status (active, consumed, discarded)
        added_at: Timestamp when item was added
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    category: str = ""
    quantity: float = 1.0
    unit: str = "item"
    expiry_date: Optional[date] = None
    freshness: str = "fresh"
    storage_tip: str = ""
    confidence: float = 0.0
    status: str = "active"
    added_at: datetime = field(default_factory=datetime.now)

    def days_until_expiry(self) -> int:
        """
        Calculate days remaining until expiry.

        Returns:
            Number of days until expiry (999 if no expiry date set)
        """
        if not self.expiry_date:
            return 999
        return (self.expiry_date - date.today()).days

    def expiry_status(self) -> str:
        """
        Determine expiry urgency level.

        Returns:
            Status string: expired, urgent, warning, good, or fresh
        """
        days = self.days_until_expiry()
        if days < 0:
            return "expired"
        elif days <= 1:
            return "urgent"
        elif days <= 3:
            return "warning"
        elif days <= 5:
            return "good"
        else:
            return "fresh"

    def to_dict(self) -> dict:
        """
        Convert FoodItem to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the food item
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "quantity": self.quantity,
            "unit": self.unit,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "freshness": self.freshness,
            "storage_tip": self.storage_tip,
            "confidence": self.confidence,
            "status": self.status,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FoodItem":
        """
        Create FoodItem instance from dictionary.

        Args:
            data: Dictionary containing food item data

        Returns:
            New FoodItem instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data["name"],
            category=data["category"],
            quantity=data.get("quantity", 1.0),
            unit=data.get("unit", "item"),
            expiry_date=date.fromisoformat(data["expiry_date"]) if data.get("expiry_date") else None,
            freshness=data.get("freshness", "fresh"),
            storage_tip=data.get("storage_tip", ""),
            confidence=data.get("confidence", 0.0),
            status=data.get("status", "active"),
            added_at=datetime.fromisoformat(data["added_at"]) if data.get("added_at") else datetime.now(),
        )
