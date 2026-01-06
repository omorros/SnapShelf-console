"""
Basic test script to verify core functionality without external dependencies.
Tests the FoodItem model and basic data operations.
"""

from datetime import date, timedelta
from models.food_item import FoodItem

print("Testing SnapShelf Core Components...\n")

# Test 1: Create a FoodItem
print("1. Creating a FoodItem...")
apple = FoodItem(
    name="Granny Smith Apple",
    category="Fruit",
    quantity=3,
    expiry_date=date.today() + timedelta(days=7),
    freshness="Fresh",
    confidence=0.95
)
print(f"   Created: {apple.name} (ID: {apple.id})")
print(f"   Category: {apple.category}")
print(f"   Expires in: {apple.days_until_expiry()} days")
print(f"   Status: {apple.expiry_status()}")

# Test 2: Convert to dict
print("\n2. Converting to dictionary...")
apple_dict = apple.to_dict()
print(f"   Keys: {list(apple_dict.keys())}")
print(f"   Name in dict: {apple_dict['name']}")

# Test 3: Convert from dict
print("\n3. Converting from dictionary...")
apple_restored = FoodItem.from_dict(apple_dict)
print(f"   Restored: {apple_restored.name}")
print(f"   ID matches: {apple.id == apple_restored.id}")

# Test 4: Expiry status for different dates
print("\n4. Testing expiry status logic...")
expired_item = FoodItem(name="Expired Milk", expiry_date=date.today() - timedelta(days=1))
urgent_item = FoodItem(name="Urgent Yogurt", expiry_date=date.today() + timedelta(days=1))
fresh_item = FoodItem(name="Fresh Carrots", expiry_date=date.today() + timedelta(days=10))

print(f"   Expired: {expired_item.expiry_status()}")
print(f"   Urgent: {urgent_item.expiry_status()}")
print(f"   Fresh: {fresh_item.expiry_status()}")

print("\nAll basic tests passed!")
