"""
Storage layer test script.
Tests JSON file operations without external dependencies.
"""

from datetime import date, timedelta
from pathlib import Path
from models.food_item import FoodItem
from storage.inventory import InventoryStorage

print("Testing SnapShelf Storage Layer...\n")

# Use a test file so we don't interfere with real data
test_file = Path("data/test_inventory.json")
storage = InventoryStorage(filepath=test_file)

# Test 1: Initial state should be empty
print("1. Testing initial empty state...")
items = storage.get_all()
print(f"   Items in new inventory: {len(items)}")
assert len(items) == 0, "New inventory should be empty"
print("   OK: Inventory starts empty")

# Test 2: Add a single item
print("\n2. Adding a single item...")
apple = FoodItem(
    name="Granny Smith Apple",
    category="Fruit",
    quantity=3,
    expiry_date=date.today() + timedelta(days=7),
    freshness="Fresh"
)
storage.add(apple)
print(f"   Added: {apple.name} (ID: {apple.id})")

# Verify it was saved
items = storage.get_all()
print(f"   Items in inventory: {len(items)}")
assert len(items) == 1, "Should have 1 item"
print("   OK: Item saved successfully")

# Test 3: Add multiple items at once
print("\n3. Adding multiple items...")
milk = FoodItem(name="Whole Milk", category="Dairy", expiry_date=date.today() + timedelta(days=5))
carrots = FoodItem(name="Baby Carrots", category="Vegetable", expiry_date=date.today() + timedelta(days=10))
storage.add_many([milk, carrots])
print(f"   Added: {milk.name} and {carrots.name}")

items = storage.get_all()
print(f"   Total items: {len(items)}")
assert len(items) == 3, "Should have 3 items"
print("   OK: Multiple items added")

# Test 4: Get item by ID
print("\n4. Retrieving item by ID...")
retrieved = storage.get_by_id(apple.id)
print(f"   Retrieved: {retrieved.name}")
assert retrieved.name == apple.name, "Should retrieve correct item"
print("   OK: Item retrieved by ID")

# Test 5: Update item
print("\n5. Updating item status...")
storage.update(apple.id, {"status": "consumed"})
updated_item = storage.get_by_id(apple.id)
print(f"   Status changed to: {updated_item.status}")
assert updated_item.status == "consumed", "Status should be updated"
print("   OK: Item updated")

# Test 6: Filter by status
print("\n6. Filtering by status...")
active_items = storage.get_all(status="active")
consumed_items = storage.get_all(status="consumed")
print(f"   Active items: {len(active_items)}")
print(f"   Consumed items: {len(consumed_items)}")
assert len(active_items) == 2, "Should have 2 active items"
assert len(consumed_items) == 1, "Should have 1 consumed item"
print("   OK: Status filtering works")

# Test 7: Get expiring items
print("\n7. Testing expiring items filter...")
expiring = storage.get_expiring(days=7)
print(f"   Items expiring in 7 days: {len(expiring)}")
assert len(expiring) >= 1, "Should have items expiring soon"
print(f"   Found: {[item.name for item in expiring]}")
print("   OK: Expiring filter works")

# Test 8: Remove item
print("\n8. Removing item...")
removed = storage.remove(apple.id)
print(f"   Removed: {removed}")
assert removed == True, "Should return True when item removed"
remaining = storage.get_all(status=None)
print(f"   Remaining items: {len(remaining)}")
assert len(remaining) == 2, "Should have 2 items left"
print("   OK: Item removed")

# Test 9: Clear all items
print("\n9. Clearing inventory...")
storage.clear()
items = storage.get_all(status=None)
print(f"   Items after clear: {len(items)}")
assert len(items) == 0, "Should be empty after clear"
print("   OK: Inventory cleared")

# Test 10: Verify JSON file exists
print("\n10. Verifying JSON file...")
print(f"   File exists: {test_file.exists()}")
print(f"   File location: {test_file.absolute()}")
assert test_file.exists(), "JSON file should exist"
print("   OK: File created successfully")

# Cleanup
print("\n11. Cleaning up test file...")
test_file.unlink()
print("   Test file deleted")

print("\n" + "="*50)
print("All storage tests passed!")
print("="*50)
