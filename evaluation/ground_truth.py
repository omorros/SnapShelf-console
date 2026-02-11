"""
Ground truth parser: YOLO .txt annotation files -> Inventory dicts.

Each .txt label file has lines: <class_id> <x_center> <y_center> <width> <height>
We only need class_id to build the inventory (class -> count).
"""

from pathlib import Path
from typing import Dict

from config import ID_TO_CLASS

Inventory = Dict[str, int]


def load_ground_truth(label_path: str | Path) -> Inventory:
    """
    Parse a single YOLO-format .txt annotation file into an Inventory.

    Args:
        label_path: Path to a YOLO annotation .txt file.

    Returns:
        Inventory dict, e.g. {"apple": 3, "banana": 1}.
    """
    label_path = Path(label_path)
    inventory: Inventory = {}

    if not label_path.exists():
        return inventory

    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        class_name = ID_TO_CLASS.get(class_id)
        if class_name is None:
            continue
        inventory[class_name] = inventory.get(class_name, 0) + 1

    return inventory


def load_ground_truth_dir(labels_dir: str | Path) -> Dict[str, Inventory]:
    """
    Load ground truth for all .txt files in a directory.

    Args:
        labels_dir: Directory containing YOLO .txt annotation files.

    Returns:
        Dict mapping image stem (filename without extension) -> Inventory.
    """
    labels_dir = Path(labels_dir)
    results: Dict[str, Inventory] = {}

    for txt_file in sorted(labels_dir.glob("*.txt")):
        stem = txt_file.stem
        results[stem] = load_ground_truth(txt_file)

    return results
