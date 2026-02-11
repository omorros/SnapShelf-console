"""
Error analysis: missed items, false positives, over/under-counting breakdown.

Provides a per-image and aggregate view of where each pipeline fails.
"""

from typing import Dict, List

from config import CLASSES

Inventory = Dict[str, int]


def analyze_errors(
    predictions: Dict[str, Inventory],
    ground_truths: Dict[str, Inventory],
) -> Dict[str, any]:
    """
    Perform error analysis across all images.

    Returns:
        Dict with keys:
            - missed: classes with FN > 0, sorted by total FN
            - false_positives: classes with FP > 0, sorted by total FP
            - over_counted: images where pred > gt for some class
            - under_counted: images where pred < gt for some class
            - per_image: list of per-image error summaries
    """
    class_fp: Dict[str, int] = {c: 0 for c in CLASSES}
    class_fn: Dict[str, int] = {c: 0 for c in CLASSES}

    per_image: List[Dict] = []
    over_counted: List[Dict] = []
    under_counted: List[Dict] = []

    for stem in sorted(ground_truths.keys()):
        pred = predictions.get(stem, {})
        gt = ground_truths[stem]

        img_errors = {"image": stem, "errors": []}

        for cls in set(list(pred.keys()) + list(gt.keys())):
            p = pred.get(cls, 0)
            g = gt.get(cls, 0)

            if p > g:
                diff = p - g
                if cls in class_fp:
                    class_fp[cls] += diff
                img_errors["errors"].append({
                    "class": cls, "type": "over_count",
                    "predicted": p, "actual": g, "excess": diff,
                })
                over_counted.append({
                    "image": stem, "class": cls,
                    "predicted": p, "actual": g,
                })

            elif p < g:
                diff = g - p
                if cls in class_fn:
                    class_fn[cls] += diff
                img_errors["errors"].append({
                    "class": cls, "type": "under_count",
                    "predicted": p, "actual": g, "deficit": diff,
                })
                under_counted.append({
                    "image": stem, "class": cls,
                    "predicted": p, "actual": g,
                })

        if img_errors["errors"]:
            per_image.append(img_errors)

    # Sort classes by error magnitude
    missed = sorted(
        [{"class": c, "total_fn": v} for c, v in class_fn.items() if v > 0],
        key=lambda x: x["total_fn"], reverse=True,
    )
    false_positives = sorted(
        [{"class": c, "total_fp": v} for c, v in class_fp.items() if v > 0],
        key=lambda x: x["total_fp"], reverse=True,
    )

    return {
        "missed": missed,
        "false_positives": false_positives,
        "over_counted": over_counted,
        "under_counted": under_counted,
        "per_image": per_image,
        "total_images_with_errors": len(per_image),
        "total_images": len(ground_truths),
    }
