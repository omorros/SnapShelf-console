"""
Count-based Precision / Recall / F1 metrics for inventory comparison.

For each image, per class:
    TP = min(pred, gt)
    FP = max(0, pred - gt)
    FN = max(0, gt - pred)

Micro-averaged across all images for overall P/R/F1.
"""

from typing import Dict, List, Tuple

from config import CLASSES

Inventory = Dict[str, int]


def _per_image_counts(pred: Inventory, gt: Inventory) -> Tuple[int, int, int]:
    """
    Compute TP, FP, FN for a single image across all classes.

    Returns:
        (tp, fp, fn) tuple.
    """
    all_classes = set(pred.keys()) | set(gt.keys())
    tp = fp = fn = 0
    for cls in all_classes:
        p = pred.get(cls, 0)
        g = gt.get(cls, 0)
        tp += min(p, g)
        fp += max(0, p - g)
        fn += max(0, g - p)
    return tp, fp, fn


def compute_metrics(
    predictions: Dict[str, Inventory],
    ground_truths: Dict[str, Inventory],
) -> Dict[str, float]:
    """
    Compute micro-averaged P/R/F1 across all images.

    Args:
        predictions: Dict mapping image stem -> predicted inventory.
        ground_truths: Dict mapping image stem -> ground truth inventory.

    Returns:
        Dict with keys: precision, recall, f1, total_tp, total_fp, total_fn.
    """
    total_tp = total_fp = total_fn = 0

    for stem in ground_truths:
        pred = predictions.get(stem, {})
        gt = ground_truths[stem]
        tp, fp, fn = _per_image_counts(pred, gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def compute_per_class_metrics(
    predictions: Dict[str, Inventory],
    ground_truths: Dict[str, Inventory],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class P/R/F1 (micro-averaged across images for each class).

    Args:
        predictions: Dict mapping image stem -> predicted inventory.
        ground_truths: Dict mapping image stem -> ground truth inventory.

    Returns:
        Dict mapping class name -> {precision, recall, f1, tp, fp, fn}.
    """
    class_tp: Dict[str, int] = {c: 0 for c in CLASSES}
    class_fp: Dict[str, int] = {c: 0 for c in CLASSES}
    class_fn: Dict[str, int] = {c: 0 for c in CLASSES}

    for stem in ground_truths:
        pred = predictions.get(stem, {})
        gt = ground_truths[stem]
        for cls in CLASSES:
            p = pred.get(cls, 0)
            g = gt.get(cls, 0)
            class_tp[cls] += min(p, g)
            class_fp[cls] += max(0, p - g)
            class_fn[cls] += max(0, g - p)

    results: Dict[str, Dict[str, float]] = {}
    for cls in CLASSES:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    return results
