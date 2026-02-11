"""
Confusion matrix builder and heatmap plotter for inventory predictions.

Builds a class-level confusion matrix from inventory dicts:
- Rows = ground truth classes
- Columns = predicted classes
- Cell (i, j) = how many times GT class i was predicted as class j

Note: since we deal with counts (not bounding-box assignments), we
approximate by assuming correct matches up to min(pred, gt) and
distributing over-predictions uniformly as false positives.
For the strict count-based setting the main metrics are in metrics.py;
this matrix is for visual error analysis only.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import CLASSES

Inventory = Dict[str, int]


def build_confusion_matrix(
    predictions: Dict[str, Inventory],
    ground_truths: Dict[str, Inventory],
) -> np.ndarray:
    """
    Build a count-based confusion matrix (N_classes x N_classes).

    Diagonal: correctly predicted counts (TP per class).
    Off-diagonal: always 0 in the pure count setting (no assignment info).
    Last row/col can be used for "other" if needed.

    For a richer matrix, Pipeline B/C could provide box-level assignments;
    for now this captures the per-class TP/FP/FN pattern.

    Returns:
        np.ndarray of shape (num_classes, num_classes).
    """
    n = len(CLASSES)
    matrix = np.zeros((n, n), dtype=int)

    for stem in ground_truths:
        pred = predictions.get(stem, {})
        gt = ground_truths[stem]

        for i, cls in enumerate(CLASSES):
            g = gt.get(cls, 0)
            p = pred.get(cls, 0)
            # True positives on diagonal
            tp = min(p, g)
            matrix[i, i] += tp

    return matrix


def build_count_matrix(
    predictions: Dict[str, Inventory],
    ground_truths: Dict[str, Inventory],
) -> Dict[str, Dict[str, int]]:
    """
    Build per-class TP/FP/FN count summary for visualization.

    Returns:
        Dict mapping class -> {"tp": int, "fp": int, "fn": int}.
    """
    result = {}
    for cls in CLASSES:
        tp = fp = fn = 0
        for stem in ground_truths:
            p = predictions.get(stem, {}).get(cls, 0)
            g = ground_truths[stem].get(cls, 0)
            tp += min(p, g)
            fp += max(0, p - g)
            fn += max(0, g - p)
        result[cls] = {"tp": tp, "fp": fp, "fn": fn}
    return result


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: str | Path,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        matrix: (N, N) confusion matrix.
        output_path: Where to save the PNG.
        title: Plot title.
        figsize: Figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(CLASSES),
        yticklabels=list(CLASSES),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
