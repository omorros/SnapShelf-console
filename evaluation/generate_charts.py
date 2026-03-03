"""
Cross-condition visualizations for Experiment 2 dissertation.

Reads comparison_summary.json from each results subfolder and generates
publication-quality charts that span all 4 image conditions.

Usage:
    python -m evaluation.generate_charts
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "font.family": "serif",
})

PIPELINE_LABELS = {
    "yolo-14": "YOLO-14",
    "vlm":     "VLM (GPT-5.2)",
    "yolo-cnn": "YOLO + CNN",
}
PIPELINE_ORDER = ["yolo-14", "vlm", "yolo-cnn"]
CONDITION_ORDER = ["clean", "d1_blur", "d2_noise", "d3_jpeg"]
CONDITION_LABELS = {
    "clean":    "Clean",
    "d1_blur":  "Gaussian Blur",
    "d2_noise": "Gaussian Noise",
    "d3_jpeg":  "JPEG Compression",
}
PIPELINE_COLORS = {
    "yolo-14":  "#2196F3",
    "vlm":      "#4CAF50",
    "yolo-cnn": "#FF9800",
}
PIPELINE_MARKERS = {
    "yolo-14":  "o",
    "vlm":      "s",
    "yolo-cnn": "^",
}

RESULTS_ROOT = Path("results")
OUTPUT_DIR = RESULTS_ROOT / "charts"


# ── Data loading ─────────────────────────────────────────────────────────

def load_all_summaries() -> Dict[str, Dict]:
    """Load comparison_summary.json from each condition folder."""
    data = {}
    for cond in CONDITION_ORDER:
        path = RESULTS_ROOT / cond / "comparison_summary.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        with open(path) as f:
            data[cond] = json.load(f)
    return data


# ── Chart 1: F1 degradation lines ───────────────────────────────────────

def plot_f1_degradation(data: Dict[str, Dict]):
    """Line chart: F1 across conditions, one line per pipeline."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    conditions = [c for c in CONDITION_ORDER if c in data]
    x = np.arange(len(conditions))

    for pipe in PIPELINE_ORDER:
        f1_vals = []
        for cond in conditions:
            f1_vals.append(data[cond][pipe]["metrics"]["f1"])
        ax.plot(
            x, f1_vals,
            marker=PIPELINE_MARKERS[pipe],
            color=PIPELINE_COLORS[pipe],
            label=PIPELINE_LABELS[pipe],
            linewidth=2.2,
            markersize=9,
        )
        # annotate values
        for i, v in enumerate(f1_vals):
            ax.annotate(
                f"{v:.3f}",
                (x[i], v),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions])
    ax.set_ylabel("Micro-Averaged F1 Score")
    ax.set_title("Pipeline Robustness: F1 Across Image Conditions")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    out = OUTPUT_DIR / "f1_degradation_lines.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Chart 2: Per-class F1 heatmap (clean) ───────────────────────────────

def plot_perclass_heatmap(data: Dict[str, Dict]):
    """Heatmap: per-class F1 for each pipeline (clean condition)."""
    clean = data.get("clean")
    if not clean:
        return

    classes = sorted(list(clean[PIPELINE_ORDER[0]]["per_class"].keys()))
    # Nice class labels
    class_labels = [c.replace("_", " ").title() for c in classes]

    matrix = []
    pipe_labels = []
    for pipe in PIPELINE_ORDER:
        row = [clean[pipe]["per_class"][c]["f1"] for c in classes]
        matrix.append(row)
        pipe_labels.append(PIPELINE_LABELS[pipe])

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        xticklabels=class_labels,
        yticklabels=pipe_labels,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_title("Per-Class F1 Score — Clean Condition")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    out = OUTPUT_DIR / "perclass_f1_heatmap_clean.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Chart 3: Latency vs F1 scatter ──────────────────────────────────────

def plot_latency_vs_f1(data: Dict[str, Dict]):
    """Scatter plot: latency (log-x) vs F1, one point per pipeline×condition."""
    fig, ax = plt.subplots(figsize=(9, 6))

    cond_markers = {
        "clean":    "o",
        "d1_blur":  "D",
        "d2_noise": "X",
        "d3_jpeg":  "P",
    }

    conditions = [c for c in CONDITION_ORDER if c in data]

    for pipe in PIPELINE_ORDER:
        for cond in conditions:
            entry = data[cond][pipe]
            latency = entry["timing"]["avg_per_image_ms"]
            f1 = entry["metrics"]["f1"]
            ax.scatter(
                latency, f1,
                color=PIPELINE_COLORS[pipe],
                marker=cond_markers[cond],
                s=120,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )

    # Legend: pipelines (color)
    from matplotlib.lines import Line2D
    pipe_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PIPELINE_COLORS[p],
               markersize=10, label=PIPELINE_LABELS[p])
        for p in PIPELINE_ORDER
    ]
    cond_handles = [
        Line2D([0], [0], marker=cond_markers[c], color="gray", markersize=9,
               linestyle="None", label=CONDITION_LABELS[c])
        for c in conditions
    ]

    leg1 = ax.legend(handles=pipe_handles, title="Pipeline", loc="lower right",
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=cond_handles, title="Condition", loc="lower left",
              framealpha=0.9)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Average Latency per Image (ms, log scale)")
    ax.set_ylabel("Micro-Averaged F1 Score")
    ax.set_title("Speed–Accuracy Tradeoff Across Conditions")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / "latency_vs_f1_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Chart 4: Per-class F1 grouped bars (clean) ──────────────────────────

def plot_perclass_bars(data: Dict[str, Dict]):
    """Grouped bar chart: per-class F1 for all 3 pipelines (clean)."""
    clean = data.get("clean")
    if not clean:
        return

    classes = sorted(list(clean[PIPELINE_ORDER[0]]["per_class"].keys()))
    class_labels = [c.replace("_", " ").title() for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, pipe in enumerate(PIPELINE_ORDER):
        vals = [clean[pipe]["per_class"][c]["f1"] for c in classes]
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=PIPELINE_LABELS[pipe],
            color=PIPELINE_COLORS[pipe],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on very low bars for emphasis
        for j, v in enumerate(vals):
            if v < 0.15:
                ax.text(
                    x[j] + offsets[i], v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="red",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison — Clean Condition")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    out = OUTPUT_DIR / "perclass_f1_bars_clean.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Chart 5: F1 delta (degradation impact) ──────────────────────────────

def plot_f1_delta(data: Dict[str, Dict]):
    """Bar chart: F1 drop from clean for each degradation × pipeline."""
    clean = data.get("clean")
    if not clean:
        return

    degradations = [c for c in CONDITION_ORDER if c != "clean" and c in data]
    deg_labels = [CONDITION_LABELS[c] for c in degradations]

    x = np.arange(len(degradations))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, pipe in enumerate(PIPELINE_ORDER):
        clean_f1 = clean[pipe]["metrics"]["f1"]
        deltas = []
        for cond in degradations:
            deg_f1 = data[cond][pipe]["metrics"]["f1"]
            deltas.append(clean_f1 - deg_f1)

        bars = ax.bar(
            x + offsets[i], deltas, width,
            label=PIPELINE_LABELS[pipe],
            color=PIPELINE_COLORS[pipe],
            edgecolor="white",
            linewidth=0.5,
        )
        # annotate
        for j, v in enumerate(deltas):
            ax.text(
                x[j] + offsets[i], v + 0.005,
                f"−{v:.3f}" if v > 0 else f"+{abs(v):.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(deg_labels)
    ax.set_ylabel("F1 Drop from Clean Baseline")
    ax.set_title("Degradation Impact: F1 Loss by Condition")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")

    out = OUTPUT_DIR / "f1_delta_degradation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Chart 6: Per-class heatmap across ALL conditions ────────────────────

def plot_perclass_heatmap_all_conditions(data: Dict[str, Dict]):
    """
    Heatmap grid: one row per pipeline × condition combo,
    columns are the 14 classes. Shows how per-class performance shifts.
    """
    conditions = [c for c in CONDITION_ORDER if c in data]
    if not conditions:
        return

    classes = sorted(list(data[conditions[0]][PIPELINE_ORDER[0]]["per_class"].keys()))
    class_labels = [c.replace("_", " ").title() for c in classes]

    rows = []
    row_labels = []
    for pipe in PIPELINE_ORDER:
        for cond in conditions:
            row = [data[cond][pipe]["per_class"][c]["f1"] for c in classes]
            rows.append(row)
            row_labels.append(f"{PIPELINE_LABELS[pipe]} — {CONDITION_LABELS[cond]}")

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        xticklabels=class_labels,
        yticklabels=row_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 7.5},
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_title("Per-Class F1 Score — All Pipelines × All Conditions")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Add horizontal separators between pipeline groups
    for i in range(1, len(PIPELINE_ORDER)):
        ax.axhline(y=i * len(conditions), color="black", linewidth=2)

    out = OUTPUT_DIR / "perclass_f1_heatmap_all.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results from all conditions...")
    data = load_all_summaries()

    if not data:
        print("ERROR: No results found. Run evaluations first.")
        return

    print(f"Found {len(data)} conditions: {list(data.keys())}\n")

    print("Generating charts...")
    plot_f1_degradation(data)
    plot_perclass_heatmap(data)
    plot_latency_vs_f1(data)
    plot_perclass_bars(data)
    plot_f1_delta(data)
    plot_perclass_heatmap_all_conditions(data)

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
