"""
Report generation: comparison tables, bar charts, and LaTeX output.

Generates visual and textual summaries from evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import CONFIG, CLASSES
from evaluation.metrics import compute_metrics, compute_per_class_metrics
from evaluation.confusion import build_confusion_matrix, build_count_matrix, plot_confusion_matrix
from evaluation.error_analysis import analyze_errors

Inventory = Dict[str, int]


def generate_report(
    all_results: Dict[str, Dict],
    output_dir: str | Path | None = None,
):
    """
    Generate a full evaluation report for all pipelines.

    Args:
        all_results: Output from evaluate_runner.run_evaluation().
        output_dir: Where to save report artifacts.
    """
    output_dir = Path(output_dir or CONFIG.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for pipe_name, data in all_results.items():
        preds = data["predictions"]
        gts = data["ground_truths"]

        # Overall metrics
        metrics = compute_metrics(preds, gts)
        per_class = compute_per_class_metrics(preds, gts)
        errors = analyze_errors(preds, gts)

        # Confusion matrix
        cm = build_confusion_matrix(preds, gts)
        plot_confusion_matrix(
            cm,
            output_dir / f"{pipe_name}_confusion.png",
            title=f"Confusion Matrix — {pipe_name}",
        )

        # Timing
        total_time = data.get("total_time_ms", 0)
        n_images = len(gts)
        avg_time = total_time / n_images if n_images > 0 else 0

        summary[pipe_name] = {
            "metrics": metrics,
            "per_class": per_class,
            "timing": {
                "total_ms": round(total_time, 2),
                "avg_per_image_ms": round(avg_time, 2),
                "n_images": n_images,
            },
            "errors_summary": {
                "total_images_with_errors": errors["total_images_with_errors"],
                "top_missed": errors["missed"][:5],
                "top_false_positives": errors["false_positives"][:5],
            },
        }

        # Save per-pipeline report
        pipe_report_path = output_dir / f"{pipe_name}_report.json"
        with open(pipe_report_path, "w") as f:
            json.dump({
                "pipeline": pipe_name,
                "metrics": metrics,
                "per_class_metrics": per_class,
                "error_analysis": errors,
                "timing": summary[pipe_name]["timing"],
            }, f, indent=2, default=str)

    # Save comparison summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison summary: {summary_path}")

    # Print comparison table
    _print_comparison_table(summary)

    # Generate bar chart
    _plot_comparison_bars(summary, output_dir / "comparison_bars.png")

    # Generate LaTeX table
    _generate_latex_table(summary, output_dir / "comparison_table.tex")

    return summary


def _print_comparison_table(summary: Dict):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 70)
    print("PIPELINE COMPARISON SUMMARY")
    print("=" * 70)

    header = f"{'Pipeline':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg ms':>10}"
    print(header)
    print("-" * 70)

    for pipe_name, data in summary.items():
        m = data["metrics"]
        t = data["timing"]
        row = (
            f"{pipe_name:<12} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{t['avg_per_image_ms']:>10.1f}"
        )
        print(row)

    print("=" * 70)


def _plot_comparison_bars(summary: Dict, output_path: Path):
    """Generate comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pipelines = list(summary.keys())
    if not pipelines:
        return

    metrics_names = ["precision", "recall", "f1"]
    x = np.arange(len(pipelines))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics_names):
        values = [summary[p]["metrics"][metric] for p in pipelines]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_ylabel("Score")
    ax.set_title("Pipeline Comparison — P / R / F1")
    ax.set_xticks(x + width)
    ax.set_xticklabels(pipelines)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _generate_latex_table(summary: Dict, output_path: Path):
    """Generate a LaTeX table for the dissertation."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Pipeline Comparison — Micro-Averaged Metrics}",
        r"\label{tab:pipeline-comparison}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Pipeline & Precision & Recall & F1 & Avg. Latency (ms) \\",
        r"\midrule",
    ]

    for pipe_name, data in summary.items():
        m = data["metrics"]
        t = data["timing"]
        lines.append(
            f"{pipe_name} & {m['precision']:.4f} & {m['recall']:.4f} & "
            f"{m['f1']:.4f} & {t['avg_per_image_ms']:.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
