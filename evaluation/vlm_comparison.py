"""
VLM Comparison: Run GPT-5.2, Claude Opus 4.6, and Gemini 3.1 Pro on the
same test set to determine the best VLM for Pipeline A.

Usage:
    python -m evaluation.vlm_comparison \
        --images dataset_exp2/images \
        --labels dataset_exp2/labels \
        --output results/vlm_comparison
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Callable, List

from config import CLASSES, Timer
from evaluation.ground_truth import load_ground_truth_dir
from evaluation.metrics import compute_metrics, compute_per_class_metrics

Inventory = Dict[str, int]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── VLM registry ──────────────────────────────────────────────────────

VLM_MODELS = {
    "gpt-5.2": {
        "provider": "OpenAI",
        "model": "GPT-5.2",
        "env_key": "OPENAI_API_KEY",
    },
    "claude-opus-4.6": {
        "provider": "Anthropic",
        "model": "Claude Opus 4.6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini-3.1-pro": {
        "provider": "Google",
        "model": "Gemini 3.1 Pro",
        "env_key": "GOOGLE_API_KEY",
    },
}


def _get_vlm_fn(vlm_name: str) -> Callable[[str], Inventory]:
    """Import and return the identify function for a VLM."""
    if vlm_name == "gpt-5.2":
        from clients.vlm_openai import identify
        return identify
    elif vlm_name == "claude-opus-4.6":
        from clients.vlm_anthropic import identify
        return identify
    elif vlm_name == "gemini-3.1-pro":
        from clients.vlm_google import identify
        return identify
    else:
        raise ValueError(f"Unknown VLM: {vlm_name}")


def _check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    import os
    available = {}
    for name, info in VLM_MODELS.items():
        key = os.getenv(info["env_key"])
        available[name] = bool(key)
        status = "OK" if key else "MISSING"
        print(f"  {info['model']:<25} {info['env_key']:<25} {status}")
    return available


def _find_test_images(images_dir: Path) -> List[Path]:
    """Find all image files, sorted by name."""
    return sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_vlm_comparison(
    images_dir: str,
    labels_dir: str,
    output_dir: str = "results/vlm_comparison",
    vlms: List[str] | None = None,
):
    """
    Run all VLMs on the test set and compare results.

    Args:
        images_dir: Directory with test images.
        labels_dir: Directory with YOLO .txt ground truth labels.
        output_dir: Where to save comparison results.
        vlms: List of VLM names to run. Default: all available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check API keys
    print("\n=== API Key Check ===")
    available = _check_api_keys()

    if vlms is None:
        vlms = [name for name, ok in available.items() if ok]

    if not vlms:
        print("\nNo API keys found. Set at least one of:")
        for info in VLM_MODELS.values():
            print(f"  {info['env_key']}")
        return

    # Load ground truth
    ground_truths = load_ground_truth_dir(labels_dir)
    print(f"\nLoaded ground truth for {len(ground_truths)} images")

    # Find test images
    images = _find_test_images(Path(images_dir))
    images = [img for img in images if img.stem in ground_truths]
    print(f"Matched {len(images)} images with labels")

    if not images:
        print("No images matched. Check paths.")
        return

    # Run each VLM
    all_results = {}

    for vlm_name in vlms:
        if not available.get(vlm_name, False):
            print(f"\nSkipping {vlm_name}: API key not set")
            continue

        info = VLM_MODELS[vlm_name]
        identify_fn = _get_vlm_fn(vlm_name)

        print(f"\n{'='*60}")
        print(f"Running: {info['model']} ({info['provider']})")
        print(f"{'='*60}")

        predictions: Dict[str, Inventory] = {}
        timings: List[float] = []
        errors: List[str] = []

        for i, img_path in enumerate(images):
            stem = img_path.stem
            print(f"  [{i+1}/{len(images)}] {img_path.name}...", end=" ", flush=True)

            try:
                with Timer("vlm") as t:
                    inventory = identify_fn(str(img_path))
                predictions[stem] = inventory
                timings.append(t.duration_ms)
                print(f"{t.duration_ms:.0f}ms — {inventory}")
            except Exception as e:
                predictions[stem] = {}
                timings.append(0)
                errors.append(f"{img_path.name}: {e}")
                print(f"ERROR: {e}")

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truths)
        per_class = compute_per_class_metrics(predictions, ground_truths)

        # Timing stats
        valid_timings = [t for t in timings if t > 0]
        import numpy as np
        timing_stats = {
            "mean_ms": round(float(np.mean(valid_timings)), 1) if valid_timings else 0,
            "median_ms": round(float(np.median(valid_timings)), 1) if valid_timings else 0,
            "std_ms": round(float(np.std(valid_timings)), 1) if valid_timings else 0,
            "total_ms": round(sum(valid_timings), 1),
        }

        result = {
            "model": info["model"],
            "provider": info["provider"],
            "model_id": vlm_name,
            "metrics": metrics,
            "per_class_metrics": per_class,
            "timing": timing_stats,
            "images_processed": len(predictions),
            "errors": len(errors),
            "error_details": errors,
        }

        all_results[vlm_name] = result

        # Save per-model results
        model_file = output_dir / f"{vlm_name.replace('.', '_')}_results.json"
        with open(model_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Save predictions
        pred_file = output_dir / f"{vlm_name.replace('.', '_')}_predictions.json"
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=2)

    if not all_results:
        print("\nNo VLMs ran successfully.")
        return

    # ── Comparison summary ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VLM COMPARISON SUMMARY")
    print(f"{'='*70}")

    header = f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg ms':>10}"
    print(header)
    print("-" * 70)

    best_f1 = -1
    winner = None

    for name, data in all_results.items():
        m = data["metrics"]
        t = data["timing"]
        f1 = m["f1"]

        row = (
            f"{data['model']:<25} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{f1:>10.4f} "
            f"{t['mean_ms']:>10.1f}"
        )
        print(row)

        if f1 > best_f1:
            best_f1 = f1
            winner = name

    print("=" * 70)
    print(f"\nWINNER: {all_results[winner]['model']} (F1 = {best_f1:.4f})")

    # Save comparison
    comparison = {
        "winner": winner,
        "winner_model": all_results[winner]["model"],
        "winner_f1": best_f1,
        "results": {
            name: {
                "model": d["model"],
                "provider": d["provider"],
                "precision": d["metrics"]["precision"],
                "recall": d["metrics"]["recall"],
                "f1": d["metrics"]["f1"],
                "mean_latency_ms": d["timing"]["mean_ms"],
                "errors": d["errors"],
            }
            for name, d in all_results.items()
        },
    }

    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {summary_file}")

    # Save ground truth
    gt_file = output_dir / "ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(ground_truths, f, indent=2)

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare VLMs on the test set to pick the best for Pipeline A"
    )
    parser.add_argument("--images", default="dataset_exp2/images",
                        help="Test images directory")
    parser.add_argument("--labels", default="dataset_exp2/labels",
                        help="Ground truth labels directory")
    parser.add_argument("--output", default="results/vlm_comparison",
                        help="Output directory")
    parser.add_argument("--vlms", nargs="+", default=None,
                        choices=list(VLM_MODELS.keys()),
                        help="Which VLMs to run (default: all with API keys)")
    args = parser.parse_args()

    run_vlm_comparison(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        vlms=args.vlms,
    )
