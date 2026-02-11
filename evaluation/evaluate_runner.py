"""
Evaluation orchestrator: runs all pipelines on the test set and collects results.

Expects:
    - Test images in a directory (jpg/png)
    - Corresponding YOLO .txt label files in a labels directory
    - Trained model weights in weights/

Outputs:
    - Per-pipeline JSON results to results/
    - Metrics summary
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

from config import CONFIG, CLASSES, Timer
from pipelines.output import Inventory, PipelineResult
from evaluation.ground_truth import load_ground_truth_dir

# Image extensions to look for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_test_images(images_dir: str | Path) -> List[Path]:
    """Find all image files in a directory, sorted by name."""
    images_dir = Path(images_dir)
    images = []
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def _run_single_pipeline(
    pipeline_fn: Callable[[str], PipelineResult],
    image_path: Path,
) -> PipelineResult:
    """Run a single pipeline on one image, catching errors."""
    try:
        return pipeline_fn(str(image_path))
    except Exception as e:
        return {
            "inventory": {},
            "meta": {
                "pipeline": "error",
                "image": image_path.name,
                "runtime_ms": 0,
                "error": f"{type(e).__name__}: {e}",
            }
        }


def run_evaluation(
    images_dir: str,
    labels_dir: str,
    pipelines: Optional[List[str]] = None,
    output_dir: str | None = None,
) -> Dict[str, Dict]:
    """
    Run evaluation across selected pipelines on the test set.

    Args:
        images_dir: Directory with test images.
        labels_dir: Directory with YOLO .txt ground truth labels.
        pipelines: List of pipeline names to evaluate. Default: all available.
                   Options: "vlm", "yolo-14", "yolo-cnn"
        output_dir: Where to save JSON results. Default: CONFIG.results_dir.

    Returns:
        Dict mapping pipeline name -> {
            "predictions": {stem: inventory},
            "ground_truths": {stem: inventory},
            "results": [PipelineResult, ...],
            "total_time_ms": float,
        }
    """
    output_dir = Path(output_dir or CONFIG.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    ground_truths = load_ground_truth_dir(labels_dir)
    print(f"Loaded ground truth for {len(ground_truths)} images")

    # Find test images
    images = _find_test_images(images_dir)
    print(f"Found {len(images)} test images")

    # Filter to images that have ground truth
    images = [img for img in images if img.stem in ground_truths]
    print(f"Matched {len(images)} images with labels")

    if not images:
        print("No images matched with labels. Check paths.")
        return {}

    # Build pipeline registry
    available_pipelines = _get_pipeline_registry()
    if pipelines is None:
        pipelines = list(available_pipelines.keys())

    all_results = {}

    for pipe_name in pipelines:
        if pipe_name not in available_pipelines:
            print(f"Unknown pipeline: {pipe_name}, skipping")
            continue

        pipe_fn = available_pipelines[pipe_name]
        print(f"\n{'='*60}")
        print(f"Running pipeline: {pipe_name}")
        print(f"{'='*60}")

        predictions: Dict[str, Inventory] = {}
        results_list: List[PipelineResult] = []
        total_time = 0.0

        for i, img_path in enumerate(images):
            stem = img_path.stem
            print(f"  [{i+1}/{len(images)}] {img_path.name}...", end=" ", flush=True)

            with Timer("image") as t:
                result = _run_single_pipeline(pipe_fn, img_path)

            predictions[stem] = result["inventory"]
            results_list.append(result)
            total_time += t.duration_ms
            print(f"{t.duration_ms:.0f}ms")

        # Save predictions to JSON
        json_path = output_dir / f"{pipe_name}_predictions.json"
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions: {json_path}")

        # Save full results
        results_path = output_dir / f"{pipe_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results_list, f, indent=2, default=str)
        print(f"Saved full results: {results_path}")

        all_results[pipe_name] = {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "results": results_list,
            "total_time_ms": total_time,
        }

    # Save ground truth for reference
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truths, f, indent=2)

    return all_results


def _get_pipeline_registry() -> Dict[str, Callable]:
    """Return available pipeline run functions."""
    registry = {}

    # Pipeline B: always available if weights exist
    try:
        from pipelines.yolo_pipeline import run as yolo_run
        if Path(CONFIG.yolo_14class_weights).exists():
            registry["yolo-14"] = yolo_run
        else:
            print(f"Skipping yolo-14: weights not found at {CONFIG.yolo_14class_weights}")
    except ImportError as e:
        print(f"Skipping yolo-14: {e}")

    # Pipeline A: needs OpenAI key
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            from pipelines.vlm_pipeline import run as vlm_run
            registry["vlm"] = vlm_run
        else:
            print("Skipping vlm: OPENAI_API_KEY not set")
    except ImportError as e:
        print(f"Skipping vlm: {e}")

    # Pipeline C: needs both objectness weights and CNN weights
    try:
        from pipelines.yolo_cnn_pipeline import run as yolo_cnn_run
        obj_ok = Path(CONFIG.yolo_objectness_weights).exists()
        cnn_ok = Path(CONFIG.cnn_weights).exists()
        if obj_ok and cnn_ok:
            registry["yolo-cnn"] = yolo_cnn_run
        else:
            missing = []
            if not obj_ok:
                missing.append(f"objectness weights ({CONFIG.yolo_objectness_weights})")
            if not cnn_ok:
                missing.append(f"CNN weights ({CONFIG.cnn_weights})")
            print(f"Skipping yolo-cnn: missing {', '.join(missing)}")
    except ImportError as e:
        print(f"Skipping yolo-cnn: {e}")

    return registry
