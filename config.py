"""
Configuration and reproducibility module for Food Detection Pipeline Comparison.

This module provides:
- Environment validation (API keys, dependencies)
- Structured logging setup
- Random seed control for reproducibility
- Model versioning and hash verification
- Timing utilities that exclude initialization

DISSERTATION ARTIFACT: All settings frozen for fair comparison.
"""

import os
import sys
import hashlib
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import time

import numpy as np
from dotenv import load_dotenv

# Load environment variables at module import
load_dotenv()


# =============================================================================
# FROZEN EXPERIMENTAL SETTINGS
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Frozen configuration for reproducible experiments.
    All values locked - do not modify during experiment runs.
    """
    # LLM Settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_image_detail: str = "high"
    llm_max_tokens_single: int = 150
    llm_max_tokens_multi: int = 500

    # YOLO Settings (identical for B and C for fair comparison)
    yolo_conf_threshold: float = 0.15
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 8
    yolo_crop_padding: float = 0.10

    # Geometric Filters (identical for B and C)
    min_bbox_area_pct: float = 0.02
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0

    # Model files
    yolo_standard_model: str = "yolov8s.pt"
    yolo_world_model: str = "yolov8s-worldv2.pt"

    # YOLO-World prompts (FIXED - never changes per image)
    yolo_world_prompts: tuple = ("food", "fruit", "vegetable", "packaged food")

    # Random seed for reproducibility
    random_seed: int = 42


# Global config instance
CONFIG = ExperimentConfig()


# =============================================================================
# MODEL VERSION TRACKING
# =============================================================================

# Expected SHA256 hashes for model files (first 16 chars for brevity)
# Update these after downloading models to ensure consistency
EXPECTED_MODEL_HASHES = {
    "yolov8s.pt": None,  # Will be set on first verified run
    "yolov8s-worldv2.pt": None,  # Will be set on first verified run
}


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for version verification."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()[:16]  # First 16 chars


def verify_model_hash(model_path: str) -> Dict[str, Any]:
    """
    Verify model file hash and return version info.

    Returns:
        Dict with 'path', 'hash', 'verified', 'expected'
    """
    path = Path(model_path)
    if not path.exists():
        return {
            "path": str(path),
            "hash": None,
            "verified": False,
            "expected": EXPECTED_MODEL_HASHES.get(path.name),
            "error": "File not found"
        }

    actual_hash = compute_file_hash(str(path))
    expected = EXPECTED_MODEL_HASHES.get(path.name)

    return {
        "path": str(path),
        "hash": actual_hash,
        "verified": expected is None or actual_hash == expected,
        "expected": expected,
        "size_bytes": path.stat().st_size
    }


# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def validate_environment() -> Dict[str, Any]:
    """
    Validate all required environment variables and dependencies.

    Returns:
        Dict with validation results and any errors
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": {}
    }

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        results["valid"] = False
        results["errors"].append(
            "OPENAI_API_KEY not found. Set it in .env file or environment."
        )
    elif not api_key.startswith(("sk-", "sk-proj-")):
        results["warnings"].append(
            "OPENAI_API_KEY doesn't start with expected prefix (sk- or sk-proj-)"
        )

    # Record config for logging
    results["config"] = {
        "llm_model": CONFIG.llm_model,
        "llm_temperature": CONFIG.llm_temperature,
        "yolo_conf_threshold": CONFIG.yolo_conf_threshold,
        "yolo_max_detections": CONFIG.yolo_max_detections,
        "random_seed": CONFIG.random_seed,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }

    return results


# =============================================================================
# RANDOM SEED CONTROL
# =============================================================================

def set_reproducibility_seed(seed: Optional[int] = None) -> int:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed (defaults to CONFIG.random_seed)

    Returns:
        The seed that was set
    """
    seed = seed or CONFIG.random_seed

    random.seed(seed)
    np.random.seed(seed)

    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Note: Full determinism requires additional settings
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

@dataclass
class PipelineLog:
    """Structured log entry for pipeline execution."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline: str = ""
    image: str = ""
    step: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ExperimentLogger:
    """
    Structured logger for experiment runs.
    Captures all data needed for post-hoc analysis and reproducibility.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs: list[PipelineLog] = []

        # Setup file logging
        self.log_file = self.log_dir / f"experiment_{self.session_id}.jsonl"

        # Also setup standard logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / f"experiment_{self.session_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log(self, entry: PipelineLog):
        """Add a log entry."""
        self.logs.append(entry)

        # Write to JSONL file immediately (append mode)
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp,
                "pipeline": entry.pipeline,
                "image": entry.image,
                "step": entry.step,
                "duration_ms": entry.duration_ms,
                "details": entry.details,
                "error": entry.error
            }) + "\n")

        # Also log to standard logger
        if entry.error:
            self.logger.error(f"[{entry.pipeline}] {entry.step}: {entry.error}")
        else:
            self.logger.info(
                f"[{entry.pipeline}] {entry.step} ({entry.duration_ms:.2f}ms)"
            )

    def log_detection(
        self,
        pipeline: str,
        image: str,
        bbox: Dict,
        confidence: float,
        passed_filter: bool,
        prompt_match: str = "object"
    ):
        """Log a single detection event."""
        self.log(PipelineLog(
            pipeline=pipeline,
            image=image,
            step="detection",
            details={
                "bbox": bbox,
                "confidence": confidence,
                "passed_filter": passed_filter,
                "prompt_match": prompt_match
            }
        ))

    def log_llm_call(
        self,
        pipeline: str,
        image: str,
        crop_index: Optional[int],
        raw_response: str,
        parsed_result: Any,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Log an LLM API call."""
        self.log(PipelineLog(
            pipeline=pipeline,
            image=image,
            step="llm_call",
            duration_ms=duration_ms,
            details={
                "crop_index": crop_index,
                "raw_response": raw_response[:500] if raw_response else None,  # Truncate for storage
                "parsed_result": parsed_result
            },
            error=error
        ))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        return {
            "session_id": self.session_id,
            "total_logs": len(self.logs),
            "errors": sum(1 for log in self.logs if log.error),
            "log_file": str(self.log_file)
        }


# Global logger instance (lazy initialization)
_logger: Optional[ExperimentLogger] = None


def get_logger() -> ExperimentLogger:
    """Get or create the global experiment logger."""
    global _logger
    if _logger is None:
        _logger = ExperimentLogger()
    return _logger


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    duration_ms: float
    start_time: float
    end_time: float


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def result(self) -> TimingResult:
        return TimingResult(
            name=self.name,
            duration_ms=self.duration_ms,
            start_time=self.start_time,
            end_time=self.end_time
        )


@contextmanager
def timed_operation(name: str):
    """Context manager that yields timing information."""
    timer = Timer(name)
    with timer:
        yield timer


# =============================================================================
# WARM-UP UTILITIES
# =============================================================================

def warmup_models(yolo_standard: Any = None, yolo_world: Any = None, iterations: int = 1):
    """
    Run warm-up iterations to stabilize GPU/CPU caches.

    Args:
        yolo_standard: YOLODetectorAgnostic instance
        yolo_world: YOLODetector instance
        iterations: Number of warm-up iterations
    """
    from PIL import Image
    import io

    # Create a dummy image for warm-up
    dummy_img = Image.new("RGB", (640, 480), color="gray")
    buffer = io.BytesIO()
    dummy_img.save(buffer, format="PNG")
    dummy_path = Path("_warmup_temp.png")
    dummy_img.save(dummy_path)

    try:
        for i in range(iterations):
            if yolo_standard is not None:
                _ = yolo_standard.detect(str(dummy_path))
            if yolo_world is not None:
                _ = yolo_world.detect(str(dummy_path))
    finally:
        # Clean up temp file
        if dummy_path.exists():
            dummy_path.unlink()


# =============================================================================
# INITIALIZATION CHECK
# =============================================================================

def init_experiment() -> Dict[str, Any]:
    """
    Initialize experiment environment.
    Call this before running any pipelines.

    Returns:
        Dict with initialization status and config
    """
    # Validate environment
    env_result = validate_environment()

    if not env_result["valid"]:
        raise RuntimeError(
            "Environment validation failed:\n" +
            "\n".join(f"  - {e}" for e in env_result["errors"])
        )

    # Set reproducibility seed
    seed = set_reproducibility_seed()

    # Initialize logger
    logger = get_logger()
    logger.logger.info(f"Experiment initialized with seed {seed}")
    logger.logger.info(f"Config: {env_result['config']}")

    return {
        "status": "initialized",
        "seed": seed,
        "config": env_result["config"],
        "warnings": env_result["warnings"],
        "logger": logger.get_summary()
    }
