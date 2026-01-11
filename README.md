# Food Detection Pipeline Comparison

Dissertation artefact comparing two approaches for multi-item food identification:

| System | Approach | Description |
|--------|----------|-------------|
| **A** | LLM-only | Full image → GPT-4o Vision → All items |
| **B** | YOLO + LLM | YOLO regions → GPT-4o per crop → Aggregated |

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Interactive mode (menu-driven)
python main.py

# CLI mode (JSON output)
python main.py llm <image_path>
python main.py yolo-llm <image_path>
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download YOLO model
# Place yolov8s.pt in project root
# Download from: https://github.com/ultralytics/assets/releases

# 4. Configure API key
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Output Schema (Frozen)

Both systems produce identical output format:

```json
{
  "items": [
    {"name": "apple", "state": "fresh"},
    {"name": "milk", "state": "packaged"}
  ],
  "meta": {
    "pipeline": "llm",
    "image": "groceries.jpg",
    "runtime_ms": 1234.56,
    "fallback_used": false
  }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `name` | Generic food name (lowercase, normalized) |
| `state` | `fresh` \| `packaged` \| `cooked` \| `unknown` |
| `pipeline` | `llm` or `yolo-llm` |
| `runtime_ms` | Execution time in milliseconds |
| `fallback_used` | True if YOLO found no detections and used full image |

## Project Structure

```
├── main.py                  # CLI + Interactive entrypoint
├── requirements.txt         # Dependencies
├── .env.example             # API key template
├── .gitignore
├── yolov8s.pt               # YOLO model (not tracked)
├── clients/
│   ├── llm_client.py        # OpenAI Vision (frozen prompts)
│   └── yolo_detector.py     # YOLOv8 detection
└── pipelines/
    ├── output.py            # Frozen output schema
    ├── llm_pipeline.py      # System A
    └── yolo_llm_pipeline.py # System B
```

## Configuration

### LLM Settings (clients/llm_client.py)

| Setting | Value | Notes |
|---------|-------|-------|
| Model | `gpt-4o-mini` | Cost-efficient |
| Temperature | `0` | Deterministic |
| Image Detail | `high` | Same for both systems |

### YOLO Settings (clients/yolo_detector.py)

| Setting | Default | Notes |
|---------|---------|-------|
| `CONF_THRESHOLD` | `0.25` | Lower = more detections |
| `IOU_THRESHOLD` | `0.45` | NMS overlap threshold |
| `MAX_DETECTIONS` | `15` | Limit per image |
| `CROP_PADDING_PCT` | `0.10` | 10% padding around crops |

### Fallback Behavior (pipelines/yolo_llm_pipeline.py)

| Setting | Default | Notes |
|---------|---------|-------|
| `USE_FALLBACK` | `True` | Use full image if YOLO finds nothing |

Set to `False` for strict comparison (System B returns empty if no YOLO detections).

## Requirements

- Python 3.10+
- OpenAI API key
- ~25MB disk (YOLO model)
