# Food Detection Pipeline Comparison

Interactive console application for comparing two food detection approaches:

| System | Approach | Description |
|--------|----------|-------------|
| **A** | LLM-only | Full image → GPT-4o Vision → All items detected |
| **B** | YOLO + LLM | YOLO regions → GPT-4o per crop → Aggregated results |

## Quick Start

```bash
# 1. Activate virtual environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 2. Run the application
python main.py
```

## Setup (First Time)

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## How It Works

1. Run `python main.py`
2. Select pipeline (1 for LLM-only, 2 for YOLO+LLM)
3. File picker opens — select an image
4. Results displayed in table format + raw JSON

## Output Schema

Both systems produce identical output format:

```json
{
  "items": [
    {"name": "apple", "state": "fresh"},
    {"name": "milk", "state": "packaged"}
  ],
  "meta": {
    "pipeline": "llm",
    "image": "groceries.jpg"
  }
}
```

### State Values

| State | Description |
|-------|-------------|
| `fresh` | Raw produce, unpackaged |
| `packaged` | In container/wrapper/box |
| `cooked` | Prepared/cooked food |
| `unknown` | Cannot determine |

## Project Structure

```
├── main.py              # Interactive console app
├── config.py            # Environment configuration
├── clients/
│   ├── llm_client.py    # OpenAI Vision API (frozen prompts)
│   └── yolo_detector.py # YOLOv8 detection
├── pipelines/
│   ├── output.py        # Shared output schema
│   ├── llm_pipeline.py  # System A implementation
│   └── yolo_llm_pipeline.py  # System B implementation
├── requirements.txt
├── .env.example
└── yolov8s.pt           # YOLO model weights
```

## Requirements

- Python 3.10+
- OpenAI API key
- ~50MB disk space (YOLO model)
