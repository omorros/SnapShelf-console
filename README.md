# SnapShelf Console

AI-powered food inventory management CLI using YOLOv8 and GPT-4o mini Vision.

## Features

- **AI Food Detection**: Automatically detect food items in images using YOLOv8
- **Smart Identification**: Identify food types, categories, and expiry estimates using GPT-4o mini
- **Expiry Tracking**: Track when items will expire with visual indicators
- **Simple CLI**: Easy-to-use command-line interface
- **Local Storage**: All data stored locally in JSON format

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SnapShelf-console
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Scan Food Image

Detect and identify food items from an image:

```bash
python main.py scan path/to/image.jpg
```

### List Inventory

View all items in your inventory:

```bash
python main.py list
```

Options:
- `--category` or `-c`: Filter by category (e.g., Fruit, Dairy, Vegetable)
- `--sort` or `-s`: Sort by expiry, name, or added date
- `--status`: Filter by active, consumed, or discarded

Examples:
```bash
# List only fruits
python main.py list --category Fruit

# Sort by name
python main.py list --sort name

# View consumed items
python main.py list --status consumed
```

### View Expiring Items

See items expiring soon:

```bash
python main.py expiring
```

Options:
- `--days` or `-d`: Number of days to look ahead (default: 7)

Example:
```bash
# View items expiring in next 3 days
python main.py expiring --days 3
```

### Mark Items as Consumed

Record when you use items:

```bash
python main.py consume <item-id>

# Multiple items
python main.py consume <id1> <id2> <id3>
```

### Mark Items as Discarded

Track food waste:

```bash
python main.py discard <item-id>
```

### Remove Items

Permanently delete items from inventory:

```bash
python main.py remove <item-id>
```

### Clear Inventory

Remove all items:

```bash
python main.py clear

# Skip confirmation
python main.py clear --confirm
```

## Project Structure

```
snapshelf-console/
├── main.py                 # CLI entry point
├── config.py               # Configuration & environment
├── services/
│   ├── yolo_service.py     # YOLOv8 detection
│   ├── llm_service.py      # OpenAI Vision API
│   └── scan_service.py     # Orchestrates detection + identification
├── storage/
│   └── inventory.py        # JSON file operations
├── models/
│   └── food_item.py        # Data models (dataclasses)
├── utils/
│   └── display.py          # Terminal formatting (rich)
├── data/
│   └── inventory.json      # Local storage file
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.11+**
- **Typer**: CLI framework
- **YOLOv8**: Object detection
- **OpenAI GPT-4o mini**: Food identification
- **Rich**: Terminal formatting
- **Pillow**: Image processing

## Data Storage

All inventory data is stored locally in `data/inventory.json`. No external database required.

## License

MIT License

## Author

Oriol Morros Vilaseca
