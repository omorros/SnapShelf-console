"""
Configuration module for SnapShelf Console Application.
Manages environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Application Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "inventory.json"

# YOLO Configuration
YOLO_MODEL = "yolov8s.pt"
YOLO_CONFIDENCE = 0.3
