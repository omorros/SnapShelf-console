"""
LLM client module for OpenAI Vision API.
Frozen prompts for fair research comparison.
"""

import os
import base64
import json
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# =============================================================================
# API KEY - loaded from environment variable
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found in environment.\n"
        "Please create a .env file with: OPENAI_API_KEY=sk-your-key-here\n"
        "Or set the environment variable directly."
    )


# =============================================================================
# FROZEN SETTINGS - Locked for fair comparison
# =============================================================================

# Image detail level: "high" for both systems (fair comparison)
# Options: "low" (faster, cheaper) or "high" (more accurate)
IMAGE_DETAIL = "high"

# Model: gpt-4o-mini for cost efficiency
MODEL = "gpt-4o-mini"

# Temperature: 0 for deterministic outputs
TEMPERATURE = 0


# =============================================================================
# FROZEN PROMPTS - Same logic for both System A and System B
# =============================================================================

FROZEN_PROMPT_SINGLE = """Analyze this image. Identify the food item visible.

Respond with ONLY this JSON (no markdown, no explanation):
{
  "is_food": true,
  "name": "<generic name, not brand>",
  "state": "<fresh|packaged|cooked|unknown>"
}

If NOT food: {"is_food": false}

Rules:
- name: Use generic names (e.g., "apple" not "Granny Smith", "chips" not "Lay's")
- state: fresh (raw produce), packaged (in container/wrapper), cooked (prepared), unknown
"""

FROZEN_PROMPT_MULTI = """Analyze this image. Identify ALL food items visible.

Respond with ONLY this JSON (no markdown, no explanation):
{
  "items": [
    {"name": "<generic name>", "state": "<fresh|packaged|cooked|unknown>"}
  ]
}

If no food visible: {"items": []}

Rules:
- name: Use generic names (e.g., "apple" not "Granny Smith", "milk" not "Lactaid")
- state: fresh (raw produce), packaged (in container/wrapper), cooked (prepared), unknown
- Include ALL distinct food items
"""


# =============================================================================
# OUTPUT NORMALIZATION
# =============================================================================

def normalize_item(item: dict) -> dict:
    """
    Normalize item output for consistency.
    - Lowercase and strip name
    - Validate state enum
    """
    name = item.get("name", "")
    if not name or not isinstance(name, str):
        name = "unknown"
    else:
        name = name.strip().lower()

    state = item.get("state", "")
    if not state or state not in ("fresh", "packaged", "cooked", "unknown"):
        state = "unknown"

    return {"name": name, "state": state}


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """OpenAI Vision API client with frozen prompts and settings."""

    def __init__(self):
        """Initialize with API key from environment."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = MODEL
        self.detail = IMAGE_DETAIL
        self.temperature = TEMPERATURE

    def identify_single(self, image_bytes: bytes) -> Optional[dict]:
        """
        Identify single food item from cropped image.
        Used by System B (YOLO-LLM) for per-crop analysis.

        Args:
            image_bytes: PNG image bytes

        Returns:
            Normalized dict with name, state if food detected, None otherwise
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FROZEN_PROMPT_SINGLE},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": self.detail
                        }}
                    ]
                }],
                max_tokens=150,
                temperature=self.temperature
            )

            content = response.choices[0].message.content.strip()
            content = self._clean_json(content)
            result = json.loads(content)

            if result.get("is_food"):
                return normalize_item(result)
            return None

        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def identify_all(self, image_bytes: bytes) -> List[dict]:
        """
        Identify ALL food items in full image.
        Used by System A (LLM-only).

        Args:
            image_bytes: PNG image bytes

        Returns:
            List of normalized dicts with name, state for each item
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FROZEN_PROMPT_MULTI},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": self.detail
                        }}
                    ]
                }],
                max_tokens=500,
                temperature=self.temperature
            )

            content = response.choices[0].message.content.strip()
            content = self._clean_json(content)
            result = json.loads(content)

            items = result.get("items", [])
            return [normalize_item(item) for item in items]

        except Exception as e:
            print(f"LLM Error: {e}")
            return []

    def _clean_json(self, content: str) -> str:
        """Remove markdown formatting if present."""
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return content.strip()
