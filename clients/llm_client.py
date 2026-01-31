"""
LLM client module for OpenAI Vision API.

FROZEN PROMPTS AND SETTINGS for fair research comparison.

EXPERIMENTAL DESIGN NOTE (Prompt Difference):
- identify_all(): Used by Pipeline A for full images. Asks LLM to find ALL items
  in a single image. Uses multi-target prompt with higher max_tokens (500).
- identify_single(): Used by Pipelines B/C for cropped regions. Asks LLM to
  identify ONE item per crop. Uses single-target prompt with lower max_tokens (150).

This difference is INTENTIONAL and FAIR because:
1. Pipeline A receives ONE full image and must enumerate all items
2. Pipelines B/C receive MULTIPLE crops, each expected to contain ONE item
3. The per-item cognitive task is equivalent; only the enumeration differs

The max_tokens difference reflects expected output size:
- Multi-item: {"items": [{...}, {...}, ...]} - needs more tokens
- Single-item: {"is_food": true, "name": "...", "state": "..."} - needs fewer
"""

import os
import base64
import json
import time
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from config import CONFIG, get_logger, Timer

# Load environment variables
load_dotenv()


# =============================================================================
# FROZEN PROMPTS - Locked for fair comparison
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


def _clean_json(content: str) -> str:
    """Remove markdown formatting if present."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json) and last line (```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return content.strip()


# =============================================================================
# LLM CLIENT (Singleton pattern for efficiency)
# =============================================================================

class LLMClient:
    """
    OpenAI Vision API client with frozen prompts and settings.

    Uses singleton pattern to avoid repeated client initialization,
    which would bias timing measurements.
    """

    _instance: Optional["LLMClient"] = None
    _initialized: bool = False

    def __new__(cls) -> "LLMClient":
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize with API key from environment (only once)."""
        if LLMClient._initialized:
            return

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found.\n"
                "To fix this:\n"
                "  1. Create a .env file in the project root\n"
                "  2. Add this line: OPENAI_API_KEY=sk-your-key-here\n"
                "  Or set the environment variable directly:\n"
                "  - Windows: set OPENAI_API_KEY=sk-your-key-here\n"
                "  - Linux/Mac: export OPENAI_API_KEY=sk-your-key-here"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = CONFIG.llm_model
        self.detail = CONFIG.llm_image_detail
        self.temperature = CONFIG.llm_temperature

        LLMClient._initialized = True

    def identify_single(
        self,
        image_bytes: bytes,
        pipeline: str = "unknown",
        image_name: str = "unknown",
        crop_index: Optional[int] = None
    ) -> Tuple[Optional[dict], float]:
        """
        Identify single food item from cropped image.
        Used by Pipeline B and C for per-crop analysis.

        Args:
            image_bytes: PNG image bytes
            pipeline: Pipeline name for logging
            image_name: Image filename for logging
            crop_index: Crop index for logging

        Returns:
            Tuple of (normalized dict with name/state if food detected, None otherwise)
            and duration in milliseconds
        """
        logger = get_logger()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        raw_response = ""
        parsed_result = None
        error_msg = None

        with Timer("llm_single") as timer:
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
                    max_tokens=CONFIG.llm_max_tokens_single,
                    temperature=self.temperature
                )

                raw_response = response.choices[0].message.content.strip()
                content = _clean_json(raw_response)
                result = json.loads(content)

                if result.get("is_food"):
                    parsed_result = normalize_item(result)
                else:
                    parsed_result = None

            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error: {e}. Raw: {raw_response[:200]}"
                parsed_result = None

            except Exception as e:
                error_msg = f"LLM API error: {type(e).__name__}: {e}"
                parsed_result = None

        # Log the call
        logger.log_llm_call(
            pipeline=pipeline,
            image=image_name,
            crop_index=crop_index,
            raw_response=raw_response,
            parsed_result=parsed_result,
            duration_ms=timer.duration_ms,
            error=error_msg
        )

        return parsed_result, timer.duration_ms

    def identify_all(
        self,
        image_bytes: bytes,
        pipeline: str = "llm",
        image_name: str = "unknown"
    ) -> Tuple[List[dict], float]:
        """
        Identify ALL food items in full image.
        Used by Pipeline A (LLM-only baseline).

        Args:
            image_bytes: PNG image bytes
            pipeline: Pipeline name for logging
            image_name: Image filename for logging

        Returns:
            Tuple of (list of normalized dicts with name/state for each item)
            and duration in milliseconds
        """
        logger = get_logger()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        raw_response = ""
        parsed_result = []
        error_msg = None

        with Timer("llm_multi") as timer:
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
                    max_tokens=CONFIG.llm_max_tokens_multi,
                    temperature=self.temperature
                )

                raw_response = response.choices[0].message.content.strip()
                content = _clean_json(raw_response)
                result = json.loads(content)

                items = result.get("items", [])
                parsed_result = [normalize_item(item) for item in items]

            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error: {e}. Raw: {raw_response[:200]}"
                parsed_result = []

            except Exception as e:
                error_msg = f"LLM API error: {type(e).__name__}: {e}"
                parsed_result = []

        # Log the call
        logger.log_llm_call(
            pipeline=pipeline,
            image=image_name,
            crop_index=None,
            raw_response=raw_response,
            parsed_result=parsed_result,
            duration_ms=timer.duration_ms,
            error=error_msg
        )

        return parsed_result, timer.duration_ms


# =============================================================================
# MODULE-LEVEL CLIENT (for timing fairness)
# =============================================================================

def get_llm_client() -> LLMClient:
    """
    Get the singleton LLM client instance.
    Use this instead of LLMClient() to ensure initialization
    happens before timing starts.
    """
    return LLMClient()


def warmup_llm_client():
    """
    Warm up the LLM client by initializing it.
    Call this before timing measurements to exclude init time.
    """
    _ = get_llm_client()
