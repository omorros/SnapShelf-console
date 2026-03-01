"""
Anthropic VLM client: Claude Opus 4.6 constrained to 14 labels.

Sends the full image to Claude Opus 4.6 with a prompt that explicitly lists
all 14 class names and instructs the model to count precisely.
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict

import anthropic

from config import CLASSES

MODEL = "claude-opus-4-6"

FROZEN_PROMPT = f"""You are an inventory counting assistant. Examine the image and count every
visible item that matches one of the following 14 classes:

{', '.join(CLASSES)}

Rules:
- ONLY use the exact class names listed above (case-sensitive).
- Count each individual item you can see. If there are 3 apples, report "apple": 3.
- If a class is not present, do NOT include it.
- Do NOT invent classes outside the list.
- Respond ONLY with valid JSON in this exact format (no markdown, no explanation):

{{"inventory": {{"class_name": count, ...}}}}

Example: {{"inventory": {{"apple": 3, "banana": 1, "tomato": 2}}}}
"""

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for the image."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def identify(image_path: str) -> Dict[str, int]:
    """Send image to Claude Opus 4.6 and return inventory dict."""
    client = _get_client()
    b64_data, media_type = _encode_image(image_path)

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": FROZEN_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    return _parse_response(raw)


def _parse_response(raw: str) -> Dict[str, int]:
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    raw_inventory = data.get("inventory", data)
    if not isinstance(raw_inventory, dict):
        return {}

    valid_classes = set(CLASSES)
    inventory: Dict[str, int] = {}
    for key, value in raw_inventory.items():
        key_clean = key.strip().lower()
        if key_clean in valid_classes:
            try:
                count = int(value)
                if count > 0:
                    inventory[key_clean] = count
            except (ValueError, TypeError):
                continue
    return inventory
