"""
OpenAI VLM client: GPT-5.2 constrained to 14 labels.

Sends the full image to GPT-5.2 with a prompt that explicitly lists
all 14 class names and instructs the model to count precisely.
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict

from openai import OpenAI

from config import CLASSES

MODEL = "gpt-5.2"

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

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def _encode_image(image_path: str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def identify(image_path: str) -> Dict[str, int]:
    """Send image to GPT-5.2 and return inventory dict."""
    client = _get_client()
    data_uri = _encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_completion_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FROZEN_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
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
