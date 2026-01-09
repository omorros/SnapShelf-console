"""
LLM-based food identification service using OpenAI Vision API.
Analyzes cropped images to identify food items and extract metadata.
"""

import base64
import json
from openai import OpenAI
from typing import Optional, List
from config import OPENAI_API_KEY


class LLMService:
    """
    Food identification service using GPT-4o mini with vision.

    Attributes:
        client: OpenAI API client
        model: Model identifier to use
    """

    def __init__(self):
        """Initialize LLM service with OpenAI client."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-mini"

    def identify_food(self, image_bytes: bytes) -> Optional[dict]:
        """
        Analyze cropped image and identify food item.

        Args:
            image_bytes: Binary image data (PNG format)

        Returns:
            Dictionary containing food metadata if food detected, None otherwise
            Expected keys: name, category, freshness, expiry_days, confidence, storage_tip
        """
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct prompt for food identification
        prompt = """Analyze this image. If it contains a food item, respond with ONLY this JSON:
{
  "is_food": true,
  "name": "Specific name (e.g., 'Granny Smith Apple')",
  "category": "One of: Fruit, Vegetable, Dairy, Meat, Seafood, Grain, Beverage, Condiment, Snack, Prepared Food, Other",
  "freshness": "One of: Fresh, Good, Fair, Poor",
  "expiry_days": <integer estimate>,
  "unit": "Best unit for this item: 'unit' for countable items (eggs, apples), 'g' or 'kg' for weight-based, 'ml' or 'L' for liquids",
  "confidence": <float 0-1>,
  "storage_tip": "Brief storage recommendation"
}

If NOT a food item, respond: {"is_food": false}

Return ONLY valid JSON."""

        try:
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "low"
                        }}
                    ]
                }],
                max_tokens=300,
                temperature=0.1
            )

            # Extract and parse response
            content = response.choices[0].message.content.strip()

            # Clean markdown formatting if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            # Parse JSON response
            result = json.loads(content)

            # Return food data if identified as food
            if result.get("is_food"):
                return result
            return None

        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def identify_all_foods(self, image_bytes: bytes) -> List[dict]:
        """
        Analyze image and identify ALL food items visible.

        Args:
            image_bytes: Binary image data (PNG/JPEG format)

        Returns:
            List of dictionaries containing food metadata for each item found
        """
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct prompt for multi-food identification
        prompt = """Analyze this image and identify ALL food items visible.
For EACH food item, provide details in a JSON array.

Respond with ONLY this JSON format:
{
  "foods": [
    {
      "name": "Specific name (e.g., 'Blueberries', 'Gnocchi')",
      "category": "One of: Fruit, Vegetable, Dairy, Meat, Seafood, Grain, Beverage, Condiment, Snack, Prepared Food, Other",
      "freshness": "One of: Fresh, Good, Fair, Poor",
      "expiry_days": <integer estimate based on typical shelf life>,
      "unit": "Best unit: 'unit' for countable items, 'g' or 'kg' for weight-based, 'ml' or 'L' for liquids",
      "confidence": <float 0-1>,
      "storage_tip": "Brief storage recommendation"
    }
  ]
}

Important:
- Include ALL distinct food items you can see
- Packaged foods count (gnocchi, yogurt, etc.)
- If no food items visible, return {"foods": []}

Return ONLY valid JSON."""

        try:
            # Call OpenAI Vision API with higher detail for full image
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }}
                    ]
                }],
                max_tokens=1000,
                temperature=0.1
            )

            # Extract and parse response
            content = response.choices[0].message.content.strip()

            # Clean markdown formatting if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            # Parse JSON response
            result = json.loads(content)
            return result.get("foods", [])

        except Exception as e:
            print(f"LLM Error: {e}")
            return []
