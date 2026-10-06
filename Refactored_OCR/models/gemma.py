import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from .base import BaseOCRModel

class GemmaVLMModel(BaseOCRModel):
    """
    OCR Model leveraging Gemma 4 Vision-Language Model via OpenAI-compatible API.
    """
    def __init__(self, api_key: str, base_url: str, model_name: str = None, default_langs: list = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name or "/models/gemma-4-26B-A4B-it-Q8_0.gguf"
        self.default_langs = default_langs if default_langs else ["fas", "eng", "ara"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def process(self, image: Image.Image, lang: str = None) -> str:
        if lang:
            if isinstance(lang, str):
                language_list = lang.split('+')
            else:
                language_list = lang
        else:
            language_list = self.default_langs

        lang_map = {"fas": "Persian", "eng": "English", "ara": "Arabic"}
        languages = ', '.join([lang_map.get(l, l) for l in language_list])

        buffer = BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a high-accuracy OCR engine specializing in Persian, Arabic, and English documents. "
                    "Extract and transcribe all text from the image exactly as it appears with correct line breaks. "
                    "Do not add introductions, explanations, summaries, or Markdown commentary. Output only the verbatim extracted text."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Transcribe all text from this image precisely. Document languages: {languages}.\n\nImage:\n"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8000
        )

        return response.choices[0].message.content or ""

    def ping(self) -> dict:
        """Probe remote Gemma 4 VLM OpenAI-compatible API endpoint health."""
        import urllib.request
        import urllib.error
        import time

        url = f"{self.base_url.rstrip('/')}/models"
        start_t = time.perf_counter()
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Ghaemieh-OCR-Probe/1.0"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                elapsed = round((time.perf_counter() - start_t) * 1000, 2)
                return {
                    "status": "online",
                    "status_code": resp.getcode(),
                    "response_time_ms": elapsed,
                    "model_name": self.model_name
                }
        except urllib.error.HTTPError as e:
            elapsed = round((time.perf_counter() - start_t) * 1000, 2)
            is_reachable = e.code in (200, 401, 403, 404, 405)
            return {
                "status": "online" if is_reachable else "error",
                "status_code": e.code,
                "response_time_ms": elapsed,
                "error": f"HTTP {e.code}: {e.reason}",
                "model_name": self.model_name
            }
        except Exception as e:
            elapsed = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "offline",
                "response_time_ms": elapsed,
                "error": str(e),
                "model_name": self.model_name
            }
