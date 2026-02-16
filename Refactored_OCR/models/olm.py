import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from .base import BaseOCRModel

class OlmOCRModel(BaseOCRModel):
    def __init__(self, api_key, base_url, default_langs=None):
        self.api_key = api_key
        self.base_url = base_url
        self.default_langs = default_langs if default_langs else ["fas", "eng"]
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
        # Convert to RGB if needed to save as JPEG
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        messages = [
            {
                "role": "system",
                "content": "You are a high-accuracy OCR engine. Output only the extracted text."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Read all text in this image. Languages might be {languages}. Here is the image:\n\n"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model="olmOCR-2-7B-1025-Q8_0.gguf",
            messages=messages,
            max_tokens=8000
        )

        return response.choices[0].message.content
