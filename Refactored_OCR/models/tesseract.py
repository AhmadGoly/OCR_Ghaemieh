import pytesseract
import re
from .base import BaseOCRModel
from PIL import Image

class TesseractModel(BaseOCRModel):
    def __init__(self, default_lang: str = "eng+ara+fas"):
        self.default_lang = default_lang

    def process(self, image: Image.Image, lang: str = None) -> str:
        use_lang = lang if lang else self.default_lang
        result = pytesseract.image_to_string(image, lang=use_lang)
        # Post-process: replace single newlines with spaces
        result = re.sub(r'(?<!\n)\n(?!\n)', ' ', result)
        return result
