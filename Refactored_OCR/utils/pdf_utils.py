from pdf2image import convert_from_path
from typing import List
from PIL import Image

class PDFUtils:
    @staticmethod
    def pdf_to_images(pdf_path: str, start_page: int = 1, end_page: int = None) -> List[Image.Image]:
        return convert_from_path(pdf_path, first_page=start_page, last_page=end_page)
