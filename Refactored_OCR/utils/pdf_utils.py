from pdf2image import convert_from_path, pdfinfo_from_path
from typing import List, Optional
from PIL import Image


class PDFUtils:
    """Helper utilities for PDF inspection and page-to-image conversion."""

    @staticmethod
    def pdf_to_images(pdf_path: str, start_page: int = 1, end_page: Optional[int] = None) -> List[Image.Image]:
        """Convert a range of PDF pages into PIL Images."""
        return convert_from_path(pdf_path, first_page=start_page, last_page=end_page)

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Retrieve total page count of a PDF without loading full images into memory."""
        try:
            info = pdfinfo_from_path(pdf_path)
            pages = int(info.get("Pages", 0))
            if pages > 0:
                return pages
        except Exception:
            pass

        # Fallback if pdfinfo binary fails
        images = convert_from_path(pdf_path)
        count = len(images)
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        return max(1, count)

    @staticmethod
    def render_single_page(pdf_path: str, page_number: int) -> Optional[Image.Image]:
        """Render a single PDF page into a PIL Image to keep RAM usage minimal during book OCR."""
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        if not images:
            return None
        # Close any unexpected extra images if returned
        for extra in images[1:]:
            try:
                extra.close()
            except Exception:
                pass
        return images[0]

