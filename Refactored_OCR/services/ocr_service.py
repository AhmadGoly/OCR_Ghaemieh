import time
from typing import Dict, Optional, List
from PIL import Image
from models.base import BaseOCRModel
from utils.image_processing import ImageProcessor
from utils.pdf_utils import PDFUtils
from .merger import LLMMerger

class OCRService:
    def __init__(self, models: Dict[str, BaseOCRModel], merger: Optional[LLMMerger] = None):
        self.models = models
        self.merger = merger

    def process_image(self,
                      image: Image.Image,
                      primary_model_name: str,
                      secondary_model_name: Optional[str] = None,
                      lang: Optional[str] = None,
                      preprocess: bool = False,
                      contrast: bool = False,
                      scale: float = 1.0,
                      crop_whitespaces: bool = False,
                      use_llm: bool = False) -> dict:
        # Preprocessing
        original_image = image.copy()
        processed_image = image

        if preprocess:
            processed_image = ImageProcessor.preprocess_page(processed_image)
        if contrast:
            processed_image = ImageProcessor.enhance_contrast(processed_image)
        if scale != 1.0:
            processed_image = ImageProcessor.rescale_image(processed_image, scale)
        if crop_whitespaces:
            processed_image = ImageProcessor.crop_whitespaces(processed_image)

        # OCR
        start_t = time.time()
        primary_model = self.models.get(primary_model_name)
        if not primary_model:
            # Fallback for old model names or handle error
            if primary_model_name == "olmocr+tesseract+llm":
                 primary_model = self.models.get("olmocr_2b")
                 secondary_model_name = "tesseract"
                 use_llm = True
            else:
                 raise ValueError(f"Primary model {primary_model_name} not found")

        text1 = primary_model.process(processed_image, lang)
        ocr_duration = time.time() - start_t

        final_text = text1
        llm_duration = -1

        ocr_outputs = [text1]

        if use_llm:
            if secondary_model_name:
                secondary_model = self.models.get(secondary_model_name)
                if secondary_model:
                    text2 = secondary_model.process(processed_image, lang)
                    ocr_outputs.append(text2)

            if self.merger:
                llm_start = time.time()
                final_text = self.merger.merge(ocr_outputs)
                llm_duration = time.time() - llm_start

        return {
            "text": final_text,
            "ocr_model": primary_model_name,
            "secondary_model": secondary_model_name if len(ocr_outputs) > 1 else None,
            "ocr_duration": ocr_duration,
            "llm_duration": llm_duration,
            "original_image": ImageProcessor.image_to_base64(original_image),
            "processed_image": ImageProcessor.image_to_base64(processed_image),
        }

    def process_pdf(self,
                    pdf_path: str,
                    primary_model_name: str,
                    secondary_model_name: Optional[str] = None,
                    start_page: int = 1,
                    end_page: Optional[int] = None,
                    lang: Optional[str] = None,
                    preprocess: bool = False,
                    contrast: bool = False,
                    scale: float = 1.0,
                    crop_whitespaces: bool = False,
                    use_llm: bool = False) -> List[dict]:
        images = PDFUtils.pdf_to_images(pdf_path, start_page, end_page)
        results = []
        for i, img in enumerate(images, start=start_page):
            res = self.process_image(img, primary_model_name, secondary_model_name, lang,
                                     preprocess, contrast, scale, crop_whitespaces, use_llm)
            res["page"] = i
            # Remove images from result to match original PDF response
            res.pop("original_image", None)
            res.pop("processed_image", None)
            results.append(res)
        return results
