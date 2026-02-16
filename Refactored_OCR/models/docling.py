import tempfile
from pathlib import Path
from PIL import Image
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from .base import BaseOCRModel

class DoclingModel(BaseOCRModel):
    def __init__(self, default_langs=None):
        self.default_langs = default_langs if default_langs else ["eng", "ara", "fas"]

    def process(self, image: Image.Image, lang: str = None) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            image.save(tmp_path, format="PNG")

        # lang can be a string like "eng+ara" or a list.
        # Docling expects a list of languages for TesseractCliOcrOptions
        if lang:
            if isinstance(lang, str):
                use_langs = lang.split('+')
            else:
                use_langs = lang
        else:
            use_langs = self.default_langs

        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=use_langs)
        pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
        converter = DocumentConverter(
            format_options={InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        try:
            doc = converter.convert(tmp_path).document
            result = doc.export_to_markdown()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return result
