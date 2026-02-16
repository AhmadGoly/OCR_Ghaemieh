import io
import os
import sys
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from models.tesseract import TesseractModel
from models.qwen import QwenModel
from models.varco import VarcoModel
from models.docling import DoclingModel
from models.olm import OlmOCRModel
from services.merger import LLMMerger
from services.ocr_service import OCRService

# --- Enums for API Documentation ---

class ModelName(str, Enum):
    tesseract = "tesseract"
    docling = "docling"
    qwen = "qwen"
    varco = "varco"
    olmocr_2b = "olmocr_2b"

# --- Pydantic Models for API Documentation ---

class BaseOCRResponse(BaseModel):
    text: str = Field(..., description="The extracted OCR text from the image or page.")
    ocr_model: str = Field(..., description="The name of the OCR model used for processing.")
    secondary_model: Optional[str] = Field(None, description="The name of the secondary OCR model used, if any.")
    ocr_duration: float = Field(..., description="The time taken for the OCR process in seconds.")
    llm_duration: float = Field(..., description="The time taken for the LLM enhancement in seconds. A value of -1 indicates that the LLM was not used.")

class ImageOCRResponse(BaseOCRResponse):
    original_image: Optional[str] = Field(None, description="Base64 encoded original image.")
    processed_image: Optional[str] = Field(None, description="Base64 encoded processed image.")

class PDFPageOCRResponse(BaseOCRResponse):
    page: int = Field(..., description="The page number of the processed page.")

# Global shared instance of our service.
ocr_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_service
    print("Initializing application and loading models...")
    loaded_models = {}

    if config.LOAD_TESSERACT:
        print("Loading Tesseract model...")
        loaded_models['tesseract'] = TesseractModel(default_lang=config.DEFAULT_LANG)
        print("Tesseract model loaded.")

    if config.LOAD_DOCLING:
        print("Loading Docling model...")
        loaded_models['docling'] = DoclingModel()
        print("Docling model loaded.")

    if config.LOAD_QWEN:
        print("Loading Qwen model...")
        try:
            loaded_models['qwen'] = QwenModel()
            print("Qwen model loaded.")
        except Exception as e:
            print(f"Failed to load Qwen: {e}")

    if config.LOAD_VARCO:
        print("Loading Varco model...")
        try:
            loaded_models['varco'] = VarcoModel()
            print("Varco model loaded.")
        except Exception as e:
            print(f"Failed to load Varco: {e}")

    if config.LOAD_OLMOCR_2B:
        print("Loading OlmOCR model...")
        loaded_models['olmocr_2b'] = OlmOCRModel(
            api_key=config.OLMOCR_API_KEY,
            base_url=config.OLMOCR_LLM_URL_V1,
            default_langs=config.DEFAULT_LANG.split('+')
        )
        print("OlmOCR model loaded.")

    merger = LLMMerger(
        api_key=config.DEFAULT_LLM_API_KEY,
        base_url=config.DEFAULT_LLM_URL,
        model_name=config.DEFAULT_LLM_MODEL_NAME
    )

    ocr_service = OCRService(models=loaded_models, merger=merger)

    print("-" * 20)
    print(f"Startup complete. Models loaded: {list(loaded_models.keys())}")
    print("-" * 20)
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Refactored OCR Processing API",
    version="2.0.0",
)

@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get("/style.css", include_in_schema=False)
async def read_style():
    return FileResponse(os.path.join(BASE_DIR, 'style.css'))

@app.get("/script.js", include_in_schema=False)
async def read_script():
    return FileResponse(os.path.join(BASE_DIR, 'script.js'))

@app.post("/ocr/image", response_model=ImageOCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form(config.DEFAULT_LANG),
    model: ModelName = Form(config.DEFAULT_MODEL),
    secondary_model: Optional[ModelName] = Form(None),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS),
    contrast: bool = Form(config.DEFAULT_CONTRAST),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.1, le=5.0),
    crop_whitespaces: bool = Form(False),
    use_llm: bool = Form(config.DEFAULT_USE_LLM),
):
    if model.value not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"Model '{model.value}' not available.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image.load()

    try:
        result = ocr_service.process_image(
            image,
            primary_model_name=model.value,
            secondary_model_name=secondary_model.value if secondary_model else None,
            lang=lang,
            preprocess=preprocess,
            contrast=contrast,
            scale=scale,
            crop_whitespaces=crop_whitespaces,
            use_llm=use_llm
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/pdf", response_model=List[PDFPageOCRResponse])
async def ocr_pdf(
    file: UploadFile = File(...),
    lang: str = Form(config.DEFAULT_LANG),
    model: ModelName = Form(config.DEFAULT_MODEL),
    secondary_model: Optional[ModelName] = Form(None),
    start_page: int = Form(1, gt=0),
    end_page: Optional[int] = Form(None, gt=0),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS),
    contrast: bool = Form(config.DEFAULT_CONTRAST),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.1, le=5.0),
    crop_whitespaces: bool = Form(False),
    use_llm: bool = Form(config.DEFAULT_USE_LLM),
):
    if model.value not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"Model '{model.value}' not available.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        pdf_path = tmp_file.name

    try:
        results = ocr_service.process_pdf(
            pdf_path=pdf_path,
            primary_model_name=model.value,
            secondary_model_name=secondary_model.value if secondary_model else None,
            lang=lang,
            start_page=start_page,
            end_page=end_page,
            preprocess=preprocess,
            contrast=contrast,
            scale=scale,
            crop_whitespaces=crop_whitespaces,
            use_llm=use_llm
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

@app.get("/health/models")
def health_models():
    return {model: "loaded" for model in ocr_service.models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.FASTAPI_PORT)
