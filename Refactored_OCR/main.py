import io
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
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
from models.gemma import GemmaVLMModel
from services.merger import LLMMerger
from services.ocr_service import OCRService

from db.init_db import init_database
from db.session import get_db
from db.models import User, ExtractionHistory
from api.auth import router as auth_router
from api.admin import router as admin_router
from core.deps import get_current_user_optional
from sqlalchemy.ext.asyncio import AsyncSession

# --- Enums for API Documentation ---

class ModelName(str, Enum):
    tesseract = "tesseract"
    docling = "docling"
    qwen = "qwen"
    varco = "varco"
    olmocr_2b = "olmocr_2b"
    gemma4 = "gemma4"

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
    config.print_startup_banner()
    try:
        await init_database()
    except Exception as e:
        print(f"Warning: Database initialization postponed/failed: {e}", flush=True)
    print("Initializing application and loading models...", flush=True)
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
            default_langs=config.DEFAULT_LANG.split('+'),
            model_name=config.OLMOCR_MODEL_NAME
        )
        print("OlmOCR model loaded.")

    if config.LOAD_GEMMA4:
        print("Loading Gemma 4 VLM model...")
        loaded_models['gemma4'] = GemmaVLMModel(
            api_key=config.GEMMA4_API_KEY,
            base_url=config.GEMMA4_LLM_URL,
            default_langs=config.DEFAULT_LANG.split('+'),
            model_name=config.GEMMA4_MODEL_NAME
        )
        print("Gemma 4 VLM model loaded.")

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
    version=config.VERSION,
)

app.include_router(auth_router)
app.include_router(admin_router)

@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get("/login", include_in_schema=False)
async def read_login():
    return FileResponse(os.path.join(BASE_DIR, 'login.html'))

@app.get("/admin", include_in_schema=False)
async def read_admin():
    return FileResponse(os.path.join(BASE_DIR, 'admin.html'))

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
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
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

        # Record in database history
        try:
            hist = ExtractionHistory(
                user_id=current_user.id if current_user else None,
                filename=file.filename or "image.jpg",
                file_type="image",
                pages_count=1,
                primary_model=model.value,
                secondary_model=secondary_model.value if secondary_model else None,
                use_llm=use_llm,
                ocr_duration=result.get("ocr_duration", 0.0),
                llm_duration=result.get("llm_duration", -1.0)
            )
            db.add(hist)
            await db.commit()
        except Exception:
            pass

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
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
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

        # Record in database history
        try:
            total_ocr = sum(p.get("ocr_duration", 0.0) for p in results) if results else 0.0
            total_llm = sum(p.get("llm_duration", 0.0) for p in results if p.get("llm_duration", -1) > 0)
            hist = ExtractionHistory(
                user_id=current_user.id if current_user else None,
                filename=file.filename or "document.pdf",
                file_type="pdf",
                pages_count=len(results),
                primary_model=model.value,
                secondary_model=secondary_model.value if secondary_model else None,
                use_llm=use_llm,
                ocr_duration=total_ocr,
                llm_duration=total_llm if total_llm > 0 else -1.0
            )
            db.add(hist)
            await db.commit()
        except Exception:
            pass

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

@app.get("/health/models")
def health_models():
    return {model: "loaded" for model in ocr_service.models}

@app.get("/health/config")
def health_config():
    """Returns the loaded runtime and environment configuration (with secrets masked)."""
    return config.get_config_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.FASTAPI_PORT)
