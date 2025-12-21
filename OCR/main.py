print("Importing standard libraries...")
import io
import os
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager
from enum import Enum

print("Importing FastAPI and Pydantic...")
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

print("Importing imaging and ML libraries...")
import torch
from PIL import Image
from openai import OpenAI

print("Importing local modules...")
from ocr import PDFOCRProcessor
import config

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
    ocr_duration: float = Field(..., description="The time taken for the OCR process in seconds.")
    llm_model: Optional[str] = Field(None, description="The name of the Language Model used for text enhancement, if any.")
    llm_duration: float = Field(..., description="The time taken for the LLM enhancement in seconds. A value of -1 indicates that the LLM was not used.")

class ImageOCRResponse(BaseOCRResponse):
    original_image: Optional[str] = Field(None, description="Base64 encoded original image.")
    processed_image: Optional[str] = Field(None, description="Base64 encoded processed image.")

class PDFPageOCRResponse(BaseOCRResponse):
    page: int = Field(..., description="The page number of the processed page.")

# This dictionary will hold the globally shared instances of our processors.
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load OCR models at application startup.
    This runs only once when the FastAPI app starts.
    """
    print("Initializing application and loading models...")
    if config.LOAD_TESSERACT:
        print("Loading Tesseract model...")
        models['tesseract'] = PDFOCRProcessor(ocr_backend='tesseract')
        print("Tesseract model loaded successfully.")
    if config.LOAD_DOCLING:
        print("Loading Docling model...")
        models['docling'] = PDFOCRProcessor(ocr_backend='docling')
        print("Docling model loaded successfully.")
    if config.LOAD_QWEN:
        print("Loading Qwen model...")
        try:
            models['qwen'] = PDFOCRProcessor(ocr_backend='qwen')
            print("Qwen model loaded successfully.")
        except RuntimeError as e:
            print(f"Failed to load Qwen model on startup: {e}")
    if config.LOAD_VARCO:
        print("Loading Varco model...")
        models['varco'] = PDFOCRProcessor(ocr_backend='varco')
        print("Varco model loaded successfully.")
    if config.LOAD_OLMOCR_2B:
        print("Loading OLMOCR 2B model...")
        models['olmocr_2b'] = PDFOCRProcessor(ocr_backend='olmocr_2b')
        print("OLMOCR 2B model loaded successfully.")

    print("-" * 20)
    print(f"Startup complete. Models loaded: {list(models.keys())}")
    print("-" * 20)
    yield
    # No cleanup needed, but yield is required for the context manager

app = FastAPI(
    lifespan=lifespan,
    title="OCR Processing API",
    description="""
A powerful and flexible API for performing Optical Character Recognition (OCR) on images and PDF documents.

This API provides endpoints to process files using various OCR backends, including **Tesseract**, **Docling**, and GPU-accelerated models like **Qwen** and **Varco**.
It also supports optional text enhancement using a configurable Large Language Model (LLM).

### Features:
-   **Image and PDF Processing**: Endpoints for both single image files and multi-page PDF documents.
-   **Configurable OCR Backends**: Choose the best model for your needs on a per-request basis.
-   **Image Preprocessing**: Options to enable preprocessing, contrast enhancement, and scaling to improve OCR accuracy.
-   **LLM-Powered Text Correction**: Optionally use a language model to correct and clean up the raw OCR output.
-   **Detailed Responses**: Get structured JSON responses with the extracted text, model information, and performance metrics.
    """,
    version="1.0.0",
)

# Get the absolute path to the directory containing main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get("/style.css", include_in_schema=False)
async def read_style():
    return FileResponse(os.path.join(BASE_DIR, 'style.css'))

@app.get("/script.js", include_in_schema=False)
async def read_script():
    return FileResponse(os.path.join(BASE_DIR, 'script.js'))


@app.post("/ocr/image",
          summary="Perform OCR on a Single Image",
          description="Upload an image file to extract text content. This endpoint is ideal for processing single-page documents, photographs of text, or screenshots.",
          response_model=ImageOCRResponse)
async def ocr_image(
    file: UploadFile = File(..., description="The image file to be processed. Common formats like PNG, JPEG, and TIFF are supported."),
    lang: str = Form(config.DEFAULT_LANG, description="The language(s) to be used for OCR, specified in Tesseract format (e.g., 'eng+fas').", examples=["eng+fas"]),
    model: ModelName = Form(config.DEFAULT_MODEL, description="The OCR model to use for processing. Choose 'tesseract' or 'docling' for CPU-based processing, or 'qwen'/'varco' for GPU-accelerated models."),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="If true, applies adaptive thresholding and morphological operations to clean up the image before OCR."),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="If true, enhances the image contrast, which can improve OCR accuracy on washed-out documents."),
    scale: float = Form(config.DEFAULT_SCALE, description="A scaling factor for the image. Values less than 1.0 will downscale, while values greater than 1.0 will upscale. A value of 1.0 means no change.", ge=0.1, le=5.0),
    crop_whitespaces: bool = Form(False, description="If true, crops the whitespaces from the borders of the image."),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="If true, a Large Language Model will be used to correct and enhance the raw OCR output."),
    llm_url: str = Form(config.DEFAULT_LLM_URL, description="The base URL for the LLM API endpoint.", examples=["http://192.168.159.92:8080/v1"]),
    llm_model_name: str = Form(config.DEFAULT_LLM_MODEL_NAME, description="The specific name of the LLM to use for text enhancement.", examples=["gemma-3-4b-it-Q8_0"]),
    llm_api_key: str = Form(config.DEFAULT_LLM_API_KEY, description="The API key for authenticating with the LLM service."),
):
    # Input validation
    if model.value not in config.ACCEPTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Accepted models are: {config.ACCEPTED_MODELS}")

    requested_langs = set(lang.split('+'))
    if not requested_langs.issubset(config.ACCEPTED_LANGUAGES):
        raise HTTPException(status_code=400, detail=f"Invalid languages provided. Accepted languages are: {config.ACCEPTED_LANGUAGES}")

    if model.value not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model.value}' is not loaded or available. Check the `config.py` file to enable it.")

    processor = models[model.value]
    llm_client = None
    if use_llm:
        llm_client = OpenAI(api_key=llm_api_key, base_url=llm_url)

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image.load()  # Force loading the image data to prevent errors

    result = processor.process_image(
        image,
        lang=lang,
        preprocess=preprocess,
        contrast=contrast,
        scale=scale,
        crop_whitespaces=crop_whitespaces,
        rewrite_llm=use_llm,
        llm_client=llm_client,
        llm_model_name=llm_model_name,
    )
    return result


@app.post("/ocr/pdf",
          summary="Perform OCR on a PDF Document",
          description="Upload a PDF file to extract text content from a specified range of pages. This is suitable for multi-page documents.",
          response_model=List[PDFPageOCRResponse])
async def ocr_pdf(
    file: UploadFile = File(..., description="The PDF document to be processed."),
    lang: str = Form(config.DEFAULT_LANG, description="The language(s) to be used for OCR, specified in Tesseract format (e.g., 'eng+fas').", examples=["eng+fas"]),
    model: ModelName = Form(config.DEFAULT_MODEL, description="The OCR model to use for processing. Choose 'tesseract' or 'docling' for CPU-based processing, or 'qwen'/'varco' for GPU-accelerated models."),
    start_page: int = Form(1, description="The first page of the document to process (1-indexed).", gt=0),
    end_page: Optional[int] = Form(None, description="The last page of the document to process. If omitted, all pages from the start page to the end of the document will be processed.", gt=0),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="If true, applies adaptive thresholding and morphological operations to clean up each page's image before OCR."),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="If true, enhances the contrast of each page, which can improve OCR accuracy."),
    scale: float = Form(config.DEFAULT_SCALE, description="A scaling factor for each page's image. Values less than 1.0 will downscale, while values greater than 1.0 will upscale. A value of 1.0 means no change.", ge=0.1, le=5.0),
    crop_whitespaces: bool = Form(False, description="If true, crops the whitespaces from the borders of each page's image."),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="If true, a Large Language Model will be used to correct and enhance the raw OCR output for each page."),
    llm_url: str = Form(config.DEFAULT_LLM_URL, description="The base URL for the LLM API endpoint.", examples=["http://192.168.159.92:8080/v1"]),
    llm_model_name: str = Form(config.DEFAULT_LLM_MODEL_NAME, description="The specific name of the LLM to use for text enhancement.", examples=["gemma-3-4b-it-Q8_0"]),
    llm_api_key: str = Form(config.DEFAULT_LLM_API_KEY, description="The API key for authenticating with the LLM service."),
):
    # Input validation
    if model.value not in config.ACCEPTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Accepted models are: {config.ACCEPTED_MODELS}")

    requested_langs = set(lang.split('+'))
    if not requested_langs.issubset(config.ACCEPTED_LANGUAGES):
        raise HTTPException(status_code=400, detail=f"Invalid languages provided. Accepted languages are: {config.ACCEPTED_LANGUAGES}")

    if model.value not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model.value}' is not loaded or available. Check the `config.py` file to enable it.")

    processor = models[model.value]
    llm_client = None
    if use_llm:
        llm_client = OpenAI(api_key=llm_api_key, base_url=llm_url)

    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        pdf_path = tmp_file.name

    try:
        results = processor.process(
            pdf_path=pdf_path,
            lang=lang,
            start_page=start_page,
            end_page=end_page,
            preprocess=preprocess,
            contrast=contrast,
            scale=scale,
            crop_whitespaces=crop_whitespaces,
            rewrite_llm=use_llm,
            llm_client=llm_client,
            llm_model_name=llm_model_name,
        )
        return results

    finally:
        # Ensure the temporary file is always deleted
        os.unlink(pdf_path)


@app.get("/health/models")
def health_models():
    return {model: "loaded" for model in models}


@app.get("/health/gpu")
def health_gpu():
    if torch.cuda.is_available():
        return {
            "status": "available",
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0),
        }
    return {"status": "unavailable"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.FASTAPI_PORT)
