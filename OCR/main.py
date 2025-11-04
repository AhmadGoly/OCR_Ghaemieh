import io
import os
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from openai import OpenAI
from ocr import PDFOCRProcessor
import config

app = FastAPI()

# This dictionary will hold the globally shared instances of our processors.
models = {}

@app.on_event("startup")
def load_models():
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

    print("-" * 20)
    print(f"Startup complete. Models loaded: {list(models.keys())}")
    print("-" * 20)


@app.get("/")
def read_root():
    return {"message": "Welcome to the OCR API"}


@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form(config.DEFAULT_LANG),
    model: str = Form(config.DEFAULT_MODEL),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS),
    contrast: bool = Form(config.DEFAULT_CONTRAST),
    scale: float = Form(config.DEFAULT_SCALE),
    use_llm: bool = Form(config.DEFAULT_USE_LLM),
    llm_url: str = Form(config.DEFAULT_LLM_URL),
    llm_model_name: str = Form(config.DEFAULT_LLM_MODEL_NAME),
    llm_api_key: str = Form(config.DEFAULT_LLM_API_KEY),
):
    # Input validation
    if model not in config.ACCEPTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Accepted models are: {config.ACCEPTED_MODELS}")

    requested_langs = set(lang.split('+'))
    if not requested_langs.issubset(config.ACCEPTED_LANGUAGES):
        raise HTTPException(status_code=400, detail=f"Invalid languages provided. Accepted languages are: {config.ACCEPTED_LANGUAGES}")

    if model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not loaded or available.")

    processor = models[model]
    llm_client = None
    if use_llm:
        llm_client = OpenAI(api_key=llm_api_key, base_url=llm_url)

    image = Image.open(io.BytesIO(await file.read()))
    result = processor.process_image(
        image,
        lang=lang,
        preprocess=preprocess,
        contrast=contrast,
        scale=scale,
        rewrite_llm=use_llm,
        llm_client=llm_client,
        llm_model_name=llm_model_name,
    )
    return result


@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    lang: str = Form(config.DEFAULT_LANG),
    model: str = Form(config.DEFAULT_MODEL),
    start_page: int = Form(1),
    end_page: int = Form(None),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS),
    contrast: bool = Form(config.DEFAULT_CONTRAST),
    scale: float = Form(config.DEFAULT_SCALE),
    use_llm: bool = Form(config.DEFAULT_USE_LLM),
    llm_url: str = Form(config.DEFAULT_LLM_URL),
    llm_model_name: str = Form(config.DEFAULT_LLM_MODEL_NAME),
    llm_api_key: str = Form(config.DEFAULT_LLM_API_KEY),
):
    # Input validation
    if model not in config.ACCEPTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Accepted models are: {config.ACCEPTED_MODELS}")

    requested_langs = set(lang.split('+'))
    if not requested_langs.issubset(config.ACCEPTED_LANGUAGES):
        raise HTTPException(status_code=400, detail=f"Invalid languages provided. Accepted languages are: {config.ACCEPTED_LANGUAGES}")

    if model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not loaded or available.")

    processor = models[model]
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
