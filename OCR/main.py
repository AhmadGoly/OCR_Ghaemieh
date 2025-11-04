import io
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from ocr import PDFOCRProcessor
import config

app = FastAPI()

models = {}

def get_processor(model_name: str, llm_props: list = None):
    if model_name not in models:
        if model_name == 'tesseract' and config.LOAD_TESSERACT:
            models['tesseract'] = PDFOCRProcessor(pdf_path=None, ocr_backend='tesseract', llm_props=llm_props)
        elif model_name == 'docling' and config.LOAD_DOCLING:
            models['docling'] = PDFOCRProcessor(pdf_path=None, ocr_backend='docling', llm_props=llm_props)
        elif model_name == 'qwen' and config.LOAD_QWEN:
            try:
                models['qwen'] = PDFOCRProcessor(pdf_path=None, ocr_backend='qwen', llm_props=llm_props)
            except RuntimeError as e:
                print(f"Failed to load Qwen model: {e}")
                raise HTTPException(status_code=500, detail="Qwen model not loaded")
        elif model_name == 'varco' and config.LOAD_VARCO:
            models['varco'] = PDFOCRProcessor(pdf_path=None, ocr_backend='varco', llm_props=llm_props)
        else:
            raise HTTPException(status_code=400, detail="Model not available")
    return models[model_name]

@app.get("/")
def read_root():
    return {"message": "Welcome to the OCR API"}

@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    model: str = Form("tesseract"),
    preprocess: bool = Form(False),
    contrast: bool = Form(False),
    scale: float = Form(1.0),
    use_llm: bool = Form(True),
    llm_url: str = Form("http://192.168.159.92:8080/v1"),
    llm_model_name: str = Form("gemma-3-4b-it-Q8_0"),
    llm_api_key: str = Form("your_dummy_or_real_key"),
):
    llm_props = [llm_api_key, llm_url, llm_model_name] if use_llm else None
    processor = get_processor(model, llm_props)

    image = Image.open(io.BytesIO(await file.read()))

    text, duration = processor.process_image(
        image,
        preprocess=preprocess,
        contrast=contrast,
        scale=scale,
        rewrite_llm=use_llm,
    )

    return {"text": text, "duration": duration}

@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    model: str = Form("tesseract"),
    start_page: int = Form(1),
    end_page: int = Form(None),
    preprocess: bool = Form(False),
    contrast: bool = Form(False),
    scale: float = Form(1.0),
    use_llm: bool = Form(True),
    llm_url: str = Form("http://192.168.159.92:8080/v1"),
    llm_model_name: str = Form("gemma-3-4b-it-Q8_0"),
    llm_api_key: str = Form("your_dummy_or_real_key"),
):
    llm_props = [llm_api_key, llm_url, llm_model_name] if use_llm else None
    processor = get_processor(model, llm_props)

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        processor.pdf_path = tmp.name

    results = processor.process(
        start_page=start_page,
        end_page=end_page,
        preprocess=preprocess,
        contrast=contrast,
        scale=scale,
        rewrite_llm=use_llm,
    )

    import os
    os.unlink(processor.pdf_path)

    return {"results": results}

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
