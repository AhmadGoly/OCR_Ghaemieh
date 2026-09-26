import asyncio
import io
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.openapi.utils import get_openapi
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
from api.user import router as user_router
from core.deps import get_current_user, get_current_user_optional
from core.security import decode_access_token
from sqlalchemy.ext.asyncio import AsyncSession

# --- Enums for API Documentation ---

class ModelName(str, Enum):
    gemma4 = "gemma4"
    olmocr_2b = "olmocr_2b"
    tesseract = "tesseract"
    docling = "docling"
    qwen = "qwen"
    varco = "varco"

# --- Pydantic Models for API Documentation ---

class BaseOCRResponse(BaseModel):
    text: str = Field(..., description="متن نهایی استخراج‌شده از سند یا تصویر.")
    ocr_model: str = Field(..., description="نام مدل هوش مصنوعی استفاده‌شده برای پردازش.")
    secondary_model: Optional[str] = Field(None, description="نام مدل دوم مورد استفاده در ترکیب دوگانه (در صورت فعال بودن).")
    ocr_duration: float = Field(..., description="مدت زمان پردازش موتور بینایی (ثانیه).")
    llm_duration: float = Field(..., description="مدت زمان تصحیح یا ادغام هوشمند توسط LLM (ثانیه). مقدار ۱- به معنی عدم استفاده است.")

class ImageOCRResponse(BaseOCRResponse):
    original_image: Optional[str] = Field(None, description="تصویر ورودی اولیه با کدگذاری Base64.")
    processed_image: Optional[str] = Field(None, description="تصویر پس از فیلترهای بهینه‌سازی با کدگذاری Base64.")

class PDFPageOCRResponse(BaseOCRResponse):
    page: int = Field(..., description="شماره صفحه پردازش‌شده سند.")

# Concurrency & Worker Resources
ocr_service = None
ocr_executor = ThreadPoolExecutor(
    max_workers=getattr(config, "OCR_THREAD_WORKERS", 24),
    thread_name_prefix="ocr-worker"
)
ocr_semaphore = asyncio.Semaphore(getattr(config, "MAX_CONCURRENT_OCR", 20))

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
    print("Shutting down OCR thread pool executor...")
    ocr_executor.shutdown(wait=False)

app = FastAPI(
    lifespan=lifespan,
    title="سامانه استخراج هوشمند متن قائمیه (Ghaemieh OCR API)",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Custom OpenAPI Schema with Token Auth & User-Centric Documentation
API_DOCUMENTATION_MARKDOWN = """
## راهنمای استفاده و احراز هویت با توکن اختصاصی (API Token Guide)

کلیه درخواست‌ها به این سامانه نیازمند احراز هویت از طریق **توکن اختصاصی (API Token)** می‌باشند. هر کاربر دارای یک توکن یکتا با پیشوند `sk-gh-...` است که می‌تواند در تمامی درخواست‌های برنامه‌نویسی و وب‌سرویس مورد استفاده قرار گیرد.

---

### ۱. نحوه ارسال توکن در هدر درخواست

شما می‌توانید توکن اختصاصی خود را به یکی از دو روش زیر ارسال نمایید:

#### الف) هدر اختصاصی `X-API-Key` (روش پیشنهادی و استاندارد):
```http
X-API-Key: sk-gh-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### ب) هدر استاندارد `Authorization`:
```http
Authorization: Bearer sk-gh-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### ۲. نمونه فراخوانی‌ها

#### نمونه ۱: استخراج متن از تصویر با cURL
```bash
curl -X POST "http://localhost:4567/ocr/image" \\
  -H "X-API-Key: YOUR_API_TOKEN" \\
  -F "file=@sample.png" \\
  -F "model=gemma4" \\
  -F "lang=eng+ara+fas" \\
  -F "preprocess=true" \\
  -F "contrast=true"
```

#### نمونه ۲: استخراج متن از سند چندصفحه‌ای PDF با Python
```python
import requests

url = "http://localhost:4567/ocr/pdf"
headers = {
    "X-API-Key": "YOUR_API_TOKEN"
}
data = {
    "model": "gemma4",
    "lang": "eng+ara+fas",
    "start_page": 1,
    "end_page": 5,
    "use_llm": "true"
}

with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, headers=headers, data=data, files=files)

print(response.json())
```

---

### ۳. کلیدهای آزمایشی مستقیم در Swagger
برای تست مستقیم اندپوینت‌ها در این صفحه، بر روی دکمه سبز رنگ **Authorize** در بالای صفحه کلیک کرده و توکن خود را در قسمت `ApiKeyAuth` یا `BearerAuth` وارد نمایید.
"""

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="سامانه استخراج هوشمند متن قائمیه (Ghaemieh OCR API)",
        version=config.VERSION,
        description=API_DOCUMENTATION_MARKDOWN,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "توکن اختصاصی خود را وارد نمایید (مثال: sk-gh-...)"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "توکن دسترسی JWT یا کلید API"
        }
    }
    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Register Routers (Internal Admin is excluded from public Swagger docs)
app.include_router(auth_router)
app.include_router(admin_router, include_in_schema=False)
app.include_router(user_router)

# --- Browser Static & Favicon Routes ---

@app.get("/favicon.svg", include_in_schema=False)
async def get_favicon_svg():
    return FileResponse(os.path.join(BASE_DIR, 'favicon.svg'), media_type="image/svg+xml")

@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon_ico():
    return FileResponse(os.path.join(BASE_DIR, 'favicon.ico'), media_type="image/x-icon")

@app.get("/favicon.png", include_in_schema=False)
async def get_favicon_png():
    return FileResponse(os.path.join(BASE_DIR, 'favicon.png'), media_type="image/png")

@app.get("/", include_in_schema=False)
async def read_index(request: Request):
    token = request.cookies.get("access_token")
    if not token or not decode_access_token(token):
        return RedirectResponse(url="/login")
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get("/login", include_in_schema=False)
async def read_login():
    return FileResponse(os.path.join(BASE_DIR, 'login.html'))

@app.get("/admin", include_in_schema=False)
async def read_admin(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")
    payload = decode_access_token(token)
    if not payload or not payload.get("is_admin"):
        return RedirectResponse(url="/login")
    return FileResponse(os.path.join(BASE_DIR, 'admin.html'))

@app.get("/style.css", include_in_schema=False)
async def read_style():
    return FileResponse(os.path.join(BASE_DIR, 'style.css'))

@app.get("/script.js", include_in_schema=False)
async def read_script():
    return FileResponse(os.path.join(BASE_DIR, 'script.js'))

# --- Main OCR Processing Endpoints (Thread-Pool Offloaded for High Concurrency) ---

@app.post(
    "/ocr/image",
    response_model=ImageOCRResponse,
    tags=["OCR Extraction"],
    summary="استخراج هوشمند متن از تصویر (Extract Text from Image)"
)
async def ocr_image(
    file: UploadFile = File(..., description="فایل تصویر (JPG, PNG, TIFF)"),
    lang: str = Form(config.DEFAULT_LANG, description="زبان‌های موجود در سند (مانند eng+ara+fas)"),
    model: ModelName = Form(config.DEFAULT_MODEL, description="مدل بینایی اصلی استخراج متن"),
    secondary_model: Optional[ModelName] = Form(None, description="مدل دوم اختیاری جهت مقایسه و ادغام هوشمند"),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="تبدیل به مقیاس خاکستری"),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="بهبود خودکار کنتراست (CLAHE)"),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.1, le=5.0, description="ضریب مقیاس‌بندی ابعاد تصویر"),
    crop_whitespaces: bool = Form(False, description="برش حاشیه‌های خالی سند"),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="فعال‌سازی تصحیح و ادغام هوشمند با مدل زبانی"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    پردازش موازی و با کارایی بالای تصویر با استفاده از موتورهای بینایی ماشین (VLM / OCR).
    
    - این متد روی استخر ریسمان اختصاصی (Thread Pool) اجرا شده و ترافیک موازی را بدون قفل کردن سرور پردازش می‌کند.
    - ارسال توکن اختصاصی در هدر `X-API-Key` یا `Authorization: Bearer` الزامی است.
    """
    if model.value not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"مدل '{model.value}' در سرور فعال نیست.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image.load()

    try:
        async with ocr_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                ocr_executor,
                lambda: ocr_service.process_image(
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
            )

        # Record in database history
        try:
            hist = ExtractionHistory(
                user_id=current_user.id,
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

@app.post(
    "/ocr/pdf",
    response_model=List[PDFPageOCRResponse],
    tags=["OCR Extraction"],
    summary="استخراج هوشمند متن از سند PDF (Extract Text from PDF Document)"
)
async def ocr_pdf(
    file: UploadFile = File(..., description="فایل سند PDF"),
    lang: str = Form(config.DEFAULT_LANG, description="زبان‌های سند (مانند eng+ara+fas)"),
    model: ModelName = Form(config.DEFAULT_MODEL, description="مدل بینایی اصلی"),
    secondary_model: Optional[ModelName] = Form(None, description="مدل دوم اختیاری"),
    start_page: int = Form(1, gt=0, description="صفحه شروع استخراج (پیش‌فرض: ۱)"),
    end_page: Optional[int] = Form(None, gt=0, description="صفحه پایان استخراج (اختیاری)"),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="پیش‌پردازش خاکستری"),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="بهبود کنتراست تصویر"),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.1, le=5.0, description="ضریب مقیاس"),
    crop_whitespaces: bool = Form(False, description="برش حاشیه‌ها"),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="تصحیح و ادغام هوشمند با LLM"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    استخراج سریع و موازی متن از اسناد چندصفحه‌ای PDF.
    
    صفحات سند به‌صورت چندریسمانی و موازی پردازش می‌شوند تا زمان انتظار به حداقل برسد.
    """
    if model.value not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"مدل '{model.value}' در سرور فعال نیست.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        pdf_path = tmp_file.name

    try:
        async with ocr_semaphore:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                ocr_executor,
                lambda: ocr_service.process_pdf(
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
            )

        # Record in database history
        try:
            total_ocr = sum(p.get("ocr_duration", 0.0) for p in results) if results else 0.0
            total_llm = sum(p.get("llm_duration", 0.0) for p in results if p.get("llm_duration", -1) > 0)
            hist = ExtractionHistory(
                user_id=current_user.id,
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

@app.get(
    "/health/models",
    tags=["System Health"],
    summary="بررسی وضعیت مدل‌های بارگذاری‌شده (Check Loaded Models Health)"
)
def health_models():
    """وضعیت آمادگی موتورهای بینایی و هوش مصنوعی بارگذاری‌شده در حافظه."""
    if not ocr_service or not ocr_service.models:
        return {}
    return {model: "loaded" for model in ocr_service.models}

@app.get("/health/config", include_in_schema=False)
def health_config():
    """تنظیمات محیطی و پارامترهای فعال سامانه با ماسک‌گذاری کلیدهای محرمانه."""
    return config.get_config_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.FASTAPI_PORT,
        limit_concurrency=100,
        timeout_keep_alive=65
    )
