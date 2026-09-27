import asyncio
import io
import os
import sys
import time
from datetime import datetime, timezone
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import tempfile
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import json
import socket
import urllib.request
import urllib.error

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, Response
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

def resolve_model_name(name: Optional[Union[str, ModelName]]) -> Optional[str]:
    """Normalize model identifiers and aliases to canonical model names."""
    if not name:
        return None
    val = name.value if isinstance(name, Enum) else str(name)
    val = val.lower().strip()
    if val in ("olmocr", "olmocr_7b"):
        return "olmocr_2b"
    if val in ("gemma", "gemma_4"):
        return "gemma4"
    return val

# --- Pydantic Models for API Documentation ---

class BaseOCRResponse(BaseModel):
    text: str = Field(..., description="Final extracted plain text from the document or image.")
    ocr_model: str = Field(..., description="Name of the vision or OCR model used for extraction.")
    secondary_model: Optional[str] = Field(None, description="Secondary model name if dual-model merging was enabled.")
    ocr_duration: float = Field(..., description="Vision/OCR extraction processing time in seconds.")
    llm_duration: float = Field(..., description="LLM text reconciliation/cleaning duration in seconds (-1.0 if not used).")

class ImageOCRResponse(BaseOCRResponse):
    original_image: Optional[str] = Field(None, description="Base64-encoded raw input image.")
    processed_image: Optional[str] = Field(None, description="Base64-encoded preprocessed/enhanced image.")

class PDFPageOCRResponse(BaseOCRResponse):
    page: int = Field(..., description="Processed document page number.")

class ServiceHealthDetail(BaseModel):
    service: str = Field(..., description="Service identifier (e.g. LLM, OLM).")
    url: str = Field(..., description="Probed health check URL.")
    status: str = Field(..., description="Health status ('online', 'timeout', 'offline', or 'error').")
    status_code: Optional[int] = Field(None, description="HTTP status code received (200, 404, 500, etc.).")
    response: Optional[Any] = Field(None, description="Parsed JSON response payload or raw text (e.g. {'status':'ok'}).")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds.")
    error: Optional[str] = Field(None, description="Error or timeout description if the probe failed.")

class HealthStatusResponse(BaseModel):
    status: str = Field(..., description="Overall health state ('healthy', 'degraded', or 'unhealthy').")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of the health check probe.")
    services: Dict[str, ServiceHealthDetail] = Field(..., description="Health details for remote inference services.")

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
    title="Ghaemieh Intelligent OCR API",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Custom OpenAPI Schema with Token Auth & User-Centric Documentation
API_DOCUMENTATION_MARKDOWN = """
## Authentication & API Token Guide

All document extraction endpoints require authentication using a **dedicated API token (`sk-gh-...`)** or a **JWT Bearer token**. Every registered user is assigned a unique API key that can be copied or regenerated from the web portal.

---

### 1. Sending Authentication in Request Headers

Authenticate your HTTP requests using either of the following standard headers:

#### A) Dedicated `X-API-Key` Header (Recommended):
```http
X-API-Key: sk-gh-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### B) Standard `Authorization` Header:
```http
Authorization: Bearer sk-gh-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 2. Code Examples

#### Example 1: Extract Text from an Image using cURL
```bash
curl -X POST "http://localhost:4567/ocr/image" \\
  -H "X-API-Key: YOUR_API_TOKEN" \\
  -F "file=@sample.png" \\
  -F "model=gemma4" \\
  -F "lang=eng+ara+fas" \\
  -F "preprocess=true" \\
  -F "contrast=true"
```

#### Example 2: Extract Text from a Multi-Page PDF using Python
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

### 3. Interactive Testing in Swagger UI
Click the green **Authorize** button at the top right of this documentation page and enter your API token in either `ApiKeyAuth` or `BearerAuth` to test the endpoints directly from your browser.
"""

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Ghaemieh Intelligent OCR API",
        version=config.VERSION,
        description=API_DOCUMENTATION_MARKDOWN,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Enter your dedicated API key (e.g. sk-gh-...)"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "JWT access token or Bearer API key"
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
    summary="Extract Text from Image"
)
async def ocr_image(
    file: UploadFile = File(..., description="Document image file (JPG, PNG, TIFF, etc.)"),
    lang: str = Form(config.DEFAULT_LANG, description="Document language codes (e.g. eng+ara+fas)"),
    model: ModelName = Form(config.DEFAULT_MODEL, description="Primary AI vision/OCR model"),
    secondary_model: Optional[ModelName] = Form(None, description="Optional secondary model for dual-model reconciliation"),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="Convert to grayscale with adaptive thresholding"),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="Automatic contrast enhancement (CLAHE)"),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.1, le=5.0, description="Image dimension rescaling multiplier"),
    crop_whitespaces: bool = Form(False, description="Auto-crop document white margins"),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="Enable LLM post-processing and text merging"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    High-performance parallel OCR text extraction from image files.
    
    - Executed on a dedicated thread pool to ensure non-blocking concurrent request handling.
    - Requires authentication via `X-API-Key` or `Authorization: Bearer <token>`.
    """
    primary_model_name = resolve_model_name(model)
    if not primary_model_name or primary_model_name not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"Model '{model.value if isinstance(model, Enum) else model}' is not active on this server.")

    secondary_model_name = resolve_model_name(secondary_model) if secondary_model else None
    if secondary_model_name and secondary_model_name not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"Secondary model '{secondary_model.value if isinstance(secondary_model, Enum) else secondary_model}' is not active on this server.")

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
                    primary_model_name=primary_model_name,
                    secondary_model_name=secondary_model_name,
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
    summary="Extract Text from PDF Document"
)
async def ocr_pdf(
    file: UploadFile = File(..., description="PDF document file"),
    lang: str = Form(config.DEFAULT_LANG, description="Document language codes (e.g. eng+ara+fas)"),
    model: ModelName = Form(config.DEFAULT_MODEL, description="Primary AI vision/OCR model"),
    secondary_model: Optional[ModelName] = Form(None, description="Optional secondary model for dual-model reconciliation"),
    start_page: int = Form(1, gt=0, description="Starting page number (default: 1)"),
    end_page: Optional[int] = Form(None, gt=0, description="Ending page number (optional)"),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="Convert to grayscale with adaptive thresholding"),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="Automatic contrast enhancement (CLAHE)"),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.1, le=5.0, description="Image dimension rescaling multiplier"),
    crop_whitespaces: bool = Form(False, description="Auto-crop document white margins"),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="Enable LLM post-processing and text merging"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Parallel multi-page OCR text extraction from PDF documents.
    
    - Converts and extracts PDF pages concurrently across thread pool workers to minimize latency.
    - Requires authentication via `X-API-Key` or `Authorization: Bearer <token>`.
    """
    primary_model_name = resolve_model_name(model)
    if not primary_model_name or primary_model_name not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"Model '{model.value if isinstance(model, Enum) else model}' is not active on this server.")

    secondary_model_name = resolve_model_name(secondary_model) if secondary_model else None
    if secondary_model_name and secondary_model_name not in ocr_service.models:
        raise HTTPException(status_code=400, detail=f"Secondary model '{secondary_model.value if isinstance(secondary_model, Enum) else secondary_model}' is not active on this server.")

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
                    primary_model_name=primary_model_name,
                    secondary_model_name=secondary_model_name,
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

# --- Health & Status Monitoring Endpoints ---

def sync_probe_service_health(name: str, url: str, timeout: float) -> dict:
    """Probes a remote service's /health endpoint using Python's standard library urllib."""
    start_time = time.perf_counter()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Ghaemieh-OCR-HealthProbe/1.0",
            "Accept": "application/json, text/plain, */*"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            raw_body = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw_body)
            except Exception:
                payload = raw_body

            code = response.getcode()
            is_ok = (200 <= code < 300)
            return {
                "service": name,
                "url": url,
                "status": "online" if is_ok else "error",
                "status_code": code,
                "response": payload,
                "response_time_ms": latency_ms,
                "error": None if is_ok else f"HTTP status {code}"
            }
    except urllib.error.HTTPError as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        raw_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        try:
            payload = json.loads(raw_body)
        except Exception:
            payload = raw_body
        return {
            "service": name,
            "url": url,
            "status": "error",
            "status_code": e.code,
            "response": payload,
            "response_time_ms": latency_ms,
            "error": f"HTTP {e.code}: {e.reason}"
        }
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        is_timeout = (
            isinstance(e, (socket.timeout, TimeoutError))
            or (isinstance(e, urllib.error.URLError) and isinstance(e.reason, (socket.timeout, TimeoutError)))
            or "timed out" in str(e).lower()
        )
        if is_timeout:
            return {
                "service": name,
                "url": url,
                "status": "timeout",
                "status_code": None,
                "response": None,
                "response_time_ms": latency_ms,
                "error": f"Request timed out after {timeout} seconds"
            }
        return {
            "service": name,
            "url": url,
            "status": "offline",
            "status_code": None,
            "response": None,
            "response_time_ms": latency_ms,
            "error": f"Connection error: {e.reason if isinstance(e, urllib.error.URLError) else str(e)}"
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "service": name,
            "url": url,
            "status": "offline",
            "status_code": None,
            "response": None,
            "response_time_ms": latency_ms,
            "error": f"Unexpected error: {str(e)}"
        }

@app.get(
    "/health/status",
    response_model=HealthStatusResponse,
    tags=["System Health"],
    summary="Check Remote LLM & OLM Health Status"
)
@app.get(
    "/status",
    response_model=HealthStatusResponse,
    tags=["System Health"],
    summary="Check Remote LLM & OLM Health Status (Alias)",
    include_in_schema=False
)
@app.get(
    "/health",
    response_model=HealthStatusResponse,
    tags=["System Health"],
    summary="Check Remote LLM & OLM Health Status (Alias)",
    include_in_schema=False
)
async def check_remote_services_status(response: Response):
    """
    Checks connectivity and health of the remote LLM (Gemma/Merger) and OLM (OlmOCR) services.
    
    Probes the `/health` endpoint of each server using Python's standard library and reports
    whether it is online (e.g. `{"status":"ok"}`), timed out, or offline, along with response latency in milliseconds.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    timeout = config.HEALTH_CHECK_TIMEOUT
    llm_url = config.get_llm_health_url()
    olm_url = config.get_olm_health_url()

    llm_task = asyncio.to_thread(sync_probe_service_health, "LLM", llm_url, timeout)
    olm_task = asyncio.to_thread(sync_probe_service_health, "OLM", olm_url, timeout)
    llm_res, olm_res = await asyncio.gather(llm_task, olm_task)

    statuses = [llm_res["status"], olm_res["status"]]
    if all(s == "online" for s in statuses):
        overall_status = "healthy"
    elif any(s == "online" for s in statuses):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "llm": llm_res,
            "olm": olm_res
        }
    }

@app.get(
    "/health/models",
    tags=["System Health"],
    summary="Check Loaded Models Readiness"
)
@app.get(
    "/models",
    tags=["System Health"],
    summary="Check Loaded Models Readiness (Alias)",
    include_in_schema=False
)
def health_models(response: Response):
    """Returns the readiness status of all AI vision and OCR models currently loaded in memory."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    if not ocr_service or not ocr_service.models:
        return {}

    # Deduplicate aliases and return canonical model keys only
    canonical_order = ["gemma4", "olmocr_2b", "tesseract", "docling", "qwen", "varco"]
    result = {}
    seen_instances = set()
    for name in canonical_order:
        if name in ocr_service.models:
            inst = ocr_service.models[name]
            result[name] = "loaded"
            seen_instances.add(id(inst))

    for name, inst in ocr_service.models.items():
        if id(inst) not in seen_instances and name not in result:
            result[name] = "loaded"
            seen_instances.add(id(inst))

    return result

@app.get("/health/config", include_in_schema=False)
def health_config(response: Response):
    """Runtime configuration and active parameters with masked secrets."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

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
