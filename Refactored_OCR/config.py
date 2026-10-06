import os

from typing import Union, Sequence

VERSION = "3.6.5"

try:
    from dotenv import load_dotenv
    load_dotenv()
    parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
except ImportError:
    pass

# Database & Authentication Configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://ocr_admin:ocr_secret_password@localhost:5432/ghaemieh_ocr_db"
)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo123")

def get_bool_env(var_names: Union[str, Sequence[str]], default: bool) -> bool:
    """Read a boolean configuration value with support for multiple alias names."""
    if isinstance(var_names, str):
        var_names = [var_names]
    for name in var_names:
        val = os.environ.get(name)
        if val is not None:
            return val.strip().lower() in ("true", "1", "yes")
    return default

# Concurrency & Parallelism Tuning
MAX_CONCURRENT_OCR = int(os.environ.get("MAX_CONCURRENT_OCR", 20))
OCR_THREAD_WORKERS = int(os.environ.get("OCR_THREAD_WORKERS", 24))
PDF_PAGE_WORKERS = int(os.environ.get("PDF_PAGE_WORKERS", 4))
LOCAL_GPU_CONCURRENCY_LIMIT = int(os.environ.get("LOCAL_GPU_CONCURRENCY_LIMIT", 2))
MAX_USER_CONCURRENT_OCR = int(os.environ.get("MAX_USER_CONCURRENT_OCR", 4))
QUEUE_TIMEOUT_SECONDS = float(os.environ.get("QUEUE_TIMEOUT_SECONDS", 90.0))

# Background Book / Batch Task Processing Configuration
TASK_STORAGE_DIR = os.environ.get(
    "TASK_STORAGE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "tasks")
)
TASK_WORKER_CONCURRENCY = int(os.environ.get("TASK_WORKER_CONCURRENCY", 2))
TASK_PAGE_COOLDOWN_SECONDS = float(os.environ.get("TASK_PAGE_COOLDOWN_SECONDS", 1.0))
TASK_MAX_PAGE_RETRIES = int(os.environ.get("TASK_MAX_PAGE_RETRIES", 3))
TASK_DEFAULT_LIST_LIMIT = int(os.environ.get("TASK_DEFAULT_LIST_LIMIT", 10))
TASK_MAX_LIST_LIMIT = int(os.environ.get("TASK_MAX_LIST_LIMIT", 100))
TASK_RETENTION_DAYS = int(os.environ.get("TASK_RETENTION_DAYS", 7))

# API Documentation & Schema Security
DOCS_REQUIRE_AUTH = get_bool_env(["DOCS_REQUIRE_AUTH"], True)
DOCS_REQUIRE_ADMIN = get_bool_env(["DOCS_REQUIRE_ADMIN"], True)

# OCR Model Loading Configuration (supports aliases like LOAD_OLMOCR or OLMOCR)
LOAD_TESSERACT = get_bool_env(["LOAD_TESSERACT", "TESSERACT"], True)
LOAD_DOCLING = get_bool_env(["LOAD_DOCLING", "DOCLING"], False)
LOAD_QWEN = get_bool_env(["LOAD_QWEN", "QWEN"], False)
LOAD_VARCO = get_bool_env(["LOAD_VARCO", "VARCO"], False)
LOAD_OLMOCR_2B = get_bool_env(
    ["LOAD_OLMOCR_2B", "LOAD_OLMOCR", "OLMOCR", "LOAD_OLMOCR_7B", "LOAD_OLM", "OLM"],
    True
)
LOAD_GEMMA4 = get_bool_env(
    ["LOAD_GEMMA4", "LOAD_GEMMA", "GEMMA4", "GEMMA"],
    True
)

# OlmOCR VLM Engine
OLMOCR_LLM_URL_V1 = os.environ.get("OLMOCR_LLM_URL_V1", "http://172.16.20.16:12346/v1")
OLMOCR_MODEL_NAME = os.environ.get("OLMOCR_MODEL_NAME", "allenai/olmocr-2-7b")
OLMOCR_API_KEY = os.environ.get("OLMOCR_API_KEY", "no-key")

# Gemma 4 VLM OCR Engine
GEMMA4_LLM_URL = os.environ.get("GEMMA4_LLM_URL", os.environ.get("DEFAULT_LLM_URL", "http://10.0.38.50:50015/v1"))
GEMMA4_MODEL_NAME = os.environ.get("GEMMA4_MODEL_NAME", os.environ.get("DEFAULT_LLM_MODEL_NAME", "/models/gemma-4-26B-A4B-it-Q8_0.gguf"))
GEMMA4_API_KEY = os.environ.get("GEMMA4_API_KEY", os.environ.get("DEFAULT_LLM_API_KEY", "no-key"))

# Remote Health Probing Configuration
HEALTH_CHECK_TIMEOUT = float(os.environ.get("HEALTH_CHECK_TIMEOUT", 5.0))
LLM_HEALTH_URL = os.environ.get("LLM_HEALTH_URL", "")
OLMOCR_HEALTH_URL = os.environ.get("OLMOCR_HEALTH_URL", "")


def get_llm_health_url() -> str:
    """Derive target health URL for the primary LLM / Gemma server."""
    if LLM_HEALTH_URL:
        return LLM_HEALTH_URL
    base = GEMMA4_LLM_URL or DEFAULT_LLM_URL
    cleaned = base.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    return f"{cleaned}/health"


def get_olm_health_url() -> str:
    """Derive target health URL for the OlmOCR server."""
    if OLMOCR_HEALTH_URL:
        return OLMOCR_HEALTH_URL
    base = OLMOCR_LLM_URL_V1
    cleaned = base.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    return f"{cleaned}/health"

# FastAPI server configuration
FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", 4567))

# Accepted values for endpoints
ACCEPTED_MODELS = ["tesseract", "docling", "qwen", "varco", "olmocr_2b", "gemma4"]
ACCEPTED_LANGUAGES = ["eng", "ara", "fas"]

# Default endpoint parameters
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "eng+ara+fas")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "tesseract")
DEFAULT_PREPROCESS = get_bool_env(["DEFAULT_PREPROCESS", "PREPROCESS"], False)
DEFAULT_CONTRAST = get_bool_env(["DEFAULT_CONTRAST", "CONTRAST"], False)
DEFAULT_SCALE = float(os.environ.get("DEFAULT_SCALE", 1.0))
DEFAULT_USE_LLM = get_bool_env(["DEFAULT_USE_LLM", "USE_LLM"], False)
DEFAULT_LLM_URL = os.environ.get("DEFAULT_LLM_URL", "http://10.0.38.50:50015/v1")
DEFAULT_LLM_MODEL_NAME = os.environ.get("DEFAULT_LLM_MODEL_NAME", "/models/gemma-4-26B-A4B-it-Q8_0.gguf")
DEFAULT_LLM_API_KEY = os.environ.get("DEFAULT_LLM_API_KEY", "no-key")

# Whitespace cropping threshold
CROP_WHITESPACE_THRESHOLD = int(os.environ.get("CROP_WHITESPACE_THRESHOLD", 250))


def mask_secret(value: str) -> str:
    """Mask secret API keys for safe display in logs and public endpoints."""
    if not value or value in ("no-key", "your_dummy_or_real_key", "none"):
        return value or "none"
    if len(value) <= 8:
        return "****"
    return f"{value[:3]}...{value[-3:]}"


def get_config_dict() -> dict:
    """Returns runtime configuration dictionary with masked secrets."""
    return {
        "version": VERSION,
        "server": {
            "port": FASTAPI_PORT,
        },
        "models_enabled": {
            "tesseract": LOAD_TESSERACT,
            "olmocr_2b": LOAD_OLMOCR_2B,
            "gemma4": LOAD_GEMMA4,
            "docling": LOAD_DOCLING,
            "qwen": LOAD_QWEN,
            "varco": LOAD_VARCO,
        },
        "olmocr": {
            "endpoint": OLMOCR_LLM_URL_V1,
            "model_name": OLMOCR_MODEL_NAME,
            "api_key": mask_secret(OLMOCR_API_KEY),
        },
        "gemma4_vlm": {
            "endpoint": GEMMA4_LLM_URL,
            "model_name": GEMMA4_MODEL_NAME,
            "api_key": mask_secret(GEMMA4_API_KEY),
        },
        "llm_merger": {
            "default_use_llm": DEFAULT_USE_LLM,
            "endpoint": DEFAULT_LLM_URL,
            "model_name": DEFAULT_LLM_MODEL_NAME,
            "api_key": mask_secret(DEFAULT_LLM_API_KEY),
        },
        "remote_services_health": {
            "timeout_seconds": HEALTH_CHECK_TIMEOUT,
            "llm_health_url": get_llm_health_url(),
            "olm_health_url": get_olm_health_url(),
        },
        "concurrency": {
            "max_concurrent_ocr": MAX_CONCURRENT_OCR,
            "ocr_thread_workers": OCR_THREAD_WORKERS,
            "pdf_page_workers": PDF_PAGE_WORKERS,
            "local_gpu_concurrency_limit": LOCAL_GPU_CONCURRENCY_LIMIT,
            "max_user_concurrent_ocr": MAX_USER_CONCURRENT_OCR,
            "queue_timeout_seconds": QUEUE_TIMEOUT_SECONDS,
        },
        "docs_security": {
            "require_auth": DOCS_REQUIRE_AUTH,
            "require_admin": DOCS_REQUIRE_ADMIN,
        },
        "task_processing": {
            "worker_concurrency": TASK_WORKER_CONCURRENCY,
            "page_cooldown_seconds": TASK_PAGE_COOLDOWN_SECONDS,
            "max_page_retries": TASK_MAX_PAGE_RETRIES,
            "default_list_limit": TASK_DEFAULT_LIST_LIMIT,
            "max_list_limit": TASK_MAX_LIST_LIMIT,
            "retention_days": TASK_RETENTION_DAYS,
        },
        "defaults": {
            "model": DEFAULT_MODEL,
            "lang": DEFAULT_LANG,
            "preprocess": DEFAULT_PREPROCESS,
            "contrast": DEFAULT_CONTRAST,
            "scale": DEFAULT_SCALE,
            "crop_whitespace_threshold": CROP_WHITESPACE_THRESHOLD,
            "tessdata_prefix": os.environ.get("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata/"),
        },
    }


def print_startup_banner() -> None:
    """Prints a structured banner of all environment & runtime configurations at service startup."""
    cfg = get_config_dict()
    print("=" * 68, flush=True)
    print(f"             GHAEMIEH OCR SERVICE v{cfg['version']}", flush=True)
    print("=" * 68, flush=True)
    print(f" [Server]              Port: {cfg['server']['port']}", flush=True)
    print(f" [Models Enabled]", flush=True)
    print(f"   - Tesseract:        {'ENABLED' if cfg['models_enabled']['tesseract'] else 'DISABLED'}", flush=True)
    print(f"   - OlmOCR 2B (VLM):  {'ENABLED' if cfg['models_enabled']['olmocr_2b'] else 'DISABLED'}", flush=True)
    print(f"   - Gemma 4 (VLM):    {'ENABLED' if cfg['models_enabled']['gemma4'] else 'DISABLED'}", flush=True)
    print(f"   - Docling:          {'ENABLED' if cfg['models_enabled']['docling'] else 'DISABLED'}", flush=True)
    print(f"   - Qwen VL:          {'ENABLED' if cfg['models_enabled']['qwen'] else 'DISABLED'}", flush=True)
    print(f"   - Varco:            {'ENABLED' if cfg['models_enabled']['varco'] else 'DISABLED'}", flush=True)
    print(f" [OlmOCR Engine]", flush=True)
    print(f"   - Endpoint:         {cfg['olmocr']['endpoint']}", flush=True)
    print(f"   - Model:            {cfg['olmocr']['model_name']}", flush=True)
    print(f"   - API Key:          {cfg['olmocr']['api_key']}", flush=True)
    print(f" [Gemma 4 VLM OCR Engine]", flush=True)
    print(f"   - Endpoint:         {cfg['gemma4_vlm']['endpoint']}", flush=True)
    print(f"   - Model:            {cfg['gemma4_vlm']['model_name']}", flush=True)
    print(f"   - API Key:          {cfg['gemma4_vlm']['api_key']}", flush=True)
    print(f" [LLM Merger Engine]", flush=True)
    print(f"   - Active by default:{'YES' if cfg['llm_merger']['default_use_llm'] else 'NO'}", flush=True)
    print(f"   - Endpoint:         {cfg['llm_merger']['endpoint']}", flush=True)
    print(f"   - Model:            {cfg['llm_merger']['model_name']}", flush=True)
    print(f"   - API Key:          {cfg['llm_merger']['api_key']}", flush=True)
    print(f" [Remote Health Check]", flush=True)
    print(f"   - LLM Health:       {cfg['remote_services_health']['llm_health_url']}", flush=True)
    print(f"   - OlmOCR Health:    {cfg['remote_services_health']['olm_health_url']}", flush=True)
    print(f"   - Probe Timeout:    {cfg['remote_services_health']['timeout_seconds']}s", flush=True)
    print(f" [Inference Defaults]", flush=True)
    print(f"   - Primary Model:    {cfg['defaults']['model']}", flush=True)
    print(f"   - Default Lang:     {cfg['defaults']['lang']}", flush=True)
    print(f"   - Preprocess:       {cfg['defaults']['preprocess']}", flush=True)
    print(f"   - Contrast:         {cfg['defaults']['contrast']}", flush=True)
    print(f"   - Rescale Factor:   {cfg['defaults']['scale']}", flush=True)
    print(f"   - Whitespace Crop:  Threshold {cfg['defaults']['crop_whitespace_threshold']}", flush=True)
    print(f"   - Tessdata Prefix:  {cfg['defaults']['tessdata_prefix']}", flush=True)
    print("=" * 68, flush=True)
