import os

VERSION = "2.0.2"

try:
    from dotenv import load_dotenv
    load_dotenv()
    parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
except ImportError:
    pass

def get_bool_env(var_name: str, default: bool) -> bool:
    val = os.environ.get(var_name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")

# OCR Model Loading Configuration
LOAD_TESSERACT = get_bool_env("LOAD_TESSERACT", True)
LOAD_DOCLING = get_bool_env("LOAD_DOCLING", False)
LOAD_QWEN = get_bool_env("LOAD_QWEN", False)
LOAD_VARCO = get_bool_env("LOAD_VARCO", False)
LOAD_OLMOCR_2B = get_bool_env("LOAD_OLMOCR_2B", True)

OLMOCR_LLM_URL_V1 = os.environ.get("OLMOCR_LLM_URL_V1", "http://172.16.20.16:12346/v1")
OLMOCR_MODEL_NAME = os.environ.get("OLMOCR_MODEL_NAME", "allenai/olmocr-2-7b")
OLMOCR_API_KEY = os.environ.get("OLMOCR_API_KEY", "no-key")

# FastAPI server configuration
FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", 4567))

# Accepted values for endpoints
ACCEPTED_MODELS = ["tesseract", "docling", "qwen", "varco", "olmocr_2b"]
ACCEPTED_LANGUAGES = ["eng", "ara", "fas"]

# Default endpoint parameters
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "eng+ara+fas")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "tesseract")
DEFAULT_PREPROCESS = get_bool_env("DEFAULT_PREPROCESS", False)
DEFAULT_CONTRAST = get_bool_env("DEFAULT_CONTRAST", False)
DEFAULT_SCALE = float(os.environ.get("DEFAULT_SCALE", 1.0))
DEFAULT_USE_LLM = get_bool_env("DEFAULT_USE_LLM", False)
DEFAULT_LLM_URL = os.environ.get("DEFAULT_LLM_URL", "http://172.16.20.16:12347/v1")
DEFAULT_LLM_MODEL_NAME = os.environ.get("DEFAULT_LLM_MODEL_NAME", "gemma-3-27b-it-Q8_0.gguf")
DEFAULT_LLM_API_KEY = os.environ.get("DEFAULT_LLM_API_KEY", "your_dummy_or_real_key")

# Whitespace cropping threshold
CROP_WHITESPACE_THRESHOLD = int(os.environ.get("CROP_WHITESPACE_THRESHOLD", 250))
