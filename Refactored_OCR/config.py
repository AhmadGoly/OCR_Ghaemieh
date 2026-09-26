import os

VERSION = "2.1.0"

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
            "docling": LOAD_DOCLING,
            "qwen": LOAD_QWEN,
            "varco": LOAD_VARCO,
        },
        "olmocr": {
            "endpoint": OLMOCR_LLM_URL_V1,
            "model_name": OLMOCR_MODEL_NAME,
            "api_key": mask_secret(OLMOCR_API_KEY),
        },
        "llm_merger": {
            "default_use_llm": DEFAULT_USE_LLM,
            "endpoint": DEFAULT_LLM_URL,
            "model_name": DEFAULT_LLM_MODEL_NAME,
            "api_key": mask_secret(DEFAULT_LLM_API_KEY),
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
    print("=" * 68)
    print(f"             GHAEMIEH OCR SERVICE v{cfg['version']}")
    print("=" * 68)
    print(f" [Server]              Port: {cfg['server']['port']}")
    print(f" [Models Enabled]")
    print(f"   - Tesseract:        {'ENABLED' if cfg['models_enabled']['tesseract'] else 'DISABLED'}")
    print(f"   - OlmOCR 2B:        {'ENABLED' if cfg['models_enabled']['olmocr_2b'] else 'DISABLED'}")
    print(f"   - Docling:          {'ENABLED' if cfg['models_enabled']['docling'] else 'DISABLED'}")
    print(f"   - Qwen VL:          {'ENABLED' if cfg['models_enabled']['qwen'] else 'DISABLED'}")
    print(f"   - Varco:            {'ENABLED' if cfg['models_enabled']['varco'] else 'DISABLED'}")
    print(f" [OlmOCR Engine]")
    print(f"   - Endpoint:         {cfg['olmocr']['endpoint']}")
    print(f"   - Model:            {cfg['olmocr']['model_name']}")
    print(f"   - API Key:          {cfg['olmocr']['api_key']}")
    print(f" [LLM Merger Engine]")
    print(f"   - Active by default:{'YES' if cfg['llm_merger']['default_use_llm'] else 'NO'}")
    print(f"   - Endpoint:         {cfg['llm_merger']['endpoint']}")
    print(f"   - Model:            {cfg['llm_merger']['model_name']}")
    print(f"   - API Key:          {cfg['llm_merger']['api_key']}")
    print(f" [Inference Defaults]")
    print(f"   - Primary Model:    {cfg['defaults']['model']}")
    print(f"   - Default Lang:     {cfg['defaults']['lang']}")
    print(f"   - Preprocess:       {cfg['defaults']['preprocess']}")
    print(f"   - Contrast:         {cfg['defaults']['contrast']}")
    print(f"   - Rescale Factor:   {cfg['defaults']['scale']}")
    print(f"   - Whitespace Crop:  Threshold {cfg['defaults']['crop_whitespace_threshold']}")
    print(f"   - Tessdata Prefix:  {cfg['defaults']['tessdata_prefix']}")
    print("=" * 68)
