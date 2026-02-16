import os

# OCR Model Loading Configuration
LOAD_TESSERACT = True
LOAD_DOCLING = False
LOAD_QWEN = False
LOAD_VARCO = False
LOAD_OLMOCR_2B = True

OLMOCR_LLM_URL_V1 = "http://172.16.20.16:12346/v1"
OLMOCR_API_KEY = "no-key"

# FastAPI server configuration
FASTAPI_PORT = 4567

# Accepted values for endpoints
ACCEPTED_MODELS = ["tesseract", "docling", "qwen", "varco", "olmocr_2b"]
ACCEPTED_LANGUAGES = ["eng", "ara", "fas"]

# Default endpoint parameters
DEFAULT_LANG = "eng+ara+fas"
DEFAULT_MODEL = "tesseract"
DEFAULT_PREPROCESS = False
DEFAULT_CONTRAST = False
DEFAULT_SCALE = 1.0
DEFAULT_USE_LLM = False
DEFAULT_LLM_URL = "http://172.16.20.16:12347/v1"
DEFAULT_LLM_MODEL_NAME = "gemma-3-27b-it-Q8_0.gguf"
DEFAULT_LLM_API_KEY = "your_dummy_or_real_key"

# Whitespace cropping threshold
CROP_WHITESPACE_THRESHOLD = int(os.environ.get("CROP_WHITESPACE_THRESHOLD", 250))
