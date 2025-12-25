import os

# OCR Model Loading Configuration

# Enable or disable loading of the Tesseract OCR model.
LOAD_TESSERACT = True

# Enable or disable loading of the Docling OCR model.
LOAD_DOCLING = False

# Enable or disable loading of the Qwen OCR model.
# This model requires a GPU with at least 8GB of free memory.
LOAD_QWEN = False

# Enable or disable loading of the Varco OCR model.
LOAD_VARCO = False

# Enable or disable loading of the OLM model.
LOAD_OLMOCR_2B = True
OLMOCR_LLM_URL_V1 = "http://172.16.20.16:12346/v1"
OLMOCR_API_KEY = "no-key"

# FastAPI server configuration
FASTAPI_PORT = 4567

# Accepted values for endpoints
ACCEPTED_MODELS = ["tesseract", "docling", "qwen", "varco", "olmocr_2b", "tesseract+olmocr_llm"]
ACCEPTED_LANGUAGES = ["eng", "ara", "fas"]

# Default endpoint parameters
DEFAULT_LANG = "eng+ara+fas"
DEFAULT_MODEL = "tesseract"
DEFAULT_PREPROCESS = False
DEFAULT_CONTRAST = False
DEFAULT_SCALE = 1.0
DEFAULT_USE_LLM = False
DEFAULT_LLM_URL = "http://192.168.159.92:8080/v1"
DEFAULT_LLM_MODEL_NAME = "gemma-3-4b-it-Q8_0"
DEFAULT_LLM_API_KEY = "your_dummy_or_real_key"

# Whitespace cropping threshold
CROP_WHITESPACE_THRESHOLD = int(os.environ.get("CROP_WHITESPACE_THRESHOLD", 250))
