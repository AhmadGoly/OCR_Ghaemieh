FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY ./download_ocr_models.py .
RUN python download_ocr_models.py && rm download_ocr_models.py

RUN pip install --no-cache-dir fastapi uvicorn pdf2image python-bidi arabic-reshaper
RUN pip install --no-cache-dir \
        "accelerate>=0.26.0" \
        peft \
        bitsandbytes \
        python-multipart 
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir pytesseract
RUN pip install --no-cache-dir docling
RUN pip install --no-cache-dir qwen_vl_utils
RUN pip install openai