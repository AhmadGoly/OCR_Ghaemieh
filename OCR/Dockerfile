FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
software-properties-common \
wget \
curl \
build-essential \
libssl-dev \
zlib1g-dev \
libbz2-dev \
libreadline-dev \
libsqlite3-dev \
libffi-dev \
liblzma-dev \
tesseract-ocr \
poppler-utils \
libgl1 \
libglx-mesa0 \
&& rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
python3.10 python3.10-venv python3.10-dev python3-pip \
&& rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
&& update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python -m pip install --upgrade pip

RUN python -m pip install --no-cache-dir fastapi uvicorn pdf2image python-bidi arabic-reshaper
RUN python -m pip install --no-cache-dir transformers
RUN python -m pip install --no-cache-dir pytesseract
RUN python -m pip install --no-cache-dir docling
RUN python -m pip install --no-cache-dir qwen_vl_utils
RUN python -m pip install --no-cache-dir \
    "accelerate>=0.26.0" \
    peft \
    bitsandbytes \
    python-multipart
RUN python -m pip install --no-cache-dir openai
RUN python -m pip install --no-cache-dir imutils matplotlib

COPY ./download_ocr_models.py .
RUN python download_ocr_models.py && rm download_ocr_models.py
