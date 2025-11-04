# OCR FastAPI Application

This directory contains a FastAPI application for performing OCR on images and PDF files using various OCR backends.

## Setup

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Models**:

    Edit the `config.py` file to enable or disable the loading of different OCR models.

3.  **Run the Application**:

    ```bash
    uvicorn main:app --reload
    ```

    The application will be available at `http://127.0.0.1:8000`.

## Endpoints

### `/ocr/image`

This endpoint performs OCR on a single uploaded image file.

*   **Method**: `POST`
*   **Parameters**:
    *   `file`: The image file to process.
    *   `model`: The OCR model to use (`tesseract`, `docling`, `qwen`, `varco`).
    *   `preprocess`: `true` or `false` (default: `false`).
    *   `contrast`: `true` or `false` (default: `false`).
    *   `scale`: A float between 0.0 and 1.0 (default: `1.0`).
    *   `use_llm`: `true` or `false` (default: `true`).
    *   `llm_url`: The URL of the LLM API.
    *   `llm_model_name`: The name of the LLM model.
    *   `llm_api_key`: The API key for the LLM.
*   **Returns**: A JSON object with the OCR text and the processing duration.

### `/ocr/pdf`

This endpoint performs OCR on an uploaded PDF file.

*   **Method**: `POST`
*   **Parameters**:
    *   `file`: The PDF file to process.
    *   `model`: The OCR model to use (`tesseract`, `docling`, `qwen`, `varco`).
    *   `start_page`: The first page to process (default: `1`).
    *   `end_page`: The last page to process (default: `None`).
    *   `preprocess`: `true` or `false` (default: `false`).
    *   `contrast`: `true` or `false` (default: `false`).
    *   `scale`: A float between 0.0 and 1.0 (default: `1.0`).
    *   `use_llm`: `true` or `false` (default: `true`).
    *   `llm_url`: The URL of the LLM API.
    *   `llm_model_name`: The name of the LLM model.
    *   `llm_api_key`: The API key for the LLM.
*   **Returns**: A JSON object with the OCR results for each page.

### `/health/models`

This endpoint returns the status of the loaded OCR models.

*   **Method**: `GET`
*   **Returns**: A JSON object with the loaded models and their status.

### `/health/gpu`

This endpoint returns the status of the GPU.

*   **Method**: `GET`
*   **Returns**: A JSON object with the GPU status and information.
