print("Importing standard libraries for OCR...")
import base64
import io
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

print("Importing ML/AI libraries for OCR...")
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration
from openai import OpenAI
from pydantic import BaseModel, Field
from olm.OlmOCR import olm_ocr_text_extraction

print("Importing image processing libraries for OCR...")
import cv2 as cv
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import matplotlib.pyplot as plt
import imutils

print("Importing text and utility libraries for OCR...")
import arabic_reshaper
from bidi.algorithm import get_display
from qwen_vl_utils import process_vision_info
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import config
import re

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class PDFOCRProcessor:
    def __init__(self, lang='eng+ara+fas', ocr_backend='tesseract',
                 qwen_model_name="NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
                 qwen_max_tokens=2000,
                 varco_model_name="NCSOFT/VARCO-VISION-2.0-1.7B-OCR",
                 varco_max_tokens=1024):
        self.postprocess_llm = OpenAI(api_key=config.DEFAULT_LLM_API_KEY, base_url=config.DEFAULT_LLM_URL)
        self.lang = lang
        self.ocr_backend = ocr_backend
        self.qwen_model_name = qwen_model_name
        self.qwen_max_tokens = qwen_max_tokens
        self.varco_model_name = varco_model_name
        self.varco_max_tokens = varco_max_tokens
        self.qwen_model = None
        self.qwen_processor = None
        self.varco_model = None
        self.varco_processor = None
        self.olm_ocr_text_extraction = olm_ocr_text_extraction
        log("Initializing PDFOCRProcessor...")

        if self.ocr_backend == 'qwen':
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen OCR requires a GPU, but none was detected.")

            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
                free_memory_mb = int(result.stdout.strip())
                free_memory_gb = free_memory_mb / 1024
                if free_memory_gb < 8:
                    raise RuntimeError(f"Qwen OCR requires at least 8GB of free GPU memory, but only {free_memory_gb:.2f}GB is available.")
            except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
                raise RuntimeError(f"Failed to check GPU memory using nvidia-smi: {e}")

            log("Loading Qwen model and processor on GPU...")
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.qwen_model_name, torch_dtype="auto", device_map="auto")
            self.qwen_processor = AutoProcessor.from_pretrained(self.qwen_model_name)
            log("Qwen model loaded on GPU.")

        if self.ocr_backend == 'varco':
            if not torch.cuda.is_available():
                raise RuntimeError("Varco OCR requires a GPU, none detected.")
            log("Loading Varco model and processor on GPU...")
            self.varco_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.varco_model_name, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
            self.varco_processor = AutoProcessor.from_pretrained(self.varco_model_name)
            log("Varco model loaded on GPU.")

        if self.ocr_backend == 'olmocr_2b':
            log("Loading OlmOCR 2B model...")
            self.olm_ocr_text_extraction = olm_ocr_text_extraction
            log("OlmOCR 2B model loaded.")

    def _to_gray(self, pil_image):
        img = np.array(pil_image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return img

    def _image_to_base64(self, pil_image):
        """Converts a PIL image to a base64 encoded string."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def crop_whitespaces(self, pil_image, threshold=config.CROP_WHITESPACE_THRESHOLD):
        log("Cropping whitespaces...")
        img = np.array(pil_image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        _, thresh = cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)
        coords = cv.findNonZero(thresh)
        if coords is None:
            log("No content found to crop to, returning original image.")
            return pil_image

        x, y, w, h = cv.boundingRect(coords)
        cropped = img[y:y+h, x:x+w]
        log("Whitespace cropping done.")
        return Image.fromarray(cropped)

    def preprocess_page(self, pil_image):
        log("Preprocessing page...")
        img = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 11)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
        processed = cv.dilate(opening, kernel, iterations=1)
        log("Preprocessing done.")
        return Image.fromarray(processed)

    def enhance_contrast(self, pil_image):
        log("Enhancing contrast...")
        gray = self._to_gray(pil_image)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        log("Contrast enhancement done.")
        return Image.fromarray(enhanced)

    def rescale_image(self, pil_image, scale=1.0):
        if scale == 1.0:
            return pil_image
        log(f"Rescaling image by a factor of {scale}...")
        width, height = pil_image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = pil_image.resize((new_width, new_height), Image.LANCZOS)
        log("Rescaling done.")
        return img_resized

    def _fit_image_to_memory(self, pil_image, max_pixels=2_000_000):
        w, h = pil_image.size
        orig_w, orig_h = w, h
        while w * h > max_pixels:
            w = max(1, w // 2)
            h = max(1, h // 2)
            log(f"Warning: image too large ({orig_w}x{orig_h}), scaling down to ({w}x{h})")
            pil_image = pil_image.resize((w, h), Image.LANCZOS)
        return pil_image

    def _ocr_tesseract(self, pil_image, lang=None):
        log("Running Tesseract OCR...")
        use_lang = lang if lang else self.lang
        result = pytesseract.image_to_string(pil_image, lang=use_lang)
        result = re.sub(r'(?<!\n)\n(?!\n)', ' ', result)
        #result = re.sub(r'(?<!\.)\s*\n+\s*', ' ', result)
        #result = re.sub(r'\n+', '\n', result)
        log("Tesseract OCR done.")
        return result

    def _ocr_qwen(self, pil_image):
        log("Running Qwen OCR...")
        torch.cuda.empty_cache()
        fd, src = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_image.save(src)
        prompt = "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."
        messages = [{"role": "user", "content":[{"type":"image","image":f"file://{src}"},{"type":"text","text":prompt}]}]
        log("Applying chat template...")
        text_template = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        log("Processing vision info...")
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(text=[text_template], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            try:
                log(f"Generating output from Qwen model, attempt {attempt+1}...")
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=self.qwen_max_tokens)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log(f"CUDA OOM detected for Qwen, clearing cache and reducing image size...")
                    torch.cuda.empty_cache()
                    w, h = pil_image.size
                    new_w, new_h = max(1, int(w * 0.8)), max(1, int(h * 0.8))
                    log(f"Reducing image size from ({w},{h}) to ({new_w},{new_h}) and retrying...")
                    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                    pil_image.save(src)
                    messages[0]["content"][0]["image"] = f"file://{src}"
                    text_template = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.qwen_processor(text=[text_template], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                    inputs = inputs.to("cuda")
                    attempt += 1
                else:
                    try: os.remove(src)
                    except: pass
                    raise e
        else:
            try: os.remove(src)
            except: pass
            raise RuntimeError("Qwen failed after multiple memory reduction attempts.")
        try: os.remove(src)
        except: pass
        torch.cuda.empty_cache()
        log("Qwen OCR done.")
        return output_text

    def _ocr_varco(self, pil_image):
        log("Running Varco OCR...")
        torch.cuda.empty_cache()
        w, h = pil_image.size
        target_size = 2304
        if max(w, h) < target_size:
            scaling_factor = target_size / max(w, h)
            new_w = int(w * scaling_factor)
            new_h = int(h * scaling_factor)
            log(f"Upscaling image from ({w},{h}) to ({new_w},{new_h}) for Varco")
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        conversation = [{"role":"user","content":[{"type":"image","image":pil_image},{"type":"text","text":"<ocr>"}]}]
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            try:
                log(f"Applying Varco chat template, attempt {attempt+1}...")
                inputs = self.varco_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                inputs = inputs.to(self.varco_model.device, torch.float16)
                log(f"Generating output from Varco model, attempt {attempt+1}...")
                generate_ids = self.varco_model.generate(**inputs, max_new_tokens=self.varco_max_tokens)
                generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
                output = self.varco_processor.decode(generate_ids_trimmed[0], skip_special_tokens=False)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log(f"CUDA OOM detected for Varco, clearing cache and reducing image size...")
                    torch.cuda.empty_cache()
                    w, h = pil_image.size
                    new_w, new_h = max(1, int(w * 0.8)), max(1, int(h * 0.8))
                    log(f"Reducing image size from ({w},{h}) to ({new_w},{new_h}) and retrying...")
                    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                    conversation[0]["content"][0]["image"] = pil_image
                    attempt += 1
                else:
                    raise e
        else:
            raise RuntimeError("Varco failed after multiple memory reduction attempts.")
        torch.cuda.empty_cache()
        log("Varco OCR done.")
        return output

    def _ocr_docling(self, pil_image, lang=None):
        log("Running Docling OCR...")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            pil_image.save(tmp_path, format="PNG")
        use_langs = lang if lang else ["eng", "ara", "fas"]
        log(f"Languages are: {use_langs}")
        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=use_langs)
        pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
        converter = DocumentConverter(
            format_options={InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        doc = converter.convert(tmp_path).document
        try:
            tmp_path.unlink()
        except:
            pass
        log("Docling OCR done.")
        return doc.export_to_markdown()

    def _ocr_olmocr_2b(self, pil_image, lang_list):
        log("Running OlmOCR 2B OCR...")
        text = self.olm_ocr_text_extraction(pil_image,config.OLMOCR_LLM_URL_V1,config.OLMOCR_API_KEY,lang_list)
        log("OlmOCR 2B OCR done.")
        return text
    
    def _ocr_olmocr_tesseract_llm(self, pil_image, lang_list, lang):
        log("Running OlmOCR+Tesseract OCR...")
        log("Step 1: OLMOCR is processing...")
        llm_start = time.time()
        text_1 = self.olm_ocr_text_extraction(pil_image,config.OLMOCR_LLM_URL_V1,config.OLMOCR_API_KEY,lang_list)
        llm_duration = time.time() - llm_start
        log(f"OlmOCR 2B OCR completed in {llm_duration:.2f} seconds.")
        log("Step 2: Tesseract OCR is processing...")
        llm_start = time.time()
        use_lang = lang if lang else self.lang
        result = pytesseract.image_to_string(pil_image, lang=use_lang)
        text_2 = re.sub(r'(?<!\n)\n(?!\n)', ' ', result)
        llm_duration = time.time() - llm_start
        log(f"Tesseract OCR completed in {llm_duration:.2f} seconds.")
        log("Step 3: LLM Postprocessing...")
        llm_start = time.time()
        final_text = self.clean_ocr_text(
            self.postprocess_llm, config.DEFAULT_LLM_MODEL_NAME, *[text_1,text_2])
        llm_duration = time.time() - llm_start
        log(f"LLM rewriting completed in {llm_duration:.2f} seconds")
        return final_text

    def ocr_image(self, pil_image, lang=None):
        if self.ocr_backend == 'qwen':
            return self._ocr_qwen(pil_image)
        if self.ocr_backend == 'varco':
            return self._ocr_varco(pil_image)
        

        effective_lang = lang if lang is not None else self.lang
        lang_list = effective_lang.split('+') if effective_lang else None

        if self.ocr_backend == 'docling':
            log(f"Passed languages are: {lang_list}")
            return self._ocr_docling(pil_image, lang_list)
        if self.ocr_backend == 'olmocr_2b':
            log(f"Passed languages are: {lang_list}")
            return self._ocr_olmocr_2b(pil_image, lang_list)
        if self.ocr_backend == 'olmocr+tesseract+llm':
            log(f"Passed languages are: {lang_list}")
            return self._ocr_olmocr_tesseract_llm(pil_image, lang_list, effective_lang)

        return self._ocr_tesseract(pil_image, effective_lang)

    def clean_ocr_text(self, llm_client, llm_model_name, *ocr_outputs: str):
        if not llm_client or not llm_model_name:
            log("LLM not initialized, cannot rewrite OCR text.")
            return ocr_outputs[0] if ocr_outputs else ""
        formatted_outputs = "\n".join(
            f"-----\nOCR output {i+1}:\n{txt}\n-----" 
            for i, txt in enumerate(ocr_outputs)
        )
        class OCRCleanedText(BaseModel):
            text: str = Field(..., description="Cleaned and consolidated OCR text")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Persian+English+Arabic text rewriter. User will send you the results of multi OCR models for only one page. in other terms, the text of models are very similar. "
                    "Your job is to write only one cleaned version of OCR.\n\n"
                    "You will get OCR outputs separated like this (all for the same document page, but from different models):\n"
                    "-----\nOCR output 1:\n...\n-----\nOCR output 2:\n...\n-----\n"
                    "Use OCR 1 as the main template. Rewrite it to fix wrong words and obvious OCR mistakes. "
                    "Use other outputs only if they suggest a better word for corrupted parts in OCR 1. "
                    "Do not add extra words. Avoid repeating content. Focus on a good meaning on sentences."
                    "Respond only in JSON: {\"text\": \"...\"}."
                )
            },
            {"role": "user", "content": formatted_outputs}
        ]
        completion = llm_client.chat.completions.parse(
            model=llm_model_name,
            messages=messages,
            response_format=OCRCleanedText
        )
        return completion.choices[0].message.parsed.text
    
    def visualize(self, image, text, title):
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.subplot(1,2,2)
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        plt.text(0, 1, bidi_text, fontsize=10, va="top", wrap=True)
        plt.title("OCR Text")
        plt.axis("off")
        plt.show()

    def process_demo(self, pdf_path, start_page=1, end_page=None, preprocess=False, contrast=False, scale=1.0, crop_whitespaces=False, show=True, lang=None, rewrite_llm=False):
        log("Starting demo processing...")
        images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page)
        text = {}
        for i, image in enumerate(images, start=start_page):
            log(f"Processing page {i}...")
            img = image
            if preprocess: img = self.preprocess_page(img)
            if contrast: img = self.enhance_contrast(img)
            if scale != 1.0: img = self.rescale_image(img, scale)
            if crop_whitespaces: img = self.crop_whitespaces(img)

            start_t = time.time()
            page_text = self.ocr_image(img, lang)
            ocr_duration = time.time() - start_t
            log(f"OCR completed in {ocr_duration:.2f} seconds")

            text[i] = page_text

            if show:
                self.visualize(img, page_text, f"Page {i}")

            log(f"Page {i} OCR + LLM done.")

        log("Demo OCR completed.")
        return text


    def process(self, pdf_path, start_page=1, end_page=None, preprocess=False, contrast=False, scale=1.0, crop_whitespaces=False, lang=None, rewrite_llm=False, llm_client=None, llm_model_name=None):
        log("Starting batch processing...")
        images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page)
        results = []

        for i, image in enumerate(images, start=start_page):
            log(f"Processing page {i}...")
            img = image
            if preprocess: img = self.preprocess_page(img)
            if contrast: img = self.enhance_contrast(img)
            if scale != 1.0: img = self.rescale_image(img, scale)
            if crop_whitespaces: img = self.crop_whitespaces(img)

            start_t = time.time()
            page_text = self.ocr_image(img, lang)
            ocr_duration = time.time() - start_t
            log(f"OCR completed in {ocr_duration:.2f} seconds")

            llm_duration = -1
            if rewrite_llm and llm_client:
                llm_start = time.time()
                page_text = self.clean_ocr_text(llm_client, llm_model_name, page_text)
                llm_duration = time.time() - llm_start
                log(f"LLM rewriting completed in {llm_duration:.2f} seconds")

            results.append({
                "page": i,
                "text": page_text,
                "ocr_model": self.ocr_backend,
                "ocr_duration": ocr_duration,
                "llm_model": llm_model_name if rewrite_llm and llm_client else None,
                "llm_duration": llm_duration,
            })
            log(f"Page {i} OCR + LLM done.")

        log("Batch processing completed.")
        return results

    def process_image(self, image, preprocess=False, contrast=False, scale=1.0, crop_whitespaces=False, lang=None, rewrite_llm=False, llm_client=None, llm_model_name=None):
        log("Starting image processing...")
        original_image = image.copy()
        processed_image = image

        if preprocess:
            processed_image = self.preprocess_page(processed_image)
        if contrast:
            processed_image = self.enhance_contrast(processed_image)
        if scale != 1.0:
            processed_image = self.rescale_image(processed_image, scale)
        if crop_whitespaces:
            processed_image = self.crop_whitespaces(processed_image)

        start_t = time.time()
        page_text = self.ocr_image(processed_image, lang)
        ocr_duration = time.time() - start_t
        log(f"OCR completed in {ocr_duration:.2f} seconds")

        llm_duration = -1
        if rewrite_llm and llm_client:
            if self.ocr_backend == "olmocr+tesseract+llm":
                log("Skipping LLM Postprocessing for OLMOCR+Tesseract+LLM because it's already included.")
            else:
                llm_start = time.time()
                page_text = self.clean_ocr_text(
                    llm_client, llm_model_name, page_text)
                llm_duration = time.time() - llm_start
                log(f"LLM rewriting completed in {llm_duration:.2f} seconds")

        # Convert images to base64 for JSON response
        original_image_b64 = self._image_to_base64(original_image)
        processed_image_b64 = self._image_to_base64(processed_image)

        log("Image processing completed.")
        return {
            "text": page_text,
            "ocr_model": self.ocr_backend,
            "ocr_duration": ocr_duration,
            "llm_model": llm_model_name if rewrite_llm and llm_client else None,
            "llm_duration": llm_duration,
            "original_image": original_image_b64,
            "processed_image": processed_image_b64,
        }
