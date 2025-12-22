from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

def olm_ocr_text_extraction(
    pil_image,
    llm_url,
    llm_api_key,
    language_list=["fas", "eng"]
):
    client = OpenAI(api_key=llm_api_key, base_url=llm_url)
    
    lang_map = {"fas": "Persian", "eng": "English", "ara": "Arabic"}
    languages = ', '.join([lang_map.get(lang, lang) for lang in language_list])
    
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_b64}"
    
    messages = [
        {
            "role": "system",
            "content": "You are a high-accuracy OCR engine. Output only the extracted text."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Read all text in this image. Languages might be {languages}. Here is the image:\n\n"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="allenai/olmocr-2-7b",
        messages=messages,
        max_tokens=8000
    )
    
    return response.choices[0].message.content
