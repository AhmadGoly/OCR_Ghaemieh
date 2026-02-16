from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

class OCRCleanedText(BaseModel):
    text: str = Field(..., description="Cleaned and consolidated OCR text")

class LLMMerger:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def merge(self, ocr_outputs: List[str]) -> str:
        if not ocr_outputs:
            return ""
        if len(ocr_outputs) == 1:
            # If only one output, we still might want to clean it if requested,
            # but usually merging implies at least two.
            # However, the original code used it for single output too (clean_ocr_text).
            pass

        formatted_outputs = "\n".join(
            f"-----\nOCR output {i+1}:\n{txt}\n-----"
            for i, txt in enumerate(ocr_outputs)
        )

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

        completion = self.client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=OCRCleanedText
        )
        return completion.choices[0].message.parsed.text
