import torch
from PIL import Image
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from .base import BaseOCRModel

class VarcoModel(BaseOCRModel):
    def __init__(self, model_name="NCSOFT/VARCO-VISION-2.0-1.7B-OCR", max_tokens=1024):
        if not torch.cuda.is_available():
            raise RuntimeError("Varco OCR requires a GPU, none detected.")

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def process(self, image: Image.Image, lang: str = None) -> str:
        torch.cuda.empty_cache()
        w, h = image.size
        target_size = 2304
        current_image = image
        if max(w, h) < target_size:
            scaling_factor = target_size / max(w, h)
            new_w = int(w * scaling_factor)
            new_h = int(h * scaling_factor)
            current_image = current_image.resize((new_w, new_h), Image.LANCZOS)

        conversation = [{"role":"user","content":[{"type":"image","image":current_image},{"type":"text","text":"<ocr>"}]}]

        attempt = 0
        max_attempts = 5
        output = ""
        while attempt < max_attempts:
            try:
                inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                inputs = inputs.to(self.model.device, torch.float16)
                generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
                generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
                output = self.processor.decode(generate_ids_trimmed[0], skip_special_tokens=False)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    w, h = current_image.size
                    new_w, new_h = max(1, int(w * 0.8)), max(1, int(h * 0.8))
                    current_image = current_image.resize((new_w, new_h), Image.LANCZOS)
                    conversation[0]["content"][0]["image"] = current_image
                    attempt += 1
                else:
                    raise e
        else:
            raise RuntimeError("Varco failed after multiple memory reduction attempts.")

        torch.cuda.empty_cache()
        return output
