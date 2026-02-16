import torch
import tempfile
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base import BaseOCRModel

class QwenModel(BaseOCRModel):
    def __init__(self, model_name="NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct", max_tokens=2000):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen OCR requires a GPU, but none was detected.")

        # Check GPU memory
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
            free_memory_mb = int(result.stdout.strip())
            free_memory_gb = free_memory_mb / 1024
            if free_memory_gb < 8:
                 print(f"Warning: Qwen OCR requires at least 8GB of free GPU memory, but only {free_memory_gb:.2f}GB is available.")
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
            pass

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def process(self, image: Image.Image, lang: str = None) -> str:
        torch.cuda.empty_cache()
        fd, src = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        image.save(src)

        prompt = "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."
        messages = [{"role": "user", "content":[{"type":"image","image":f"file://{src}"},{"type":"text","text":prompt}]}]

        text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text_template], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        attempt = 0
        max_attempts = 5
        output_text = ""
        current_image = image

        while attempt < max_attempts:
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    w, h = current_image.size
                    new_w, new_h = max(1, int(w * 0.8)), max(1, int(h * 0.8))
                    current_image = current_image.resize((new_w, new_h), Image.LANCZOS)
                    current_image.save(src)
                    messages[0]["content"][0]["image"] = f"file://{src}"
                    text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(text=[text_template], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                    inputs = inputs.to("cuda")
                    attempt += 1
                else:
                    if os.path.exists(src): os.remove(src)
                    raise e
        else:
            if os.path.exists(src): os.remove(src)
            raise RuntimeError("Qwen failed after multiple memory reduction attempts.")

        if os.path.exists(src): os.remove(src)
        torch.cuda.empty_cache()
        return output_text
