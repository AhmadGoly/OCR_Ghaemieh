from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import OpenAI

class OCRCleanedText(BaseModel):
    text: str = Field(..., description="Cleaned and consolidated OCR text")

# Specialized Prompt Registry with user-facing titles
PROMPTS: Dict[str, Dict[str, str]] = {
    "classical": {
        "title": "متون کهن و کتب حوزوی (اعراب کامل، شعر و پاورقی)",
        "description": "مخصوص کتب کهن و اسلامی؛ حفظ کامل اعراب و حرکات، شعر دو مصراعی روبه‌روی هم با ***، پاورقی‌ها با ---، و عدم تحریف واژگان.",
        "system_prompt": (
            "تو یک ویراستار حرفه‌ای متون کهن، کتاب‌ها و نسخه‌های دینی، تاریخی و حوزوی (فارسی و عربی کلاسیک) هستی. "
            "وظیفه تو استخراج دقیق متن از خروجی‌های مدل‌های OCR و بازنویسی آن با رعایت کامل استانداردهای سخت‌گیرانه زیر است:\n\n"
            "📌 قوانین اصلی:\n"
            "۱. متن باید دقیقاً مطابق محتوای سند باشد؛ بدون اضافه یا حذف حتی یک کلمه یا حرف. هرگز متنی از خودت اضافه نکن و کلمه‌ای را تغییر نده.\n"
            "۲. اعراب‌گذاری کامل و دقیق (فتحه، ضمه، کسره، تنوین‌ها، شَدّه و سکون) باید حفظ یا اصلاح دقیق شود.\n"
            "۳. هیچ غلط املایی یا نحوی نباید در خروجی وجود داشته باشد.\n"
            "۴. تمامی اعداد به صورت صحیح و در جای خود قرار گیرند و نامرتب نباشند.\n"
            "۵. برای نام پیامبران، امامان و شخصیت‌های مقدس، عبارات احترام کتاب عیناً رعایت شود (علیه السلام، علیهم السلام، صلی الله علیه وآله، سلام الله علیها، رحمه الله علیه، قدّس سرّه، دام ظله).\n"
            "۶. ساختار صفحه کاملاً رعایت شود:\n"
            "   - تیترها با علامت زیر در ابتدای خط مشخص شوند:\n"
            "     ## عنوان\n"
            "   - شعرها باید در دو مصراع روبه‌روی هم نوشته شده و بین دو مصراع از علامت *** استفاده شود:\n"
            "     شَعَرُ أَوَّلُ شَطْرٍ *** شَعَرُ ثانِي شَطْرٍ\n"
            "   - پاورقی‌ها باید در انتهای متن صفحه و جدا از متن اصلی به این شکل آورده شوند:\n"
            "     ---\n"
            "     (1) متن پاورقی دقیقاً مطابق سند\n"
            "۷. نشانه‌گذاری و پاراگراف‌بندی دقیق مطابق کتاب رعایت شود و از شکستن بی‌دلیل خطوط پرهیز شود.\n\n"
            "پاسخ را منحصراً در قالب ساختار JSON ارسال کن: {\"text\": \"...\"}."
        )
    },
    "general": {
        "title": "متون عمومی و اداری (استاندارد معاصر)",
        "description": "مناسب اسناد معاصر، مقالات و نامه‌ها؛ پاک‌سازی خطاهای تایپی OCR، پیوسته‌سازی سطور و پاراگراف‌بندی روان.",
        "system_prompt": (
            "تو یک ویراستار حرفه‌ای متون عمومی، مقالات و اسناد معاصر به زبان‌های فارسی، عربی و انگلیسی هستی. "
            "کاربر نتایج مدل‌های مختلف OCR برای یک صفحه سند را برایت ارسال می‌کند. "
            "وظیفه تو پاک‌سازی خطاهای OCR، اصلاح املای کلمات آسیب‌دیده، پیوسته‌سازی سطور به پاراگراف‌های منسجم و ارائه نسخه‌ای تمیز و بی‌نقص است.\n\n"
            "📌 قوانین اصلی:\n"
            "۱. از خروجی اصلی به عنوان قالب پایه استفاده کن و تنها غلط‌های واضح تایپی و کاراکترهای آسیب‌دیده OCR را بر اساس سایر خروجی‌ها تصحیح کن.\n"
            "۲. سطور متن را پیوسته و پاراگراف‌بندی را طبیعی کن (از شکستن خطوط در میان جملات خودداری کن).\n"
            "۳. هیچ جمله یا مفهوم جدیدی به متن اضافه نکن و متن را بازنویسی سلیقه‌ای نکن.\n"
            "۴. علائم نگارشی (نقطه، کاما، گیومه، پرانتز) را به شکل استاندارد اصلاح کن.\n\n"
            "پاسخ را منحصراً در قالب ساختار JSON ارسال کن: {\"text\": \"...\"}."
        )
    }
}

class LLMMerger:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @classmethod
    def get_prompt_options(cls) -> List[Dict[str, str]]:
        """Return user-facing prompt titles and keys for UI selection."""
        return [
            {"id": k, "title": v["title"], "description": v["description"]}
            for k, v in PROMPTS.items()
        ]

    def merge(self, ocr_outputs: List[str], prompt_mode: str = "classical") -> str:
        if not ocr_outputs:
            return ""

        formatted_outputs = "\n".join(
            f"-----\nOCR output {i+1}:\n{txt}\n-----"
            for i, txt in enumerate(ocr_outputs)
        )

        selected_cfg = PROMPTS.get(prompt_mode, PROMPTS["classical"])
        system_instruction = selected_cfg["system_prompt"]

        messages = [
            {
                "role": "system",
                "content": system_instruction
            },
            {"role": "user", "content": formatted_outputs}
        ]

        completion = self.client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=OCRCleanedText
        )
        return completion.choices[0].message.parsed.text

    def ping(self) -> dict:
        """Probe remote LLM Merger endpoint health."""
        import urllib.request
        import urllib.error
        import time

        url = f"{self.base_url.rstrip('/')}/models"
        start_t = time.perf_counter()
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Ghaemieh-OCR-Probe/1.0"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                elapsed = round((time.perf_counter() - start_t) * 1000, 2)
                return {
                    "status": "online",
                    "status_code": resp.getcode(),
                    "response_time_ms": elapsed,
                    "model_name": self.model_name
                }
        except urllib.error.HTTPError as e:
            elapsed = round((time.perf_counter() - start_t) * 1000, 2)
            is_reachable = e.code in (200, 401, 403, 404, 405)
            return {
                "status": "online" if is_reachable else "error",
                "status_code": e.code,
                "response_time_ms": elapsed,
                "error": f"HTTP {e.code}: {e.reason}",
                "model_name": self.model_name
            }
        except Exception as e:
            elapsed = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "offline",
                "response_time_ms": elapsed,
                "error": str(e),
                "model_name": self.model_name
            }
