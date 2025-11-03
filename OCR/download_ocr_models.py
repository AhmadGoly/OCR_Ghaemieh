import os
langs = ["eng", "ara", "fas"]
tessdata_path = "/usr/share/tesseract-ocr/4.00/tessdata/"
os.makedirs(tessdata_path, exist_ok=True)
for lang in langs:
    url = f"https://github.com/tesseract-ocr/tessdata/raw/main/{lang}.traineddata"
    output_file = os.path.join(tessdata_path, f"{lang}.traineddata")
    os.system(f"wget -O {output_file} {url}")