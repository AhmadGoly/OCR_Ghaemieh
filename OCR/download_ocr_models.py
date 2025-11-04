import os

langs = ["eng", "ara", "fas"]
tessdata_path = "/usr/share/tesseract-ocr/4.00/tessdata/"

def download_models():
    """
    Downloads Tesseract language models if they don't already exist.
    """
    os.makedirs(tessdata_path, exist_ok=True)
    for lang in langs:
        output_file = os.path.join(tessdata_path, f"{lang}.traineddata")
        if not os.path.exists(output_file):
            print(f"Downloading {lang}.traineddata...")
            url = f"https://github.com/tesseract-ocr/tessdata/raw/main/{lang}.traineddata"
            os.system(f"wget -O {output_file} {url}")
        else:
            print(f"{lang}.traineddata already exists. Skipping.")

if __name__ == "__main__":
    download_models()
