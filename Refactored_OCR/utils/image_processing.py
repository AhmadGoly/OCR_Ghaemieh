import cv2 as cv
import numpy as np
import io
import base64
from PIL import Image

class ImageProcessor:
    @staticmethod
    def to_gray(pil_image):
        img = np.array(pil_image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return img

    @staticmethod
    def crop_whitespaces(pil_image, threshold=250):
        img = np.array(pil_image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        else:
            gray = img

        _, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
        coords = cv.findNonZero(thresh)
        if coords is None:
            return pil_image

        x, y, w, h = cv.boundingRect(coords)
        # Crop the original image (not just the gray one)
        if len(img.shape) == 3:
            cropped = img[y:y+h, x:x+w, :]
        else:
            cropped = img[y:y+h, x:x+w]
        return Image.fromarray(cropped)

    @staticmethod
    def preprocess_page(pil_image):
        # Convert to BGR for OpenCV
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 3:
                img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
            elif img_array.shape[2] == 4:
                img = cv.cvtColor(img_array, cv.COLOR_RGBA2BGR)
        else:
            img = cv.cvtColor(img_array, cv.COLOR_GRAY2BGR)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 11)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
        processed = cv.dilate(opening, kernel, iterations=1)
        return Image.fromarray(processed)

    @staticmethod
    def enhance_contrast(pil_image):
        gray = ImageProcessor.to_gray(pil_image)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced)

    @staticmethod
    def rescale_image(pil_image, scale=1.0):
        if scale == 1.0:
            return pil_image
        width, height = pil_image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        return pil_image.resize((new_width, new_height), Image.LANCZOS)

    @staticmethod
    def image_to_base64(pil_image):
        buffered = io.BytesIO()
        # Ensure we save as PNG for transparency support if needed, or just standard
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
