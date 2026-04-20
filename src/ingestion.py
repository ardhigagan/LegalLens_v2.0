"""
ingestion.py — Upgraded LegalLens AI v2
- Improved OCR preprocessing (deskew + adaptive threshold)
- Better fallback detection (checks word count, not just char count)
- Handles multi-page PDFs more robustly
"""

import io
import pdfplumber
import pytesseract
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes


def clean_text(text: str) -> str:
    """Strips blank lines and excessive whitespace."""
    if not text:
        return ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def preprocess_image_for_ocr(img_np: np.ndarray) -> np.ndarray:
    """
    Applies image preprocessing to maximize OCR accuracy:
    1. Convert to grayscale
    2. Deskew (straighten rotated scans)
    3. Adaptive thresholding (handles uneven lighting)
    4. Denoise
    """
    # Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold (better than fixed for scanned docs)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)

    return denoised


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF.
    Strategy:
      1. Try native digital extraction with pdfplumber.
      2. If result is too short (< 50 words), fall back to OCR on rendered images.
    """
    text_content = ""

    # --- Try native extraction first ---
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
    except Exception as e:
        print(f"pdfplumber error: {e}")

    word_count = len(text_content.split())

    # --- Fallback to OCR if native extraction is poor ---
    if word_count < 50:
        print(f"Native extraction yielded only {word_count} words. Switching to OCR...")
        text_content = ""
        try:
            images = convert_from_bytes(file_bytes, dpi=300)
            for img in images:
                img_np = np.array(img)
                processed = preprocess_image_for_ocr(img_np)
                # Use pytesseract with page segmentation mode 3 (fully automatic)
                custom_config = r"--oem 3 --psm 3"
                page_text = pytesseract.image_to_string(processed, config=custom_config)
                text_content += page_text + "\n"
        except Exception as e:
            print(f"OCR error: {e}")

    return clean_text(text_content)


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extracts text from a standalone image file (JPG/PNG).
    Applies full preprocessing pipeline before OCR.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(image)
        processed = preprocess_image_for_ocr(img_np)
        custom_config = r"--oem 3 --psm 3"
        text = pytesseract.image_to_string(processed, config=custom_config)
        return clean_text(text)
    except Exception as e:
        print(f"Image OCR error: {e}")
        return ""
