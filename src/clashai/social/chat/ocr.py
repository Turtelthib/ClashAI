# clashai/social/chat/ocr.py
# OCR engine for the clan chat (EasyOCR preferred, Tesseract fallback).

import cv2

_ocr_engine = None
_ocr_type = None


def _init_ocr():
    """Initializes the OCR engine (EasyOCR preferred, Tesseract fallback)."""
    global _ocr_engine, _ocr_type

    if _ocr_engine is not None:
        return _ocr_engine, _ocr_type

    # Try EasyOCR
    try:
        import easyocr
        _ocr_engine = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
        _ocr_type = 'easyocr'
        print(" OCR initialized: EasyOCR (fr+en)")
        return _ocr_engine, _ocr_type
    except ImportError:
        pass

    # Try pytesseract
    try:
        import pytesseract
        _ocr_engine = pytesseract
        _ocr_type = 'tesseract'
        print(" OCR initialized: Tesseract")
        return _ocr_engine, _ocr_type
    except ImportError:
        pass

    print("WARNING: No OCR engine available!")
    print(" Install: pip install easyocr")
    print(" Or: pip install pytesseract")
    _ocr_type = None
    return None, None


def _ocr_read(img_cv):
    """
    Reads text from a BGR image.

    Returns:
        lines: list of str (detected text lines)
    """
    engine, etype = _init_ocr()
    if engine is None:
        return []

    if etype == 'easyocr':
        # EasyOCR returns a list of (bbox, text, confidence)
        results = engine.readtext(img_cv, paragraph=False)
        lines = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and len(text.strip()) > 0:
                lines.append(text.strip())
        return lines

    elif etype == 'tesseract':
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        text = engine.image_to_string(gray, lang='fra+eng')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines

    return []
