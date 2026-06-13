# clashai/navigation/gdc/ocr.py
# OCR detection of enemy CW target numbers (#1..#50).

import re

import cv2
import numpy as np

from clashai.navigation.gdc.constants import TARGET_LIST_ZONE


def _detect_target_numbers(screenshot_pil):
    """
    Detects visible target numbers on the enemy CW screen.

    In CoC, each enemy has a number (#1, #2, ..., #50) displayed
    next to their name in the war list.

    Returns:
        targets: dict {number: (x_center, y_center)} of visible targets
    """
    try:
        from clashai.social.clan_chat_monitor import _init_ocr
    except ImportError:
        return {}

    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    zone = img_cv[TARGET_LIST_ZONE['top']:TARGET_LIST_ZONE['bottom'],
                   TARGET_LIST_ZONE['left']:TARGET_LIST_ZONE['right']]

    engine, etype = _init_ocr()
    if engine is None:
        return {}

    targets = {}

    if etype == 'easyocr':
        results = engine.readtext(zone, paragraph=False)
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            # Look for numbers (#1, #2, 1., 2., etc.)
            match = re.search(r'#?(\d{1,2})', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 50:
                    # Position at the center of the bbox
                    cx = int((bbox[0][0] + bbox[2][0]) / 2) + TARGET_LIST_ZONE['left']
                    cy = int((bbox[0][1] + bbox[2][1]) / 2) + TARGET_LIST_ZONE['top']
                    targets[num] = (cx, cy)

    elif etype == 'tesseract':
        import pytesseract
        data = pytesseract.image_to_data(zone, output_type=pytesseract.Output.DICT)
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
            match = re.search(r'#?(\d{1,2})', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 50:
                    x = data['left'][i] + data['width'][i] // 2 + TARGET_LIST_ZONE['left']
                    y = data['top'][i] + data['height'][i] // 2 + TARGET_LIST_ZONE['top']
                    targets[num] = (x, y)

    return targets
