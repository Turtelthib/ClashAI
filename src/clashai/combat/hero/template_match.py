# clashai/combat/hero/template_match.py
# Multi-scale template matching for ability icons.

import cv2
import numpy as np

from clashai.config import MATCH_SCALES


def _match_template_multiscale(region, template):
    """
    Multi-scale template matching.

    Returns:
        (best_val, best_loc, best_tw, best_th) or (0, None, 0, 0)
    """
    best_val = 0
    best_loc = None
    best_tw = 0
    best_th = 0

    for scale in MATCH_SCALES:
        th, tw = template.shape[:2]
        new_h = int(th * scale)
        new_w = int(tw * scale)

        if new_h > region.shape[0] or new_w > region.shape[1]:
            continue
        if new_h < 10 or new_w < 10:
            continue

        resized = cv2.resize(template, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_tw = new_w
            best_th = new_h

    return best_val, best_loc, best_tw, best_th

