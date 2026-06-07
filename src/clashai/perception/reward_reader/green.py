# clashai/perception/reward_reader/green.py
# Shared green-channel isolation helper.

import cv2
import numpy as np


def isolate_green(img_bgr, green_thresh=20, bright_thresh=70):
    """
    Isolates green CoC text from a BGR image.
    Returns a binary mask (255 = green, 0 = background).
    """
    b, g, r = cv2.split(img_bgr)
    green_diff = cv2.subtract(g, cv2.max(r, b))
    _, mask_diff = cv2.threshold(green_diff, green_thresh, 255, cv2.THRESH_BINARY)
    _, mask_bright = cv2.threshold(g, bright_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_diff, mask_bright)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

