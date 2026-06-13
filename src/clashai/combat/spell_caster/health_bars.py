# clashai/combat/spell_caster/health_bars.py
# Health-bar detection for spell targeting (green / red+orange, color param).

import cv2
import numpy as np

from clashai.combat.spell_caster.constants import (
    HP_BAR_H_MIN, HP_BAR_H_MAX, HP_BAR_S_MIN, HP_BAR_V_MIN,
    HP_RED_H_MIN, HP_RED_H_MAX, HP_RED_S_MIN, HP_RED_V_MIN,
    HP_ORANGE_H_MIN, HP_ORANGE_H_MAX, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN,
    HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO,
    UI_EXCLUSION_Y, UI_EXCLUSION_TOP,
)


def detect_health_bars(img_cv, color='green'):
    """
    Detects health bars on a combat screenshot.

    Args:
        img_cv: BGR image
        color: 'green' (healthy), 'red' (injured), 'all' (both)

    Returns:
        positions: list of (x, y) — centers of detected bars
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    y_min = int(h * UI_EXCLUSION_TOP)
    y_max = int(h * UI_EXCLUSION_Y)
    roi_hsv = hsv[y_min:y_max, :, :]

    if color == 'green':
        mask = cv2.inRange(roi_hsv,
                           (HP_BAR_H_MIN, HP_BAR_S_MIN, HP_BAR_V_MIN),
                           (HP_BAR_H_MAX, 255, 255))
    elif color == 'red':
        # Red
        mask1 = cv2.inRange(roi_hsv,
                            (HP_RED_H_MIN, HP_RED_S_MIN, HP_RED_V_MIN),
                            (HP_RED_H_MAX, 255, 255))
        mask2 = cv2.inRange(roi_hsv,
                            (170, HP_RED_S_MIN, HP_RED_V_MIN),
                            (180, 255, 255))
        # Orange
        mask3 = cv2.inRange(roi_hsv,
                            (HP_ORANGE_H_MIN, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN),
                            (HP_ORANGE_H_MAX, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
    else:
        mask_g = cv2.inRange(roi_hsv,
                             (HP_BAR_H_MIN, HP_BAR_S_MIN, HP_BAR_V_MIN),
                             (HP_BAR_H_MAX, 255, 255))
        mask_r1 = cv2.inRange(roi_hsv,
                              (HP_RED_H_MIN, HP_RED_S_MIN, HP_RED_V_MIN),
                              (HP_RED_H_MAX, 255, 255))
        mask_r2 = cv2.inRange(roi_hsv,
                              (170, HP_RED_S_MIN, HP_RED_V_MIN),
                              (180, 255, 255))
        mask_o = cv2.inRange(roi_hsv,
                             (HP_ORANGE_H_MIN, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN),
                             (HP_ORANGE_H_MAX, 255, 255))
        mask = cv2.bitwise_or(mask_g, mask_r1)
        mask = cv2.bitwise_or(mask, mask_r2)
        mask = cv2.bitwise_or(mask, mask_o)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < HP_BAR_MIN_AREA or area > HP_BAR_MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        if bw / bh < HP_BAR_MIN_RATIO:
            continue
        cx = x + bw // 2
        cy = (y + bh // 2) + y_min
        positions.append((cx, cy))

    return positions
