# clashai/combat/combat_observer/health_bars.py
# HSV health-bar detection (healthy / injured troops, heroes).

import cv2
import numpy as np

from clashai.combat.combat_observer.constants import (
    HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
    HP_RED_H_RANGE, HP_RED_S_MIN, HP_RED_V_MIN,
    HP_ORANGE_H_RANGE, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN,
    HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO,
    HERO_BAR_MIN_AREA, HERO_BAR_MAX_AREA, HERO_BAR_MIN_RATIO,
    UI_BOTTOM_Y, UI_TOP_Y, UI_LEFT_X, UI_RIGHT_X,
)


def _detect_bars(img_cv, h_range, s_min, v_min, min_area, max_area, min_ratio):
    """
    Detects horizontal health bars by HSV color.

    Returns:
        positions: list of (x, y) in image coordinates
        areas: list of areas for each bar (to distinguish heroes from troops)
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    # Color mask
    lower = np.array([h_range[0], s_min, v_min])
    upper = np.array([h_range[1], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Exclude UI zones
    mask[:int(h * UI_TOP_Y), :] = 0
    mask[int(h * UI_BOTTOM_Y):, :] = 0
    mask[:, :int(w * UI_LEFT_X)] = 0
    mask[:, int(w * UI_RIGHT_X):] = 0

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
        if h_rect == 0:
            continue
        ratio = w_rect / h_rect
        if ratio < min_ratio:
            continue

        cx = x_rect + w_rect // 2
        cy = y_rect + h_rect // 2
        positions.append((cx, cy))
        areas.append(area)

    return positions, areas


def detect_troop_bars(img_cv):
    """Detects green health bars (healthy troops)."""
    return _detect_bars(
        img_cv, HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )


def detect_hurt_bars(img_cv):
    """Detects red/orange health bars (injured troops)."""
    red_pos, red_areas = _detect_bars(
        img_cv, HP_RED_H_RANGE, HP_RED_S_MIN, HP_RED_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )
    orange_pos, orange_areas = _detect_bars(
        img_cv, HP_ORANGE_H_RANGE, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )
    return red_pos + orange_pos, red_areas + orange_areas


def detect_hero_bars(img_cv):
    """
    Detects hero health bars (larger than normal troop bars).
    Looks for large green AND orange/red bars.
    """
    green_pos, green_areas = _detect_bars(
        img_cv, HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
        HERO_BAR_MIN_AREA, HERO_BAR_MAX_AREA, HERO_BAR_MIN_RATIO
    )
    red_pos, red_areas = _detect_bars(
        img_cv, HP_RED_H_RANGE, HP_RED_S_MIN, HP_RED_V_MIN,
        HERO_BAR_MIN_AREA, HERO_BAR_MAX_AREA, HERO_BAR_MIN_RATIO
    )

    all_pos = green_pos + red_pos
    # Number of heroes detected (typically 0-5).
    # We cannot identify WHICH hero from the bar alone,
    # but we know how many are alive.
    return all_pos
