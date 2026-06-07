# clashai/perception/deploy/boundary.py
# HSV-based detection of the village boundary (red-overlay warm grass).

import cv2
import numpy as np

from clashai.config import ADB_WIDTH, ADB_HEIGHT
from clashai.perception.deploy.constants import (
    UI_EXCLUSION_ZONES, SCREEN_MARGIN,
    HSV_CHECK_RADIUS, HSV_RED_H_MAX, HSV_RED_SAT_MIN, HSV_RED_RATIO_THRESHOLD,
)


def _is_in_red_overlay(img_bgr, x, y, radius=HSV_CHECK_RADIUS):
    """
    True if the small patch around (x, y) is within the CoC red overlay.
    Fails silently (False) on dark villages → no regression.
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x) - radius)
    y1 = max(0, int(y) - radius)
    x2 = min(w, int(x) + radius + 1)
    y2 = min(h, int(y) + radius + 1)

    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red_mask = (hsv[:, :, 0] < HSV_RED_H_MAX) & (hsv[:, :, 1] > HSV_RED_SAT_MIN)
    red_ratio = float(red_mask.sum()) / float(red_mask.size)
    return red_ratio >= HSV_RED_RATIO_THRESHOLD

def detect_village_boundary(img_cv):
    """
    Detects the village boundary from a BGR screenshot.

    CoC's red overlay shifts the HSV hue of the grass: H≈33 (green)
    becomes H≈20 (warm yellow-green). We detect this warm zone to
    find the interior of the village.

    Args:
        img_cv: BGR image (numpy array) of the screenshot

    Returns:
        hull: convex hull of the village (numpy array Nx1x2) or None
        center: village center (x, y) in image pixels or None
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    # --- UI Mask ---
    # Exclude UI zones (in image coordinates)
    # Compute the image→ADB ratio for conversion
    scale_x = w / ADB_WIDTH
    scale_y = h / ADB_HEIGHT

    ui_mask = np.ones((h, w), dtype=np.uint8) * 255
    for ax1, ay1, ax2, ay2 in UI_EXCLUSION_ZONES:
        ix1 = int(ax1 * scale_x)
        iy1 = int(ay1 * scale_y)
        ix2 = int(ax2 * scale_x)
        iy2 = int(ay2 * scale_y)
        ui_mask[iy1:iy2, ix1:ix2] = 0

    # Also exclude extreme edges
    margin = int(SCREEN_MARGIN * min(scale_x, scale_y))
    ui_mask[:margin, :] = 0
    ui_mask[h - margin:, :] = 0
    ui_mask[:, :margin] = 0
    ui_mask[:, w - margin:] = 0

    # --- Warm grass detection (village red zone) ---
    # H=14-28 : grass tinted by the red overlay
    # S>80 : saturated (not grey)
    # V>100 : bright (not forest shadow)
    mask_warm = cv2.inRange(hsv, (14, 80, 100), (28, 255, 255))
    mask_warm = cv2.bitwise_and(mask_warm, ui_mask)

    # --- Morphological cleanup ---
    # Close : fill holes (buildings, walls, decorations)
    kernel_close = np.ones((30, 30), np.uint8)
    # Open : remove noise (small warm pixels in the forest)
    kernel_open = np.ones((15, 15), np.uint8)

    min_contour_area = h * w * 0.05

    # --- Cascade strategy: try from most precise to broadest ---
    strategies = [
        ("warm", mask_warm),
    ]

    # Prepare fallbacks
    # Fallback 1 : light grass (works when the red overlay is faint)
    mask_grass = cv2.inRange(hsv, (18, 70, 130), (38, 255, 255))
    mask_grass = cv2.bitwise_and(mask_grass, ui_mask)
    strategies.append(("herbe claire", mask_grass))

    # Fallback 2 : wide grass (very permissive)
    mask_wide = cv2.inRange(hsv, (15, 60, 110), (40, 255, 255))
    mask_wide = cv2.bitwise_and(mask_wide, ui_mask)
    strategies.append(("herbe large", mask_wide))

    # Fallback 3 : DARK VILLAGES (underwater theme, night, etc.)
    # The ground is blue-green (H=75-120) instead of green (H=33).
    # The village interior is brighter (V>90) than the dark exterior (V<70).
    # We use brightness + saturation to isolate the game zone.
    mask_bright = cv2.inRange(hsv, (0, 20, 90), (180, 255, 255))
    # Exclude UI zones and pure white (text, clouds)
    mask_not_white = cv2.inRange(hsv, (0, 15, 0), (180, 255, 240))
    mask_dark_village = cv2.bitwise_and(mask_bright, mask_not_white)
    mask_dark_village = cv2.bitwise_and(mask_dark_village, ui_mask)
    strategies.append(("village sombre (luminosité)", mask_dark_village))

    # Fallback 4 : Direct red border (some dark villages have a visible red line)
    # H<10 or H>170 (red), S>80, V>60
    mask_red_low = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_red_high = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask_red_border = cv2.bitwise_or(mask_red_low, mask_red_high)
    mask_red_border = cv2.bitwise_and(mask_red_border, ui_mask)
    # Dilate the border to fill the interior
    kernel_dilate = np.ones((50, 50), np.uint8)
    mask_red_filled = cv2.dilate(mask_red_border, kernel_dilate, iterations=3)
    mask_red_filled = cv2.morphologyEx(mask_red_filled, cv2.MORPH_CLOSE,
                                        np.ones((60, 60), np.uint8))
    strategies.append(("bordure rouge", mask_red_filled))

    hull = None
    center = None

    for strategy_name, mask_raw in strategies:
        mask_filled = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close)
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_contour_area:
            continue

        # Success — use this contour
        if strategy_name != "warm":
            from clashai.config.logging import pp
            pp(f" WARNING: Overlay rouge faible, fallback {strategy_name}", tag='warning')

        hull = cv2.convexHull(largest)

        M = cv2.moments(hull)
        if M["m00"] > 0:
            center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        else:
            center = np.mean(hull.reshape(-1, 2).astype(float), axis=0)

        break

    if hull is None:
        return None, None

    return hull, center

