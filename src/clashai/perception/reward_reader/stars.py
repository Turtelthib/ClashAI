# clashai/perception/reward_reader/stars.py
# Star counting (0-3) via HSV silver detection.

import os
import cv2
import numpy as np

from clashai.perception.reward_reader.constants import (
    STAR_MIN_AREA, STAR_MAX_ASPECT, STAR_SATURATION_MAX, STAR_VALUE_MIN, DEBUG_DIR,
)


def count_stars(img_cv, debug=False):
    """
    Counts earned stars by HSV filtering.

    Logic:
    - Earned stars = silver/white = high brightness + low saturation
    - Lost stars = black/dark (invisible to the filter)
    - The gold "Victory" banner is excluded because it is saturated (gold ≠ silver)

    Resolution-independent: zones are expressed as percentages of the image.
    No longer needs the star_earned.png template.
    """
    h, w = img_cv.shape[:2]

    # Star zone: upper third, center of the image
    sy1 = int(h * 0.03)
    sy2 = int(h * 0.35)
    sx1 = int(w * 0.25)
    sx2 = int(w * 0.65)

    region = img_cv[sy1:sy2, sx1:sx2]
    rh, rw = region.shape[:2]

    if debug:
        debug_dir = DEBUG_DIR
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, 'star_region.png'), region)

    # HSV filtering: silver = low saturation + high value
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, STAR_VALUE_MIN), (180, STAR_SATURATION_MAX, 255))

    # Morphological cleanup
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    # Filter: stars are large, near the top, and roughly square
    star_candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cy = centroids[i][1]
        cx = centroids[i][0]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area > STAR_MIN_AREA and cy < rh * 0.70:
            aspect = bw / bh if bh > 0 else 99
            if 1.0 / STAR_MAX_ASPECT < aspect < STAR_MAX_ASPECT:
                star_candidates.append((cx, cy, area, bw, bh))

    # Spatial NMS: remove detections that are too close together
    star_candidates.sort(key=lambda c: -c[2])
    kept = []
    for sc in star_candidates:
        cx = sc[0]
        is_dup = any(abs(cx - k[0]) < rw * 0.10 for k in kept)
        if not is_dup:
            kept.append(sc)

    stars = min(len(kept), 3)

    if debug:
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for cx, cy, area, bw, bh in kept:
            cv2.circle(debug_img, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            cv2.putText(debug_img, f"a={area}", (int(cx) - 20, int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, 'star_mask.png'), debug_img)
        print(f" * Stars: {stars} ({len(star_candidates)} candidates, {len(kept)} after NMS)")

    return stars

