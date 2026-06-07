# clashai/perception/reward_reader/percentage.py
# Destruction percentage via digit template matching.

import os
import cv2
import numpy as np

from clashai.perception.reward_reader.constants import (
    DIGITS_DIR, DIGIT_MATCH_THRESHOLD, PCT_MATCH_THRESHOLD, DEBUG_DIR,
)
from clashai.perception.reward_reader.green import isolate_green


def find_pct_region(img_cv):
    """
    Dynamically finds the green percentage zone.
    Looks for large green text in the upper third of the image.
    Resolution-independent.

    Returns:
        (x1, y1, x2, y2) in absolute coordinates, or None if not found.
    """
    h, w = img_cv.shape[:2]

    # Search zone: upper third, center third
    search_y1 = 0
    search_y2 = h // 3
    search_x1 = w // 3
    search_x2 = 2 * w // 3

    search_region = img_cv[search_y1:search_y2, search_x1:search_x2]

    # Green isolation with strict thresholds (large text only)
    mask = isolate_green(search_region, green_thresh=25, bright_thresh=100)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    # Filter components: digits = large enough, not too wide
    digit_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 80:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if bh > 10 and bw < 60:
                digit_components.append((x, y, bw, bh, area))

    if not digit_components:
        return None

    # Group components close in Y (same text line)
    median_y = np.median([c[1] for c in digit_components])
    close = [c for c in digit_components if abs(c[1] - median_y) < 15]
    if not close:
        close = digit_components

    # Enclosing bounding box + padding
    x1 = min(c[0] for c in close)
    y1 = min(c[1] for c in close)
    x2 = max(c[0] + c[2] for c in close)
    y2 = max(c[1] + c[3] for c in close)

    pad = 5
    return (search_x1 + x1 - pad, search_y1 + y1 - pad,
            search_x1 + x2 + pad, search_y1 + y2 + pad)



def load_digit_templates():
    """
    Loads digit templates 0-9 and % from reward_templates/digits/.
    Returns (digits_dict, pct_mask) where digits_dict = {int: mask} and
    pct_mask = mask of the % symbol.
    """
    digit_templates = {}
    pct_mask = None

    if not os.path.exists(DIGITS_DIR):
        print(f"WARNING: Digits folder not found: {DIGITS_DIR}")
        return digit_templates, pct_mask

    for digit in range(10):
        path = os.path.join(DIGITS_DIR, f'{digit}.png')
        if not os.path.exists(path):
            print(f"WARNING: Missing template: {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        mask = isolate_green(img)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 1
            mask = mask[max(0, y - pad):y + h + pad * 2,
                        max(0, x - pad):x + w + pad * 2]

        digit_templates[digit] = mask

    # Load the % template
    pct_path = os.path.join(DIGITS_DIR, 'pct.png')
    if os.path.exists(pct_path):
        pct_img = cv2.imread(pct_path)
        if pct_img is not None:
            pct_mask = isolate_green(pct_img)
            coords = cv2.findNonZero(pct_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                pad = 1
                pct_mask = pct_mask[max(0, y - pad):y + h + pad * 2,
                                    max(0, x - pad):x + w + pad * 2]

    if digit_templates:
        from clashai.config.logging import pp
        pp(f"{len(digit_templates)} digit templates loaded"
           f"{' + %' if pct_mask is not None else ''}", tag='init_done')

    return digit_templates, pct_mask


def read_percentage(img_cv, debug=False):
    """
    Reads the percentage by digit template matching.

    Approach:
    1. Dynamically find the green text zone (resolution-independent)
    2. Locate the "%" symbol to delimit the digit zone
    3. Match digits 0-9 only to the LEFT of %
    4. Read left to right to form the number
    """
    if debug:
        debug_dir = DEBUG_DIR
        os.makedirs(debug_dir, exist_ok=True)

    # 1. Dynamically find the percentage zone
    pct_coords = find_pct_region(img_cv)
    if pct_coords is None:
        if debug:
            print(" WARNING: PCT zone not found!")
        return -1

    rx1, ry1, rx2, ry2 = pct_coords
    region = img_cv[ry1:ry2, rx1:rx2]
    region_mask = isolate_green(region)
    region_h, region_w = region_mask.shape[:2]

    if debug:
        cv2.imwrite(os.path.join(debug_dir, 'pct_region.png'), region)
        cv2.imwrite(os.path.join(debug_dir, 'pct_green_mask.png'), region_mask)

    # Check that there is green content
    green_pixels = np.sum(region_mask > 0)
    if green_pixels < 30:
        if debug:
            print(" WARNING: Too few green pixels in PCT zone")
        return -1

    # 2. Load templates
    digit_templates, pct_tmpl = load_digit_templates()
    if not digit_templates:
        print(" ERROR: No digit templates available!")
        return -1

    # Height of green content
    coords = cv2.findNonZero(region_mask)
    if coords is None:
        return -1
    _, _, _, content_h = cv2.boundingRect(coords)

    # 3. Locate "%" to delimit the digit zone
    digits_right_limit = region_w

    if pct_tmpl is not None:
        pct_h, pct_w = pct_tmpl.shape[:2]
        base_scale_pct = content_h / pct_h
        best_pct_conf = 0
        best_pct_x = region_w

        for s in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            scale = base_scale_pct * s
            new_h = int(pct_h * scale)
            new_w = int(pct_w * scale)
            if new_h > region_h or new_w > region_w or new_h < 5 or new_w < 3:
                continue

            resized = cv2.resize(pct_tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(region_mask, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_pct_conf:
                best_pct_conf = max_val
                best_pct_x = max_loc[0]

        if best_pct_conf >= PCT_MATCH_THRESHOLD:
            digits_right_limit = best_pct_x - 2

        if debug:
            print(f" % detected: conf={best_pct_conf:.2f}, x={best_pct_x}, "
                  f"digit limit={digits_right_limit}")

    # 4. Match digits only to the left of %
    digit_region = region_mask[:, :max(1, digits_right_limit)]

    all_matches = []
    for digit, tmpl_mask in digit_templates.items():
        tmpl_h, tmpl_w = tmpl_mask.shape[:2]
        base_scale = content_h / tmpl_h

        for s in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            scale = base_scale * s
            new_h = max(5, int(tmpl_h * scale))
            new_w = max(3, int(tmpl_w * scale))

            if new_h > digit_region.shape[0] or new_w > digit_region.shape[1]:
                continue
            if new_h < 5 or new_w < 3:
                continue

            resized = cv2.resize(tmpl_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(digit_region, resized, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= DIGIT_MATCH_THRESHOLD)

            for my, mx in zip(locations[0], locations[1]):
                conf = result[my, mx]
                cx = mx + new_w // 2
                all_matches.append((cx, mx, mx + new_w, digit, conf, new_w, my, new_h))

    if not all_matches:
        if debug:
            print(" WARNING: No digit detected")
        return -1

    # 5. NMS based on bounding box overlap
    all_matches.sort(key=lambda m: -m[4])
    kept = []
    for match in all_matches:
        cx, x_left, x_right, digit, conf, mw, my, mh = match

        is_dup = False
        for existing in kept:
            ex_left, ex_right, emw = existing[1], existing[2], existing[5]
            overlap_start = max(x_left, ex_left)
            overlap_end = min(x_right, ex_right)
            overlap = max(0, overlap_end - overlap_start)
            min_width = min(mw, emw)

            if overlap > min_width * 0.4:
                is_dup = True
                break

        if not is_dup:
            kept.append(match)

    # 6. Read left to right
    kept.sort(key=lambda m: m[0])
    digits_found = [m[3] for m in kept]

    if debug:
        debug_img = cv2.cvtColor(region_mask, cv2.COLOR_GRAY2BGR)
        # Red line = right limit (start of %)
        if digits_right_limit < region_w:
            cv2.line(debug_img, (digits_right_limit, 0),
                     (digits_right_limit, region_h), (0, 0, 255), 1)
        for cx, x_left, x_right, digit, conf, mw, my, mh in kept:
            cv2.rectangle(debug_img, (x_left, my), (x_right, my + mh), (0, 255, 0), 1)
            cv2.putText(debug_img, f"{digit}({conf:.2f})", (x_left, max(12, my - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, 'pct_digit_matches.png'), debug_img)

        print(f" Raw matches: {len(all_matches)}, after NMS: {len(kept)}")
        for cx, x_left, x_right, digit, conf, mw, my, mh in kept:
            print(f" Digit {digit} at x={cx}, conf={conf:.3f}")

    if not digits_found:
        return -1

    # Limit to 3 digits max
    if len(digits_found) > 3:
        digits_found = digits_found[:3]

    # Build the number
    number = 0
    for d in digits_found:
        number = number * 10 + d

    # Validation: 0 to 100
    if number > 100:
        if len(digits_found) >= 3 and digits_found[:3] == [1, 0, 0]:
            number = 100
        elif len(digits_found) >= 2:
            n2 = digits_found[0] * 10 + digits_found[1]
            if 0 <= n2 <= 100:
                number = n2
                if debug:
                    print(f" Correction: {digits_found} → {n2}")
            else:
                number = digits_found[0]
        else:
            number = digits_found[0]

    if debug:
        print(f" Percentage read: {number}%")

    return number


def read_percentage_from_stars(stars):
    """Percentage estimate based on stars (fallback)."""
    estimates = {0: 15, 1: 55, 2: 75, 3: 100}
    return estimates.get(stars, 0)

