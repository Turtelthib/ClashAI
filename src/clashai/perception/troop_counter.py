# scripts/rl/troop_count_reader.py
# Reads the troop count (x2, x6, x10) displayed on each icon
# in the attack bar.
#
# Method:
# 1. TroopFinder gives the position of each troop in the bar
# 2. We crop the "xN" zone in the top-left of the icon
# 3. We isolate white text by thresholding
# 4. We read digits by template matching (same templates as reward_reader)
#
# Usage:
# from clashai.perception.troop_counter import read_troop_counts
# counts = read_troop_counts(screenshot_pil, troop_finder)
# # counts = {'golem': 2, 'sorcier': 6, 'sorciere': 10, ...}

import os
import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

from clashai.paths import REWARD_DIGITS_DIR

# Digit templates folder (shared with reward_reader)
DIGITS_DIR = REWARD_DIGITS_DIR

# The "xN" zone is in the top-left of each icon in the bar
# Offset relative to the template match position (icon center)
# Icons are ~80x80px in the bar, the "xN" is in the top-left
COUNT_OFFSET_X = -35
COUNT_OFFSET_Y = -45
COUNT_WIDTH = 45
COUNT_HEIGHT = 25

# Threshold for white text
WHITE_THRESHOLD = 200

# Template matching
DIGIT_MATCH_THRESHOLD = 0.65


# =============================================================================
# TEMPLATE LOADING
# =============================================================================

_digit_templates = None

def _load_digit_templates():
    """Loads digit templates 0-9."""
    global _digit_templates
    if _digit_templates is not None:
        return _digit_templates

    _digit_templates = {}

    if not os.path.exists(DIGITS_DIR):
        print(f"WARNING: Digits folder not found: {DIGITS_DIR}")
        return _digit_templates

    for digit in range(10):
        path = os.path.join(DIGITS_DIR, f'{digit}.png')
        if os.path.exists(path):
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                _digit_templates[digit] = tmpl

    return _digit_templates


# =============================================================================
# COUNTER READING
# =============================================================================

def _read_count_from_region(region_bgr):
    """
    Reads a number (1-99) from a small image containing "xN" or "xNN".

    Args:
        region_bgr: BGR image of the zone containing the text

    Returns:
        count: int or None if unreadable
    """
    if region_bgr is None or region_bgr.size == 0:
        return None

    templates = _load_digit_templates()
    if not templates:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)

    # Isolate white text (digits are white on dark background)
    _, binary = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Upscale for template matching (digits are small)
    scale = 2.0
    binary_large = cv2.resize(binary, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    # Look for digits by template matching
    found_digits = []

    for digit, tmpl in templates.items():
        # Try multiple template sizes
        for tmpl_scale in [0.4, 0.5, 0.6, 0.7, 0.8]:
            h_t = max(1, int(tmpl.shape[0] * tmpl_scale))
            w_t = max(1, int(tmpl.shape[1] * tmpl_scale))

            if h_t >= binary_large.shape[0] or w_t >= binary_large.shape[1]:
                continue

            tmpl_resized = cv2.resize(tmpl, (w_t, h_t))

            # Binarize the template as well
            _, tmpl_bin = cv2.threshold(tmpl_resized, 128, 255, cv2.THRESH_BINARY)

            result = cv2.matchTemplate(binary_large, tmpl_bin, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= DIGIT_MATCH_THRESHOLD)

            for my, mx in zip(locations[0], locations[1]):
                # Check that we don't already have a digit at this position
                duplicate = False
                for fx, fd in found_digits:
                    if abs(mx - fx) < w_t * 0.6:
                        duplicate = True
                        # Keep the best match
                        if result[my, mx] > fd[1]:
                            found_digits.remove((fx, fd))
                            found_digits.append((mx, (digit, result[my, mx])))
                        break
                if not duplicate:
                    found_digits.append((mx, (digit, result[my, mx])))

    if not found_digits:
        return None

    # Sort by X position (left → right)
    found_digits.sort(key=lambda x: x[0])

    # Build the number
    number = 0
    for _, (digit, conf) in found_digits:
        number = number * 10 + digit

    # Sanity check
    if number < 1 or number > 99:
        return None

    return number


def read_troop_counts(screenshot_pil, troop_finder):
    """
    Reads the count of each troop from the attack bar.

    Args:
        screenshot_pil: PIL Image of the full screenshot
        troop_finder: TroopFinder with positions already updated

    Returns:
        counts: dict {troop_name: count} for detected troops
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
    from clashai.perception.coord_utils import ImageScaler
    scaler = ImageScaler(img_cv)
    h, w = scaler.img_h, scaler.img_w

    counts = {}

    for name, (tx, ty, conf) in troop_finder.positions.items():
        # Convert ADB position (1920x1080 canonical) to image coordinates
        ix, iy = scaler.to_img(tx, ty)

        # "xN" zone in the top-left of the icon (offsets in canonical px)
        x1 = ix + scaler.to_img_x(COUNT_OFFSET_X)
        y1 = iy + scaler.to_img_y(COUNT_OFFSET_Y)
        x2 = x1 + scaler.to_img_x(COUNT_WIDTH)
        y2 = y1 + scaler.to_img_y(COUNT_HEIGHT)

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        region = img_cv[y1:y2, x1:x2]
        count = _read_count_from_region(region)

        if count is not None:
            counts[name] = count

    return counts


# =============================================================================
# TEST
# =============================================================================

def test_reader(image_path=None):
    """Tests the counter reader."""
    from clashai.perception.troop_finder import TroopFinder

    print("Troop Count Reader Test\n")

    if image_path:
        img_pil = Image.open(image_path).convert("RGB")
    else:
        from clashai.navigation.game_loop import adb_screenshot
        img_pil = adb_screenshot()
        if img_pil is None:
            print("ERROR: Unable to capture screen")
            return

    finder = TroopFinder()
    finder.update(img_pil)

    counts = read_troop_counts(img_pil, finder)

    print("\nCounts read:")
    for name, count in sorted(counts.items()):
        print(f" {name}: x{count}")

    # Troops detected but no count read
    for name in finder.positions:
        if name not in counts:
            print(f" {name}: ??? (non lu)")


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else None
    test_reader(img)
