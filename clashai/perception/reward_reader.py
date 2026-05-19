# scripts/rl/reward_reader.py
# Reads stars (0-3) and destruction percentage
# on the attack results screen.
#
# Method:
# - Stars: HSV detection (silver = high brightness + low saturation)
# - Percentage: digit template matching (0-9), excluding detected %
#
# RESOLUTION-INDEPENDENT: works at 1920x1080, 2000x923, etc.
# No longer needs star_earned.png — stars are detected by color.
#
# Setup (once):
# Place templates 0.png to 9.png + pct.png in reward_templates/digits/
#
# Usage:
# results = read_attack_results()
# print(results['stars'], results['percentage'], results['reward'])

import os
import sys

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

from clashai.paths import REWARD_TEMPLATES_DIR, REWARD_DIGITS_DIR, DEBUG_DIR

TEMPLATES_DIR = REWARD_TEMPLATES_DIR
DIGITS_DIR = REWARD_DIGITS_DIR

# Thresholds for digit matching
DIGIT_MATCH_THRESHOLD = 0.60
PCT_MATCH_THRESHOLD = 0.50

# Thresholds for stars (HSV)
STAR_MIN_AREA = 1000
STAR_MAX_ASPECT = 2.5
STAR_SATURATION_MAX = 60
STAR_VALUE_MIN = 180


# =============================================================================
# ADB FUNCTIONS
# =============================================================================

# Re-exported from the canonical implementation in game_loop (Phase B.1).
# That version routes through WGC (fast, occlusion-proof) with ADB fallback.
from clashai.navigation.game_loop import adb_screenshot  # noqa: E402


# =============================================================================
# GREEN ISOLATION (Shared)
# =============================================================================

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


# =============================================================================
# STAR READING (HSV)
# =============================================================================

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


# =============================================================================
# DYNAMIC PERCENTAGE ZONE DETECTION
# =============================================================================

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


# =============================================================================
# PERCENTAGE READING (Digit Template Matching)
# =============================================================================

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
        print(f"{len(digit_templates)} digit templates loaded"
              f"{' + %' if pct_mask is not None else ''}")

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


# =============================================================================
# FULL READING
# =============================================================================

def read_attack_results(img_pil=None, debug=False):
    """Reads full attack results with logical correction."""
    if img_pil is None:
        img_pil = adb_screenshot()
        if img_pil is None:
            return {'stars': 0, 'percentage': 0, 'reward': 0, 'success': False}

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 1. Read the percentage (digit template matching)
    percentage = read_percentage(img_cv, debug=debug)

    # 2. Count stars (HSV)
    stars = count_stars(img_cv, debug=debug)

    # Fallback if template matching failed
    if percentage < 0:
        print(" WARNING: Digit matching failed, estimating from stars...")
        percentage = read_percentage_from_stars(stars)

    # ---------------------------------------------------------
    # LOGICAL CORRECTION
    # ---------------------------------------------------------

    # OCR fix 6→1: template matching often confuses 6 with 1.
    # Result: 60-69% read as 10-19%. Detected by cross-checking
    # with stars: 2 requires at least 50%.
    # For 1, 16% is technically possible (TH destroyed at 16%) but
    # statistically it is almost always a misread 6X%.
    if 10 <= percentage <= 19:
        corrected_pct = percentage + 50
        if stars >= 2:
            # 2 + <50% = impossible → certain correction
            print(f" OCR fix 6→1: {percentage}% impossible with {stars}"
                  f" → corrected to {corrected_pct}%")
            percentage = corrected_pct
        elif stars == 1 and corrected_pct <= 100:
            # 1 + 1X% = suspicious → probable correction
            print(f" OCR fix 6→1: {percentage}% suspect with {stars}"
                  f" → corrected to {corrected_pct}%")
            percentage = corrected_pct

    if percentage == 100:
        stars = 3
    elif 0 <= percentage < 100 and stars == 3:
        print(" Correction: Impossible to have 3 stars without 100%. Reducing to 2.")
        stars = 2
    elif percentage >= 50 and stars == 0:
        print(" Correction: >= 50% guarantees at least 1 star.")
        stars = 1

    # Final safeguard: 2 requires ≥50%
    if stars >= 2 and percentage < 50:
        print(f" Correction: {stars} but {percentage}% → forced to 50%")
        percentage = 50

    reward = calculate_reward(stars, percentage)

    return {
        'stars': stars,
        'percentage': percentage,
        'reward': reward,
        'success': True,
    }


def calculate_reward(stars, percentage):
    """Calculates the reward for the RL agent."""
    reward = (stars * 100) + percentage

    if stars >= 1:
        reward += 50
    if stars == 0:
        reward -= 50
    if stars == 3 and percentage == 100:
        reward += 50

    return reward


# =============================================================================
# TEMPLATE EXTRACTION
# =============================================================================

def extract_result_screen():
    """Captures the results screen and saves useful zones."""
    print(" Extracting the results screen...")
    print(" Make sure you are on the attack results screen")
    print(" (with stars and 'Victory' or 'Defeat')\n")

    img_pil = adb_screenshot()
    if img_pil is None:
        print("ERROR: Unable to capture screen")
        return

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    full_path = os.path.join(TEMPLATES_DIR, '_screenshot_resultats.png')
    cv2.imwrite(full_path, img_cv)

    # Dynamic percentage zone
    pct_coords = find_pct_region(img_cv)
    if pct_coords:
        x1, y1, x2, y2 = pct_coords
        pct_region = img_cv[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(TEMPLATES_DIR, '_zone_pourcentage.png'), pct_region)

    print(f"Screenshots saved in {TEMPLATES_DIR}/")
    print()
    print(" NEXT STEP:")
    print(" Place templates 0-9 + pct.png in reward_templates/digits/")
    print(" Then run --test to verify")


# =============================================================================
# TEST
# =============================================================================

def test_reward_reader(image_path=None):
    """Tests the reward reader."""
    print("Reward Reader Test\n")

    if image_path and os.path.exists(image_path):
        print(f" Image: {image_path}")
        img_pil = Image.open(image_path).convert("RGB")
    else:
        print(" ADB capture in progress...")
        img_pil = adb_screenshot()
        if img_pil is None:
            print("ERROR: Unable to capture screen")
            return

    results = read_attack_results(img_pil, debug=True)

    print(f"\n{'=' * 40}")
    print(f"* Stars: {results['stars']}/3")
    print(f"Percentage: {results['percentage']}%")
    print(f"RL Reward: {results['reward']}")
    print(f"{'=' * 40}")

    print("\nDebug images in debug_reward/ folder")


def test_digits_only(image_path=None):
    """Tests digit reading only."""
    print(" Digit Template Matching Test\n")

    if image_path and os.path.exists(image_path):
        print(f" Image: {image_path}")
        img_pil = Image.open(image_path).convert("RGB")
    else:
        print(" ADB capture in progress...")
        img_pil = adb_screenshot()
        if img_pil is None:
            print("ERROR: Unable to capture screen")
            return

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    pct = read_percentage(img_cv, debug=True)
    print(f"\nResult: {pct}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if '--extract' in sys.argv:
        extract_result_screen()
    elif '--test' in sys.argv:
        img_path = None
        for arg in sys.argv[2:]:
            if os.path.exists(arg):
                img_path = arg
                break
        test_reward_reader(img_path)
    elif '--test-digits' in sys.argv:
        img_path = None
        for arg in sys.argv[2:]:
            if os.path.exists(arg):
                img_path = arg
                break
        test_digits_only(img_path)
    else:
        print("Reward Reader — Attack results reader")
        print()
        print("Usage:")
        print(" python scripts/rl/reward_reader.py --extract (capture screen)")
        print(" python scripts/rl/reward_reader.py --test (test everything)")
        print(" python scripts/rl/reward_reader.py --test-digits (test digits only)")
        print()
        print("Setup:")
        print(" 1. Place templates 0-9 + pct.png in reward_templates/digits/")
        print(" 2. Run --test to verify")
        print()
        print("Note: star_earned.png is no longer needed!")
        print(" Stars are detected by color (HSV).")