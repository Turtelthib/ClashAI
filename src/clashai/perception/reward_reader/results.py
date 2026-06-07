# clashai/perception/reward_reader/results.py
# Top-level API: read stars + % + reward from the results screen.

import os
import sys
import cv2
import numpy as np
from PIL import Image

from clashai.navigation.game_loop import adb_screenshot
from clashai.perception.reward_reader.constants import DEBUG_DIR, TEMPLATES_DIR
from clashai.perception.reward_reader.stars import count_stars
from clashai.perception.reward_reader.percentage import (
    read_percentage, read_percentage_from_stars, find_pct_region,
)


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
