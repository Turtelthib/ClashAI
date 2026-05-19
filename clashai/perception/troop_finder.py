# scripts/rl/troop_finder.py
# Detects and selects troops in the bottom bar via template matching.
# Robust: works regardless of slot order, missing heroes,
# event troops, etc.
#
# v2: Multi-scale matching + bar scroll if troops are missing
#
# Setup (once):
# 1. python scripts/rl/troop_finder.py --extract
# → Captures the troop bar and saves it
# 2. Open troop_templates/_barre_complete.png in an image editor
# 3. Cut out each icon and save with the correct name:
# golem.png, pekka.png, sorcier.png, etc.
#
# Usage in code:
# finder = TroopFinder()
# finder.update(screenshot_pil) # Analyze the current bar
# finder.select("golem") # Click on the golem
# finder.select("rage") # Click on the rage spell

import os
import sys
import subprocess
import time

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

from clashai.paths import TROOP_TEMPLATES_DIR

TEMPLATES_DIR = TROOP_TEMPLATES_DIR

# Troop bar zone in the screen (1920x1080)
BAR_TOP = 950
BAR_BOTTOM = 1080
BAR_LEFT = 0
BAR_RIGHT = 1920

# Confidence threshold for template matching (local — TroopFinder uses 0.45,
# lower than hero_ability/clan_castle which use 0.50/0.60).
MATCH_THRESHOLD = 0.45

# Multi-scale list re-imported from clashai/config/perception.py (Phase A).
from clashai.config import MATCH_SCALES  # noqa: E402


# =============================================================================
# ADB FUNCTIONS
# =============================================================================

def adb_tap(x, y, delay=0.05):
    """Tap ADB."""
    subprocess.run(["adb", "shell", f"input tap {x} {y}"],
                   capture_output=True, timeout=5)
    time.sleep(delay)


def adb_swipe(x1, y1, x2, y2, duration_ms=300):
    """Swipe ADB."""
    subprocess.run(
        ["adb", "shell", f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"],
        capture_output=True, timeout=5
    )
    time.sleep(0.3)


# Re-exported from the canonical implementation in game_loop (Phase B.1).
# That version routes through WGC (fast, occlusion-proof) with ADB fallback.
from clashai.navigation.game_loop import adb_screenshot  # noqa: E402


# =============================================================================
# TROOP FINDER
# =============================================================================

class TroopFinder:
    """
    Finds and selects troops in the bottom bar
    using OpenCV template matching.
    """

    def __init__(self, templates_dir=TEMPLATES_DIR, detector=None):
        self.templates_dir = templates_dir
        self.templates = {}
        self.positions = {}
        self._detector = detector  # TroopBarDetector — YOLO primary, template matching fallback
        self._load_templates()

    def _load_templates(self):
        """Loads all templates from the troop_templates/ folder."""
        if not os.path.exists(self.templates_dir):
            print(f"WARNING: Templates folder not found: {self.templates_dir}")
            print(" Run first: python scripts/rl/troop_finder.py --extract")
            return

        count = 0
        for filename in os.listdir(self.templates_dir):
            if not filename.endswith('.png'):
                continue
            if filename.startswith('_'):
                continue

            name = os.path.splitext(filename)[0]
            path = os.path.join(self.templates_dir, filename)
            template = cv2.imread(path)

            if template is not None:
                self.templates[name] = template
                count += 1

        if count > 0:
            print(f"{count} troop templates loaded: {sorted(self.templates.keys())}")
        else:
            print(f"WARNING: No template found in {self.templates_dir}")

    def _match_template_multiscale(self, bar_region, template):
        """
        Multi-scale template matching.
        Returns (max_val, max_loc, best_scale) or (0, None, None) if no match.
        """
        best_val = 0
        best_loc = None
        _best_scale = None
        best_tw = 0
        best_th = 0

        for scale in MATCH_SCALES:
            th, tw = template.shape[:2]
            new_h = int(th * scale)
            new_w = int(tw * scale)

            if new_h > bar_region.shape[0] or new_w > bar_region.shape[1]:
                continue
            if new_h < 10 or new_w < 10:
                continue

            resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(bar_region, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                _best_scale = scale
                best_tw = new_w
                best_th = new_h

        return best_val, best_loc, best_tw, best_th

    def update(self, screenshot_pil, screen='combat'):
        """
        Analyzes the current troop bar and updates self.positions.

        V4.3: uses YOLO TroopBarDetector if available (faster, no templates needed).
        Falls back to template matching if the detector is not loaded.

        Args:
            screenshot_pil: PIL image of the full screen
            screen: 'combat' (battle bar, counter top-right)
                    'prep'   (army selection screen, counter top-left)
        """
        if self._detector is not None:
            self._update_yolo(screenshot_pil, screen=screen)
        else:
            self._update_template(screenshot_pil)

    def _update_yolo(self, screenshot_pil, screen='combat'):
        """YOLO-based update — primary path.

        Args:
            screen: 'combat' (battle bar, counter top-right)
                    'prep'   (army selection, counter top-left)
        """
        detections = self._detector.detect(screenshot_pil, screen=screen)
        self.positions = self._detector.to_positions(detections)

        found = len(self.positions)
        print(f"Troops detected: {found} (YOLO)")
        for name, (x, y, conf) in sorted(self.positions.items(), key=lambda i: i[1][0]):
            print(f" {name:<25s} -> ({x:4d}, {y:4d}) conf: {conf:.2f}")

    def _update_template(self, screenshot_pil):
        """Template matching fallback — legacy path."""
        screen = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        bar_region = screen[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]

        self.positions = {}

        for name, template in self.templates.items():
            best_val, best_loc, best_tw, best_th = self._match_template_multiscale(
                bar_region, template)

            if best_val >= MATCH_THRESHOLD and best_loc is not None:
                match_x = BAR_LEFT + best_loc[0] + best_tw // 2
                match_y = BAR_TOP + best_loc[1] + best_th // 2
                self.positions[name] = (match_x, match_y, best_val)

        found = len(self.positions)
        total = len(self.templates)
        print(f"Troops detected: {found}/{total} (template)")

        for name, (x, y, conf) in sorted(self.positions.items(), key=lambda i: i[1][0]):
            print(f" {name:<25s} -> ({x:4d}, {y:4d}) conf: {conf:.2f}")

        missing = set(self.templates.keys()) - set(self.positions.keys())
        if missing:
            print(f" WARNING: Not found: {sorted(missing)}")

    def update_with_scroll(self, scroll_attempts=2):
        """
        Like update() but scrolls the bar if troops are missing.
        Useful when the army has many different types.

        Args:
            scroll_attempts: number of scrolls to try
        """
        # First scan without scroll
        img = adb_screenshot()
        if img is None:
            return
        self.update(img)

        # If all troops are found, stop
        if len(self.positions) >= len(self.templates):
            return

        missing = set(self.templates.keys()) - set(self.positions.keys())
        if not missing:
            return

        # Scroll the bar to the right and rescan
        for attempt in range(scroll_attempts):
            print(f"  Scroll de la barre (tentative {attempt+1})...")
            # Swipe right to left in the bar to reveal hidden troops
            adb_swipe(1400, 1020, 600, 1020, 300)
            time.sleep(0.5)

            img = adb_screenshot()
            if img is None:
                continue

            screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bar_region = screen[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]

            # Scan only missing troops
            newly_found = 0
            for name in list(missing):
                if name not in self.templates:
                    continue

                best_val, best_loc, best_tw, best_th = self._match_template_multiscale(
                    bar_region, self.templates[name])

                if best_val >= MATCH_THRESHOLD and best_loc is not None:
                    match_x = BAR_LEFT + best_loc[0] + best_tw // 2
                    match_y = BAR_TOP + best_loc[1] + best_th // 2
                    self.positions[name] = (match_x, match_y, best_val)
                    missing.discard(name)
                    newly_found += 1
                    print(f" Found after scroll: {name} ({best_val:.2f})")

            if not missing:
                break

        # Scroll back to the beginning to reset the bar to initial position
        if scroll_attempts > 0:
            adb_swipe(600, 1020, 1400, 1020, 300)
            time.sleep(0.3)

        print(f"Total after scroll: {len(self.positions)}/{len(self.templates)}")

    def select(self, troop_name, delay=0.15):
        """
        Selects a troop by clicking on its position.

        Returns:
            True if found and clicked, False otherwise.
        """
        if troop_name not in self.positions:
            # Do not spam warnings
            return False

        x, y, conf = self.positions[troop_name]
        adb_tap(x, y, delay=delay)
        return True

    def get_position(self, troop_name):
        """Returns (x, y) of a troop or None."""
        if troop_name in self.positions:
            x, y, _ = self.positions[troop_name]
            return (x, y)
        return None

    def is_available(self, troop_name):
        """Checks if a troop is visible in the bar."""
        return troop_name in self.positions


# =============================================================================
# TEMPLATE EXTRACTION
# =============================================================================

def extract_bar():
    """
    Captures the troop bar and saves it.
    The user then cuts out the icons manually.
    """
    print(" Extracting the troop bar...")
    print(" Make sure you are on an enemy village screen")
    print(" (with the troop bar visible at the bottom)\n")

    img_pil = adb_screenshot()
    if img_pil is None:
        print("ERROR: Unable to capture screen")
        return

    screen = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    bar = screen[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]

    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    bar_path = os.path.join(TEMPLATES_DIR, '_barre_complete.png')
    cv2.imwrite(bar_path, bar)

    full_path = os.path.join(TEMPLATES_DIR, '_screenshot_complet.png')
    cv2.imwrite(full_path, screen)

    print(f"Bar saved: {bar_path}")
    print(f"Full screenshot: {full_path}")
    print()
    print(" NEXT STEP:")
    print(f" 1. Open {bar_path} in an image editor")
    print(" 2. Cut out each troop icon separately")
    print(f" 3. Save each icon in {TEMPLATES_DIR}/ with the correct name:")
    print(" golem.png, pekka.png, sorcier.png, sorciere.png,")
    print(" archere.png, lance_buche.png, roi.png, reine.png,")
    print(" grand_gardien.png, championne.png, rage.png, soin.png, gel.png")
    print(" 4. Delete files starting with _ (reference files)")


def auto_crop_bar():
    """Automatically crops the bar into regular slots."""
    bar_path = os.path.join(TEMPLATES_DIR, '_barre_complete.png')
    if not os.path.exists(bar_path):
        print("ERROR: Bar not found. Run --extract first.")
        return

    bar = cv2.imread(bar_path)
    h, w = bar.shape[:2]

    start_x = 115
    spacing = 85
    icon_w = 75

    print(f" Auto-cropping the bar ({w}x{h})")

    slot = 0
    x = start_x
    while x + icon_w < w:
        icon = bar[:, x:x + icon_w]
        if icon.mean() > 20:
            filename = f"slot_{slot:02d}.png"
            path = os.path.join(TEMPLATES_DIR, filename)
            cv2.imwrite(path, icon)
            print(f" Slot {slot:2d} → {filename} (x={x})")
            slot += 1
        x += spacing

    print(f"\n{slot} slots extracted.")
    print(" Rename each slot_XX.png with the actual troop name.")


# =============================================================================
# TEST
# =============================================================================

def test_finder():
    """Tests the TroopFinder on a live screenshot."""
    print("TroopFinder Test...\n")

    finder = TroopFinder()
    if not finder.templates:
        print("ERROR: No templates. Run --extract and cut icons first.")
        return

    img_pil = adb_screenshot()
    if img_pil is None:
        print("ERROR: Unable to capture screen")
        return

    # First a normal scan
    finder.update(img_pil)

    # Then with scroll if troops are missing
    if len(finder.positions) < len(finder.templates):
        print("\n Trying with bar scroll...")
        finder.update_with_scroll()

    if not finder.positions:
        print("\nERROR: No troop detected.")
        return

    print("\nSelection test (Enter to click, 'q' to quit):")
    for name in sorted(finder.positions.keys()):
        response = input(f" Select '{name}'? [Enter/q] ")
        if response.lower() == 'q':
            break
        finder.select(name)
        print(f" → Tap sent to {name}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if '--extract' in sys.argv:
        extract_bar()
    elif '--auto-crop' in sys.argv:
        auto_crop_bar()
    elif '--test' in sys.argv:
        test_finder()
    else:
        print("TroopFinder — Visual troop detection in the bar")
        print()
        print("Usage:")
        print(" python scripts/rl/troop_finder.py --extract Capture the troop bar")
        print(" python scripts/rl/troop_finder.py --auto-crop Auto-crop into slots")
        print(" python scripts/rl/troop_finder.py --test Test detection")