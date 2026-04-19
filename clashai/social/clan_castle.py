# clashai/social/clan_castle.py
# Automatic clan castle management for ClashAI V4.1.
#
# Requests reinforcement troops before each attack,
# as any human player would do.
#
# Flow:
# 1. YOLO buildings locates the clan castle in the village
# 2. Check if "FULL" is displayed above it
# 3. If not full: tap CC → template matching "Request" button
# in the bottom bar → tap "Send" (calibrated position)
# 4. 15-min cooldown between each request
#
# Setup:
# 1. uv run python -m clashai.social.clan_castle --capture
# → Captures the bar and crops the "Request" button
# → Save in templates/clan_castle/request.png
# 2. uv run python -m clashai.navigation.calibrate_ui
# → Calibrate the cdc_confirmation position (green "Send" button)
#
# Usage:
# manager = ClanCastleManager(building_detector)
# manager.request_if_needed(screenshot_fn, tap_fn)

import os
import time

import cv2
import numpy as np
from PIL import Image

from clashai.paths import PROJECT_ROOT


# =============================================================================
# CONFIGURATION
# =============================================================================

# Templates directory
CC_TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates', 'clan_castle')

# Cooldown between two requests (seconds)
REQUEST_COOLDOWN = 15 * 60

# ADB delays (seconds)
DELAY_AFTER_TAP_CC = 1.5
DELAY_AFTER_TAP_REQUEST = 1.0
DELAY_AFTER_CONFIRM = 0.8
DELAY_CLOSE = 0.5

# Template matching (bottom bar — unstable position)
MATCH_THRESHOLD = 0.60
MATCH_SCALES = [1.0, 0.9, 1.1, 0.85, 1.15]

# Building options bar zone (bottom of screen)
BAR_TOP = 880
BAR_BOTTOM = 1080
BAR_LEFT = 0
BAR_RIGHT = 1920

# Zone above the castle to detect "FULL"
FULL_TEXT_OFFSET_Y = -80
FULL_TEXT_ZONE_W = 140
FULL_TEXT_ZONE_H = 40
FULL_TEXT_WHITE_THRESHOLD = 200

# Fallback position for "Send" (if not calibrated)
# Estimated from screenshot: green button center-right of the popup
FALLBACK_CONFIRM_POS = (850, 575)


# =============================================================================
# TEMPLATE MATCHING
# =============================================================================

def _load_template(name):
    """Loads a template from templates/clan_castle/."""
    for ext in ('.png', '.jpg', '.bmp'):
        path = os.path.join(CC_TEMPLATES_DIR, f'{name}{ext}')
        if os.path.exists(path):
            tmpl = cv2.imread(path)
            if tmpl is not None:
                return tmpl
    return None


def _find_template(screenshot_cv, template_cv, region=None,
                   threshold=MATCH_THRESHOLD):
    """
    Searches for a template in a screenshot using multi-scale matching.

    Returns:
        (x, y, confidence) ADB position of the center, or None
    """
    if region:
        top, bottom, left, right = region
        search_area = screenshot_cv[top:bottom, left:right]
        offset_x, offset_y = left, top
    else:
        search_area = screenshot_cv
        offset_x, offset_y = 0, 0

    search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_cv, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]

    best_val = 0
    best_loc = None
    best_scale = 1.0

    for scale in MATCH_SCALES:
        new_w = int(tw * scale)
        new_h = int(th * scale)
        if new_w < 10 or new_h < 10:
            continue
        tmpl_scaled = cv2.resize(tmpl_gray, (new_w, new_h)) if scale != 1.0 else tmpl_gray

        sh, sw = search_gray.shape[:2]
        if new_h > sh or new_w > sw:
            continue

        result = cv2.matchTemplate(
            search_gray, tmpl_scaled, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale

    if best_val < threshold or best_loc is None:
        return None

    scaled_tw = int(tw * best_scale)
    scaled_th = int(th * best_scale)
    cx = best_loc[0] + scaled_tw // 2 + offset_x
    cy = best_loc[1] + scaled_th // 2 + offset_y

    return (cx, cy, best_val)


def _get_confirm_position():
    """
    Position of the "Send" button in the confirmation popup.
    Centered popup → stable position → calibrate_ui.
    """
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position('cdc_confirmation')
    except (ImportError, KeyError, FileNotFoundError):
        return FALLBACK_CONFIRM_POS


# =============================================================================
# CLAN CASTLE MANAGER
# =============================================================================

class ClanCastleManager:
    """
    Manages clan castle troop requests.

    - analyze_village (YOLO + CNN) → locates the CC in the village
    - Template matching → finds the "Request" button (unstable bar)
    - calibrate_ui → "Send" button (stable popup)
    - 15-min cooldown between each request
    """

    def __init__(self, models=None, verbose=True):
        self._models = models
        self.verbose = verbose
        self._last_request_time = 0
        self._total_requests = 0

        # Template for the "Request" button in the bottom bar
        self._tmpl_request = _load_template('request')

        if verbose:
            status = "ok" if self._tmpl_request is not None else "MANQUANT"
            models_ok = "ok" if models is not None else "MANQUANT"
            print(f" CC Manager — models: {models_ok} | template request: {status}")
            if self._tmpl_request is None:
                print(f" → uv run python -m clashai.social.clan_castle --capture")

    # -----------------------------------------------------------------
    # API principale
    # -----------------------------------------------------------------

    def request_if_needed(self, screenshot_fn, tap_fn):
        """Requests troops if the cooldown has passed and the CC is not full."""
        if self._tmpl_request is None:
            return False
        if not self._cooldown_ready():
            return False

        img = screenshot_fn()
        if img is None:
            return False

        cc_pos = self._find_clan_castle(img)
        if cc_pos is None:
            if self.verbose:
                print(" WARNING: Château de clan non trouvé")
            return False

        if self._is_castle_full(img, cc_pos):
            if self.verbose:
                print(" CC FULL — no request")
            return False

        return self._do_request(cc_pos, tap_fn, screenshot_fn)

    def time_until_next_request(self):
        elapsed = time.time() - self._last_request_time
        return max(0, REQUEST_COOLDOWN - elapsed)

    # -----------------------------------------------------------------
    # Cooldown
    # -----------------------------------------------------------------

    def _cooldown_ready(self):
        if self._last_request_time == 0:
            return True
        elapsed = time.time() - self._last_request_time
        if elapsed >= REQUEST_COOLDOWN:
            return True
        if self.verbose:
            remaining = (REQUEST_COOLDOWN - elapsed) / 60
            print(f" ⏳ CC cooldown: encore {remaining:.0f}min")
        return False

    # -----------------------------------------------------------------
    # Clan castle location (YOLO)
    # -----------------------------------------------------------------

    def _find_clan_castle(self, screenshot_pil):
        if self._models is None:
            return None
        try:
            from clashai.navigation.game_loop import analyze_village
            buildings = analyze_village(screenshot_pil, self._models)
            for b in buildings:
                if b['class'] == 'clan_castle':
                    cx, cy = b['center']
                    if self.verbose:
                        print(f" CC à ({cx}, {cy}) conf={b['confidence']:.2f}")
                    return (cx, cy)
        except Exception as e:
            if self.verbose:
                print(f" WARNING: Erreur détection CC: {e}")
        return None

    # -----------------------------------------------------------------
    # "FULL" detection
    # -----------------------------------------------------------------

    def _is_castle_full(self, screenshot_pil, cc_pos):
        """
        Checks whether the "FULL" text is displayed above the CC.
        If in doubt, returns False (we attempt to request anyway).
        """
        try:
            img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
            _, w = img_cv.shape[:2]
            cx, cy = cc_pos
            x1 = max(0, cx - FULL_TEXT_ZONE_W // 2)
            x2 = min(w, cx + FULL_TEXT_ZONE_W // 2)
            y1 = max(0, cy + FULL_TEXT_OFFSET_Y - FULL_TEXT_ZONE_H // 2)
            y2 = max(0, cy + FULL_TEXT_OFFSET_Y + FULL_TEXT_ZONE_H // 2)
            if y2 <= y1 or x2 <= x1:
                return False
            zone = img_cv[y1:y2, x1:x2]
            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            # High threshold (230) to target only the white "FULL" text
            # ratio > 0.30 to avoid false positives (reflections, UI)
            ratio = np.sum(gray > 230) / max(gray.size, 1)
            if ratio > 0.30:
                if self.verbose:
                    print(f" 'PLEIN' détecté (ratio: {ratio:.1%})")
                return True
        except Exception as e:
            if self.verbose:
                print(f" WARNING: Erreur détection PLEIN: {e}")
        return False

    # -----------------------------------------------------------------
    # Request execution
    # -----------------------------------------------------------------

    def _do_request(self, cc_pos, tap_fn, screenshot_fn):
        """
        1. Tap CC → options bar
        2. Template matching → "Request" button (unstable bar)
        3. Tap "Send" (stable popup → calibrate_ui)
        """
        if self.verbose:
            print(f" Requesting CC troops...")

        # 1. Tap on the clan castle
        tap_fn(cc_pos[0], cc_pos[1])
        time.sleep(DELAY_AFTER_TAP_CC)

        # 2. Screenshot + template matching "Request" in the bar
        img = screenshot_fn()
        if img is None:
            self._close_menu(tap_fn)
            return False

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = _find_template(
            img_cv, self._tmpl_request,
            region=(BAR_TOP, BAR_BOTTOM, BAR_LEFT, BAR_RIGHT),
        )

        if result is None:
            if self.verbose:
                print(" WARNING: Bouton 'Demande' non trouvé dans la barre")
            self._close_menu(tap_fn)
            return False

        req_x, req_y, conf = result
        if self.verbose:
            print(f" 'Demande' à ({req_x}, {req_y}) conf={conf:.2f}")

        # 3. Tap "Request" → "Request Reinforcements" popup
        tap_fn(req_x, req_y)
        time.sleep(DELAY_AFTER_TAP_REQUEST)

        # 4. Tap "Send" (calibrated position — stable popup)
        confirm_pos = _get_confirm_position()
        tap_fn(confirm_pos[0], confirm_pos[1])
        time.sleep(DELAY_AFTER_CONFIRM)

        if self.verbose:
            print(f" 📤 'Envoyer' à ({confirm_pos[0]}, {confirm_pos[1]})")

        # 5. Close + cooldown
        self._close_menu(tap_fn)
        self._last_request_time = time.time()
        self._total_requests += 1

        if self.verbose:
            print(f" Troupes demandées ! (total: {self._total_requests})")
        return True

    def _close_menu(self, tap_fn):
        tap_fn(30, 540)
        time.sleep(DELAY_CLOSE)


# =============================================================================
# SETUP & TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    import subprocess
    import io

    parser = argparse.ArgumentParser(description="ClashAI CC Manager")
    parser.add_argument('--capture', action='store_true',
                        help="Capture the CC bar to create the template")
    parser.add_argument('--test', action='store_true',
                        help="Test template matching on the current screen")
    args = parser.parse_args()

    if args.capture:
        print("📸 Capture of the clan castle bar\n")
        print(" 1. Tap your clan castle in-game")
        print(" 2. Wait for the bar to appear at the bottom")
        input("\n → Press Enter when ready...")

        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0:
            print(" ERROR: ADB error")
        else:
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            os.makedirs(CC_TEMPLATES_DIR, exist_ok=True)
            bar = img_cv[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]
            bar_path = os.path.join(CC_TEMPLATES_DIR, '_barre_complete.png')
            cv2.imwrite(bar_path, bar)
            full_path = os.path.join(CC_TEMPLATES_DIR, '_screenshot.png')
            cv2.imwrite(full_path, img_cv)

            print(f"\n Bar → {bar_path}")
            print(f" Screenshot → {full_path}")
            print(f"\n Now:")
            print(f" 1. Open {bar_path}")
            print(f" 2. Crop the 'Request' button → request.png")
            print(f" 3. Save in {CC_TEMPLATES_DIR}/")
            print(f"\n Then calibrate the 'Send' button:")
            print(f" uv run python -m clashai.navigation.calibrate_ui")
            print(f" → add cdc_confirmation (green 'Send' button)")

    elif args.test:
        print("Test template matching CC\n")
        mgr = ClanCastleManager(building_detector=None)
        if mgr._tmpl_request is None:
            print(" ERROR: Template 'request' missing")
            print(f" → uv run python -m clashai.social.clan_castle --capture")
        else:
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True, timeout=5
            )
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            match = _find_template(
                img_cv, mgr._tmpl_request,
                region=(BAR_TOP, BAR_BOTTOM, BAR_LEFT, BAR_RIGHT),
            )
            if match:
                print(f" 'Request' button at ({match[0]}, {match[1]}) "
                      f"conf={match[2]:.3f}")
            else:
                print(" ERROR: Button not found (open the CC menu first)")

            confirm = _get_confirm_position()
            print(f" 'Send' position: {confirm}")

    else:
        print("ClashAI CC Manager")
        print(" --capture Capture the bar for the template")
        print(" --test Test template matching")
        print(f"\nTemplates: {CC_TEMPLATES_DIR}")
        req = _load_template('request')
        print(f" request.png: {'' if req is not None else 'ERROR: missing'}")
        confirm = _get_confirm_position()
        is_fallback = confirm == FALLBACK_CONFIRM_POS
        tag = "WARNING: fallback" if is_fallback else "calibrated"
        print(f" cdc_confirmation: {confirm} ({tag})")