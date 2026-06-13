# clashai/navigation/gdc/navigator.py
# GdCNavigator — navigate the Clan War UI, select a target, launch the attack.

import time

import cv2
import numpy as np

from clashai.navigation.gdc.constants import (
    _get_ui_pos, MAX_RETRIES,
    WAIT_NAVIGATION, WAIT_MENU_LOAD, WAIT_MATCHMAKING,
)
from clashai.navigation.gdc.adb_io import _adb_screenshot, _adb_tap


class GdCNavigator:
    """
    Navigates the Clan War interface and selects a target.
    """

    def __init__(self, models, verbose=True):
        self.models = models
        self.verbose = verbose

        # Imports from game_loop
        from clashai.navigation import game_loop as gl
        self._classify_screen = gl.classify_screen
        self._adb_screenshot_gl = gl.adb_screenshot

    def _get_screen_state(self):
        """Returns (state, confidence, img_pil)."""
        img = self._adb_screenshot_gl()
        if img is None:
            return None, 0.0, None
        state, conf = self._classify_screen(img, self.models)
        return state, conf, img

    def _navigate_to_state(self, target, max_retries=MAX_RETRIES):
        """Generic navigation toward a target state."""
        for attempt in range(max_retries):
            state, conf, img = self._get_screen_state()
            if state is None:
                time.sleep(1)
                continue

            if self.verbose and attempt % 3 == 0:
                print(f" GdC nav: {state} ({conf:.0%}) → target: {target}")

            if state == target:
                return True, img

            # Contextual navigation
            if state == 'village_home':
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)
            elif state == 'chat_clan':
                # Close the chat then open the clan menu
                _adb_tap(*_get_ui_pos('chat_close_tap'))
                time.sleep(0.5)
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)
            elif state == 'gdc_ally':
                if target == 'gdc_enemy':
                    _adb_tap(*_get_ui_pos('gdc_enemy_map'))
                    time.sleep(WAIT_NAVIGATION)
                elif target == 'village_home':
                    _adb_tap(*_get_ui_pos('gdc_return_home'))
                    time.sleep(WAIT_NAVIGATION)
                else:
                    _adb_tap(*_get_ui_pos('gdc_war_ended_see_map'))
                    time.sleep(WAIT_NAVIGATION)
            elif state == 'gdc_enemy':
                if target == 'village_home':
                    _adb_tap(*_get_ui_pos('gdc_return_home'))
                    time.sleep(WAIT_NAVIGATION)
                elif target == 'gdc_ally':
                    _adb_tap(*_get_ui_pos('gdc_ally_map'))
                    time.sleep(WAIT_NAVIGATION)
                elif target == 'phase_attaque':
                    return True, img
                else:
                    return True, img
            elif state == 'phase_attaque':
                return True, img
            elif state == 'resultats_attaque':
                _adb_tap(*_get_ui_pos('return_home'))
                time.sleep(WAIT_NAVIGATION)
            elif state == 'profil':
                _adb_tap(*_get_ui_pos('close_profil'))
                time.sleep(0.5)
                _adb_tap(1800, 500)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'menu_boutique':
                _adb_tap(*_get_ui_pos('close_menu'))
                time.sleep(WAIT_NAVIGATION)
            elif state == 'popup_offre':
                _adb_tap(*_get_ui_pos('close_popup'))
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chargement':
                time.sleep(2)
            else:
                _adb_tap(960, 400)
                time.sleep(WAIT_NAVIGATION)

        return False, None

    def navigate_to_war_map(self):
        """
        From any screen, navigates to the enemy CW map.

        Handles the "war ended" screen that appears when the last
        CW is over — automatically clicks "View map".

        Returns:
            success: bool
        """
        if self.verbose:
            print("\n Navigation vers la carte GdC ennemie...")

        # Step 1: Go to village if not already on a CW screen
        state, _, _ = self._get_screen_state()
        if state not in ('village_home', 'gdc_ally', 'gdc_enemy', 'gdc_ended'):
            success, _ = self._navigate_to_state('village_home')
            if not success:
                if self.verbose:
                    print(" ERROR: Unable to return to village")
                return False

        # Step 2: Open the CW menu from the village
        state, _, _ = self._get_screen_state()
        if state == 'village_home':
            if self.verbose:
                print(" Opening CW menu...")
            _adb_tap(*_get_ui_pos('gdc_open'))
            time.sleep(WAIT_MENU_LOAD)

        # Step 3: Handle possible screens after opening
        for attempt in range(MAX_RETRIES):
            state, conf, img = self._get_screen_state()

            if self.verbose and attempt % 2 == 0:
                print(f" GdC nav: {state} ({conf:.0%}) → target: gdc_enemy")

            if state == 'gdc_enemy':
                if self.verbose:
                    print(" Enemy CW map reached")
                return True

            elif state == 'gdc_ally':
                # On the ally map → switch to enemies
                if self.verbose:
                    print(" Ally map → switching to enemies")
                _adb_tap(*_get_ui_pos('gdc_enemy_map'))
                time.sleep(WAIT_NAVIGATION)

            elif state == 'chat_clan':
                # Chat opened instead of the CW menu
                _adb_tap(*_get_ui_pos('chat_close_tap'))
                time.sleep(0.5)
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)

            elif state == 'village_home':
                # Unexpected return to village → retry
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)

            elif state == 'chargement':
                time.sleep(2)

            elif state == 'gdc_ended':
                # "War ended" screen → click "View map"
                if self.verbose:
                    print(" War ended → click 'View map'")
                _adb_tap(*_get_ui_pos('gdc_war_ended_see_map'))
                time.sleep(WAIT_NAVIGATION)

            else:
                # Unknown state → try the "View map" button as fallback
                if self.verbose:
                    print(f"  Unknown state ({state} {conf:.0%}) "
                          f"→ trying 'View map' button")
                _adb_tap(*_get_ui_pos('gdc_war_ended_see_map'))
                time.sleep(WAIT_NAVIGATION)

        if self.verbose:
            print(" ERROR: CW navigation failed after max retries")
        return False

    def select_target(self, target_number):
        """
        Selects target #X on the enemy map.

        Reliable OCR-free method:
        1. Tap on a village to open a popup
        2. Press "previous" (←) many times → arrives at #1
        3. Press "next" (→) exactly (N-1) times → arrives at #N

        No OCR = no reading errors.

        Args:
            target_number: int (1-50)

        Returns:
            success: bool
        """
        if self.verbose:
            print(f"\nRecherche de la cible #{target_number}...")

        # We must be on gdc_enemy
        state, _, _ = self._get_screen_state()
        if state != 'gdc_enemy':
            if self.verbose:
                print(f" WARNING: Not on the enemy map (state: {state})")
            return False

        # --- Step 1: Open a popup by tapping on a village ---
        village_tap_positions = [
            (700, 450), (500, 400), (900, 500),
            (600, 350), (800, 550), (960, 400),
        ]

        popup_opened = False
        for tx, ty in village_tap_positions:
            _adb_tap(tx, ty)
            time.sleep(1.0)

            img = _adb_screenshot()
            if img is not None and self._check_attack_popup(img):
                popup_opened = True
                break

        if not popup_opened:
            if self.verbose:
                print(" ERROR: Unable to open a village popup")
            return False

        # --- Step 2: Go to village #1 (all the way left) ---
        # Press "previous" enough times
        # Max 30 villages in classic CW, 15 in league
        max_prev = 30
        if self.verbose:
            print(f"  Returning to village #1 ({max_prev}x prev)...")

        for i in range(max_prev):
            _adb_tap(*_get_ui_pos('gdc_village_prev'))
            time.sleep(0.3)

        time.sleep(0.5)

        # --- Step 3: Advance (N-1) villages ---
        steps_needed = target_number - 1

        if steps_needed > 0:
            if self.verbose:
                print(f"  Navigation: {steps_needed}x next → target #{target_number}")

            for i in range(steps_needed):
                _adb_tap(*_get_ui_pos('gdc_village_next'))
                time.sleep(0.4)

                # Log progress every 5 villages
                if self.verbose and (i + 1) % 5 == 0:
                    print(f" #{i + 2}...")

        time.sleep(0.5)

        # Verify that a popup is still open
        img = _adb_screenshot()
        if img is not None and self._check_attack_popup(img):
            if self.verbose:
                print(f" Target #{target_number} selected!")
            return True
        else:
            if self.verbose:
                print(" WARNING: Popup lost after navigation")
            return False

    def _check_attack_popup(self, img_pil):
        """
        Checks whether the target selection popup is displayed.
        Detects the green "Attack" button in the lower half of the screen.
        """
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        roi = img_cv[h // 2:, :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Green "Attack" button
        mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
        green_pixels = cv2.countNonZero(mask)

        return green_pixels > 500

    def _read_popup_number(self, img_pil):
        """
        Reads the target number from the selection popup.

        The popup shows "3. PlayerName" or "3. " at the top.
        We look for a number at the start of a line in the popup area.

        Returns:
            int or None
        """
        import re

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # Tight zone: just the popup title line
        # The title "N. Player" appears at ~64-74% of the height, centered
        popup_zone = img_cv[int(h * 0.64):int(h * 0.74),
                            int(w * 0.25):int(w * 0.60)]

        try:
            from clashai.social.clan_chat_monitor import _init_ocr
            engine, etype = _init_ocr()
            if engine is None:
                return None

            if etype == 'easyocr':
                results = engine.readtext(popup_zone, paragraph=False)
                for (bbox, text, conf) in results:
                    if conf < 0.2:
                        continue
                    text = text.strip()
                    # Possible patterns:
                    # "3. " → "3."
                    # "3." → "3."
                    # "#3" → "#3"
                    # "3 . name" → "3"
                    # We just look for a 1-2 digit number
                    match = re.match(r'#?(\d{1,2})', text)
                    if match:
                        num = int(match.group(1))
                        if 1 <= num <= 50:
                            return num

            elif etype == 'tesseract':
                import pytesseract
                text = pytesseract.image_to_string(popup_zone)
                for line in text.split('\n'):
                    line = line.strip()
                    match = re.match(r'#?(\d{1,2})', line)
                    if match:
                        num = int(match.group(1))
                        if 1 <= num <= 50:
                            return num

        except Exception as e:
            if self.verbose:
                print(f" WARNING: OCR popup error: {e}")

        return None

    def launch_attack(self):
        """
        From the screen with the target popup, launches the attack.
        """
        if self.verbose:
            print("  Lancement de l'attaque GdC...")

        for attempt in range(15):
            # First: check state via CNN
            img = _adb_screenshot()
            state, conf, _ = self._get_screen_state()

            if self.verbose:
                print(f" Attaque: {state} ({conf:.0%})")

            if state == 'phase_attaque':
                if self.verbose:
                    print(" Attack phase reached")
                return True

            elif state == 'prep_attaque':
                # Preparation screen → click the big "Attack" button
                if self.verbose:
                    print(" Preparation → click Attack")
                _adb_tap(*_get_ui_pos('start_attack'))
                time.sleep(WAIT_MATCHMAKING)

            elif state == 'gdc_enemy':
                # Still on the map → check if popup is visible
                if img is not None and self._check_attack_popup(img):
                    atk_pos = _get_ui_pos('gdc_attack_target')
                    if self.verbose:
                        print(f" Popup visible → click Attack "
                              f"at ({atk_pos[0]}, {atk_pos[1]})")
                    _adb_tap(*atk_pos)
                    time.sleep(WAIT_NAVIGATION)
                else:
                    if self.verbose:
                        print(" WARNING: No popup, tapping village")
                    _adb_tap(700, 450)
                    time.sleep(1.0)

            elif state == 'chargement':
                time.sleep(2)

            else:
                if self.verbose:
                    print(f"  State {state}, tap confirmation")
                _adb_tap(960, 600)
                time.sleep(WAIT_NAVIGATION)

        if self.verbose:
            print(" ERROR: Unable to reach the attack phase")
        return False

    def attack_target(self, target_number):
        """
        Full sequence: navigate → select → attack.

        Note: this method leads up to phase_attaque.
        The V3 agent (environment_v3) must take over for
        deploy + combat.

        Args:
            target_number: int (1-50)

        Returns:
            success: bool (True if ready to attack)
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print(f" CW: Attack target #{target_number}")
            print(f"{'='*50}")

        # 1. Navigate to the enemy map
        if not self.navigate_to_war_map():
            return False

        # 2. Select the target
        if not self.select_target(target_number):
            # Return to village on failure
            self._navigate_to_state('village_home')
            return False

        # 3. Launch the attack
        if not self.launch_attack():
            self._navigate_to_state('village_home')
            return False

        if self.verbose:
            print(f"\n Ready to attack target #{target_number}!")
            print(" → V3 agent takes over for combat")

        return True

    def return_to_village(self):
        """Return to village after a CW attack."""
        return self._navigate_to_state('village_home')
