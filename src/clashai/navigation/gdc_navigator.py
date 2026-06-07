# scripts/rl/gdc_navigator.py
# Automatic navigation in Clan War (CW) for ClashAI.
#
# This module orchestrates a complete CW attack:
# 1. From the village → open clan menu → CW tab
# 2. Switch to the enemy map
# 3. Scroll to target n°X
# 4. Select the target → scout → confirm attack
# 5. Agent V3 takes over for combat
# 6. Return to village after combat
#
# Usage:
# navigator = GdCNavigator(models)
# success = navigator.attack_target(3) # Attack enemy n°3
#
# Usage with agent V3:
# navigator = GdCNavigator(models)
# success = navigator.navigate_to_target(3) # Just navigate
# if success:
# # Agent V3 handles the attack via environment_v3
# env = ClashEnvV3(models)
# obs, mask = env.reset() # Resumes from phase_attaque
# ...

import os
import time

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# Re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# --- UI button positions ---
# Loaded dynamically from ui_positions.json
# Calibrated via : python scripts/rl/calibrate_ui.py
def _get_ui_pos(name):
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position(name)
    except ImportError:
        defaults = {
            'chat_open': (47, 400),
            'chat_close_tap': (1400, 400),
            'gdc_open': (47, 560),
            'gdc_war_ended_see_map': (960, 700),
            'gdc_enemy_map': (1700, 540),
            'gdc_ally_map': (200, 540),
            'gdc_attack_target': (900, 660),
            'gdc_village_next': (1050, 680),
            'gdc_village_prev': (100, 680),
            'gdc_return_home': (80, 780),
            'attack_button': (80, 830),
            'start_attack': (960, 700),
            'open_profil': (40, 50),
            'close_profil': (1270, 90),
            'close_menu': (1340, 95),
            'close_popup': (1300, 100),
            'return_home': (960, 800),
        }
        return defaults.get(name, (960, 400))

# Zone where enemy target numbers appear
# (the enemy list with their #)
TARGET_LIST_ZONE = {
    'left': 100,
    'right': 1820,
    'top': 150,
    'bottom': 850,
}

# Approximate Y positions of visible targets on screen
# (approximately 5-6 targets visible at a time in the CW list)
VISIBLE_TARGETS_PER_SCREEN = 5

# Scroll speed for navigating the list
SCROLL_DISTANCE = 400
SCROLL_DURATION = 300

# Wait time between actions
WAIT_NAVIGATION = 1.5
WAIT_MENU_LOAD = 2.0
WAIT_SCROLL = 1.0
WAIT_TARGET_LOAD = 2.0
WAIT_MATCHMAKING = 4.0

MAX_RETRIES = 15


# =============================================================================
# ADB FUNCTIONS
# =============================================================================

# Re-exported from the canonical implementation in game_loop (Phase B.1).
# That version routes through WGC (fast, occlusion-proof) with ADB fallback.
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot  # noqa: E402


def _adb_tap(x, y, delay=0.15):
    """Phase C.1: routed through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().tap(x, y, delay=delay)


def _adb_swipe(x1, y1, x2, y2, duration_ms=300):
    """Phase C.1: routed through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().swipe(x1, y1, x2, y2, duration_ms=duration_ms, delay=0.5)


# =============================================================================
# OCR NUMBER DETECTION
# =============================================================================

def _detect_target_numbers(screenshot_pil):
    """
    Detects visible target numbers on the enemy CW screen.

    In CoC, each enemy has a number (#1, #2, ..., #50) displayed
    next to their name in the war list.

    Returns:
        targets: dict {number: (x_center, y_center)} of visible targets
    """
    try:
        from clashai.social.clan_chat_monitor import _init_ocr
    except ImportError:
        return {}

    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    zone = img_cv[TARGET_LIST_ZONE['top']:TARGET_LIST_ZONE['bottom'],
                   TARGET_LIST_ZONE['left']:TARGET_LIST_ZONE['right']]

    engine, etype = _init_ocr()
    if engine is None:
        return {}

    targets = {}

    if etype == 'easyocr':
        results = engine.readtext(zone, paragraph=False)
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            # Look for numbers (#1, #2, 1., 2., etc.)
            import re
            match = re.search(r'#?(\d{1,2})', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 50:
                    # Position at the center of the bbox
                    cx = int((bbox[0][0] + bbox[2][0]) / 2) + TARGET_LIST_ZONE['left']
                    cy = int((bbox[0][1] + bbox[2][1]) / 2) + TARGET_LIST_ZONE['top']
                    targets[num] = (cx, cy)

    elif etype == 'tesseract':
        import pytesseract
        data = pytesseract.image_to_data(zone, output_type=pytesseract.Output.DICT)
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
            import re
            match = re.search(r'#?(\d{1,2})', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 50:
                    x = data['left'][i] + data['width'][i] // 2 + TARGET_LIST_ZONE['left']
                    y = data['top'][i] + data['height'][i] // 2 + TARGET_LIST_ZONE['top']
                    targets[num] = (x, y)

    return targets


# =============================================================================
# GDC NAVIGATOR
# =============================================================================

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
        import cv2
        import numpy as np

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
        import cv2
        import numpy as np
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


# =============================================================================
# GDC ORCHESTRATOR
# =============================================================================

class GdCOrchestrator:
    """
    Orchestrates a complete CW attack:
    chat monitor → CW navigation → V3 agent → return to village.

    Usage:
        orchestrator = GdCOrchestrator(models)
        orchestrator.run() # Infinite loop: monitors chat and attacks
    """

    def __init__(self, models, bot_name='mini_pekka', verbose=True):
        self.models = models
        self.verbose = verbose

        from clashai.social.clan_chat_monitor import ClanChatMonitor
        self._chat_monitor = ClanChatMonitor(bot_name=bot_name, verbose=verbose)
        self._navigator = GdCNavigator(models, verbose=verbose)

    def handle_command(self, command):
        """
        Executes a command received from the chat.

        Args:
            command: dict {'type': 'attack', 'target': 3, ...}
        """
        if command['type'] == 'attack':
            target = command['target']
            if self.verbose:
                print(f"\nCommand received: attack #{target} in CW")

            success = self._navigator.attack_target(target)

            if success:
                # We are in phase_attaque → launch the V3 agent
                self._run_attack()
            else:
                if self.verbose:
                    print(f" ERROR: Navigation to target #{target} failed")

            # Return to village in all cases
            self._navigator.return_to_village()

        elif command['type'] == 'status':
            if self.verbose:
                print(" Status requested (no action)")

    def _run_attack(self):
        """
        Launches the V4 agent for an attack from phase_attaque.
        """
        if self.verbose:
            print("\n Launching V4 attack...")

        try:
            from clashai.combat.environment_v4 import ClashEnvV4
            from clashai.combat.agent_v4 import PPOAgentV4
            from clashai.combat.action_space import MAX_STEPS_SAFETY

            env = ClashEnvV4(models=self.models, verbose=self.verbose)

            # Agent: load the best checkpoint
            agent = PPOAgentV4()
            weights_dir = os.path.join(
                os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )),
                'weights', 'rl'
            )
            best_path = os.path.join(weights_dir, 'agent_v4_best.pth')
            checkpoint_path = os.path.join(weights_dir, 'agent_v4_checkpoint.pth')

            heuristic_mode = True
            for ckpt_path in [best_path, checkpoint_path]:
                if os.path.exists(ckpt_path):
                    try:
                        agent.load(ckpt_path)
                        heuristic_mode = False
                        break
                    except RuntimeError:
                        if self.verbose:
                            print(f" WARNING: Incompatible checkpoint, heuristic mode")

            # Reset (resumes from phase_attaque)
            obs, mask = env.reset()
            grid, vector = obs

            # Heuristic or RL depending on whether a checkpoint exists
            heuristic_mode = not os.path.exists(best_path) and not os.path.exists(checkpoint_path)

            if heuristic_mode:
                actions = env.get_heuristic_sequence()
                for action in actions:
                    obs, mask, reward, done, info = env.step(action)
                    if done:
                        break
            else:
                for step in range(MAX_STEPS_SAFETY):
                    action, _, _ = agent.select_action(grid, vector, mask)
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break

            if self.verbose:
                stars = info.get('stars', '?')
                pct = info.get('percentage', '?')
                print(f"\n CW result: {stars}* {pct}%")

            env.close()

        except Exception as e:
            if self.verbose:
                print(f" ERROR: V3 attack error: {e}")
                import traceback
                traceback.print_exc()

    def run(self, monitor_interval=30):
        """
        Main loop: monitors the chat and executes commands.

        Args:
            monitor_interval: seconds between each chat check
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print(" ClashAI GdC Orchestrator")
            print(f" Bot: @{self._chat_monitor.bot_name}")
            print(f" Interval: {monitor_interval}s")
            print(f"{'='*50}\n")

        self._chat_monitor.monitor_loop(
            classify_screen_fn=self._navigator._classify_screen,
            models=self.models,
            callback=self.handle_command,
            interval=monitor_interval,
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClashAI GdC Navigator")
    parser.add_argument('--attack', type=int,
                        help="Attack target n°X in CW")
    parser.add_argument('--navigate', type=int,
                        help="Navigate to target without attacking")
    parser.add_argument('--monitor', action='store_true',
                        help="Start chat monitoring")
    parser.add_argument('--bot-name', type=str, default='mini_pekka')
    parser.add_argument('--interval', type=int, default=30)

    args = parser.parse_args()

    # Load models
    current_dir = os.path.dirname(os.path.abspath(__file__))
        
    from clashai.navigation import game_loop
    models = game_loop.load_models()

    if args.attack:
        nav = GdCNavigator(models)
        success = nav.attack_target(args.attack)
        if success:
            print("Attack phase reached — V3 agent can take over")
        else:
            print("ERROR: Navigation failed")

    elif args.navigate:
        nav = GdCNavigator(models)
        if nav.navigate_to_war_map():
            success = nav.select_target(args.navigate)
            if success:
                print(f"Target #{args.navigate} selected")
            else:
                print(f"ERROR: Target #{args.navigate} not found")

    elif args.monitor:
        orchestrator = GdCOrchestrator(models, bot_name=args.bot_name)
        orchestrator.run(monitor_interval=args.interval)

    else:
        print("Usage:")
        print(" --attack 3 Attack target #3 in CW")
        print(" --navigate 5 Navigate to target #5 (without attacking)")
        print(" --monitor Monitor chat and execute commands")
        print(" --bot-name X Bot name (default: mini_pekka)")
        print(" --interval N Monitoring interval in seconds")