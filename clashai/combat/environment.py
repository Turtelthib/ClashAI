# scripts/rl/environment_v3.py
# V3 Environment for ClashAI — reactive mid-combat AI.
#
# Major change vs V2:
# The episode now has 2 phases:
#
# DEPLOY phase (same as V2):
# The agent places its troops one by one, waits, casts spells.
# Ends when the agent does "done".
#
# COMBAT phase (new):
# The agent continues making decisions DURING combat:
# - Activate hero abilities (king's rage, queen cloak, etc.)
# - Cast remaining spells (with mid-combat screenshot → targeting)
# - Observe (wait_combat = wait 2-3s and re-screenshot)
# Ends when combat is over (results screen) or max steps.
#
# Usage:
# env = ClashEnvV3(models)
# obs, mask = env.reset()
# while True:
# action = agent.select_action(obs, mask)
# obs, mask, reward, done, info = env.step(action)
# if done: break

import time
import random

import numpy as np
import cv2

# Project imports

from clashai.combat.state_encoder import encode_state, find_best_attack_side
from clashai.perception.deploy_zone import (get_full_perimeter_positions)
from clashai.perception.reward_reader import read_attack_results
from clashai.combat.spell_caster import SpellCaster
from clashai.perception.troop_counter import read_troop_counts
from clashai.combat.combat_observer import CombatObserver, COMBAT_FEATURES_SIZE
from clashai.combat.hero_ability import HeroAbilityManager, HERO_NAMES

from clashai.combat.agent import (
    TROOP_TYPES, NUM_TROOP_TYPES, NUM_POSITIONS,
    TOTAL_ACTIONS, ACTION_WAIT_SHORT, ACTION_WAIT_LONG, ACTION_DONE,
    ACTION_WAIT_COMBAT, MAX_STEPS_PER_EPISODE, MAX_COMBAT_STEPS,
    GRID_CHANNELS, GRID_SIZE, VILLAGE_FEATURES, VECTOR_SIZE,
    decode_action, compute_action_mask, get_initial_troop_counts,
    get_troop_counts_from_finder, TROOP_NAME_TO_IDX,
)

# Optional zoom import
try:
    from clashai.navigation.zoom_control import zoom_out as _zoom_out_fn
    ZOOM_AVAILABLE = True
except (ImportError, OSError):
    ZOOM_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Screen / timing constants moved to clashai/config/ — see Phase A migration.
# Re-imported here under the same names so existing call sites keep working.
from clashai.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    WAIT_DECORATIONS, WAIT_BATTLE_MAX, WAIT_BATTLE_CHECK,
    WAIT_RESULT_SCREEN, WAIT_NAVIGATION, WAIT_MATCHMAKING,
)

# Delays between actions
DELAY_DEPLOY = 0.05
DELAY_SWITCH_TROOP = 0.15
DELAY_WAIT_SHORT = 0.5
DELAY_WAIT_LONG = 2.0
DELAY_WAIT_COMBAT = 2.5
DELAY_ABILITY = 0.3

MAX_NAV_RETRIES = 20

# Reward
REWARD_STAR_BONUS = 100
REWARD_ZERO_STAR_PENALTY = -50
REWARD_THREE_STAR_BONUS = 50
REWARD_FIRST_STAR_BONUS = 50

# Smart retreat
#
# Only reliable method: low threshold.
# If green bars <= GREEN_DEAD_THRESHOLD for N checks → troops dead.
# Green bars turning orange (injured troops) do NOT count
# as dead — an injured troop is still fighting.
#
# IMPORTANT: the "stable plateau method" was removed because it caused
# catastrophic false positives. In normal combat, green bars naturally
# decrease (green → orange when injured), which the plateau
# wrongly interpreted as "troops dead". Result: surrender at 88% 2.
#
GREEN_DEAD_THRESHOLD = 2
NO_TROOPS_CHECKS_THRESHOLD = 3
NO_TROOPS_MIN_WAIT = 5.0
# When 0 troops are detected during _wait_for_battle_end, the game will finish
# on its own in a few seconds — no need to wait 30s.

# Reward shaping V3
REWARD_ABILITY_TIMING_GOOD = 3.0
REWARD_ABILITY_TIMING_BAD = -2.0
REWARD_FREEZE_ON_INFERNO = 5.0


# =============================================================================
# ENVIRONNEMENT V3
# =============================================================================

class ClashEnvV3:
    """
    Bi-phase environment for Clash of Clans.

    Phase 1 (DEPLOY): The agent places its troops.
    Phase 2 (COMBAT): The agent reacts to the ongoing combat.
    """

    def __init__(self, models, verbose=True):
        self.models = models
        self.verbose = verbose

        # game_loop imports
        from clashai.navigation import game_loop as gl
        self._classify_screen = gl.classify_screen
        self._analyze_village = gl.analyze_village
        self._adb_screenshot = gl.adb_screenshot
        self._adb_tap = gl.adb_tap
        self._buttons = gl.BUTTONS

        # Load calibrated UI positions (once)
        try:
            from clashai.navigation.calibrate_ui import get_position
            self._ui = {
                'chat_open': get_position('chat_open'),
                'chat_close': get_position('chat_close_tap'),
                'close_profil': get_position('close_profil'),
                'close_menu': get_position('close_menu'),
                'close_popup': get_position('close_popup'),
                'gdc_return': get_position('gdc_return_home'),
                'ff_button': get_position('ff_button'),
                'confirm_ff': get_position('confirm_ff'),
            }
        except ImportError:
            self._ui = {
                'chat_open': (47, 400),
                'chat_close': (1400, 400),
                'close_profil': (1270, 90),
                'close_menu': (1340, 95),
                'close_popup': (1300, 100),
                'gdc_return': (80, 780),
                'ff_button': (1850, 550),
                'confirm_ff': (700, 550),
            }

        # Episode state
        self._grid = None
        self._features = None
        self._buildings = None
        self._remaining_troops = None
        self._deploy_map = None
        self._step_count = 0
        self._deploy_positions = None
        self._spell_positions = None
        self._village_center = None
        self._last_troop_name = None
        self._episode_count = 0

        # Phase tracking (NEW V3)
        self._phase = 'deploy'
        self._combat_step_count = 0
        self._combat_features = None

        # Smart retreat
        self._no_troops_count = 0

        # V3 modules
        self._troop_detector = self._try_load_troop_detector()
        self._combat_observer = CombatObserver(verbose=self.verbose,
                                                troop_detector=self._troop_detector)
        self._hero_manager = HeroAbilityManager(verbose=self.verbose)
        self._spell_caster = SpellCaster(verbose=self.verbose)

        # TroopFinder — uses YOLO detector if loaded, template matching as fallback
        from clashai.perception.troop_finder import TroopFinder
        _detector = models.get('troop_bar_detector') if models else None
        self._troop_finder = TroopFinder(detector=_detector)

        # Reward shaping
        self._shaping_history = []
        self._tanks_deployed = 0
        self._troops_deployed = 0
        self._spells_deployed = 0
        self._last_deploy_pos = None
        self._step_rewards = []

        if self.verbose and type(self).__name__ == 'ClashEnvV3':
            print("\nClashEnv V3 initialisé")
            print(f" Actions : {TOTAL_ACTIONS} "
                  f"(280 deploy + 3 ctrl + 5 abilities + 1 wait_combat)")
            print(f" Vector : {VECTOR_SIZE} dims")
            print(" Phases : deploy → combat")
            print(f" Max steps : {MAX_STEPS_PER_EPISODE} "
                  f"(dont {MAX_COMBAT_STEPS} combat)")

    # -----------------------------------------------------------------
    # COMPORTEMENT HUMAIN
    # -----------------------------------------------------------------

    @staticmethod
    def _try_load_troop_detector():
        """Attempts to load the YOLO TroopDetector. Returns None if unavailable."""
        try:
            from clashai.perception.troop_detector import TroopDetector, YOLO_TROOPS_PATH
            import os
            if os.path.exists(YOLO_TROOPS_PATH):
                detector = TroopDetector(verbose=True)
                print(" TroopDetector YOLO chargé (mode V4)")
                return detector
            else:
                print(f" WARNING: YOLO troupes introuvable ({YOLO_TROOPS_PATH}), fallback barres de vie")
        except ImportError:
            print(" WARNING: TroopDetector non disponible, fallback barres de vie")
        return None

    def _human_idle(self):
        """Simulates human-like behaviour between episodes."""
        wait_time = random.uniform(15, 60)
        if self.verbose:
            print(f"  Pause humaine ({wait_time:.0f}s)...")

        elapsed = 0
        while elapsed < wait_time:
            action = random.choices(
                ['wait', 'zoom_in', 'zoom_out', 'small_scroll'],
                weights=[0.5, 0.15, 0.15, 0.2], k=1
            )[0]

            if action in ('zoom_in', 'zoom_out'):
                try:
                    from clashai.navigation.zoom_control import zoom_in, zoom_out
                    fn = zoom_in if action == 'zoom_in' else zoom_out
                    fn(scrolls=random.randint(2, 5))
                except ImportError:
                    pass
                pause = random.uniform(1.5, 4.0)
            elif action == 'small_scroll':
                x1 = random.randint(400, 1500)
                y1 = random.randint(200, 600)
                dx, dy = random.randint(-150, 150), random.randint(-100, 100)
                try:
                    import subprocess
                    from clashai.paths import ADB_DEVICE as _ADB_DEV
                    subprocess.run(
                        ["adb", "-s", _ADB_DEV, "shell",
                         f"input swipe {x1} {y1} {x1+dx} {y1+dy} "
                         f"{random.randint(200, 500)}"],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass
                pause = random.uniform(2.0, 5.0)
            else:
                pause = random.uniform(2.0, 6.0)

            time.sleep(pause)
            elapsed += pause

    # -----------------------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------------------

    def _get_obs(self):
        """Builds the complete observation (grid, vector)."""
        step_norm = np.array(
            [self._step_count / MAX_STEPS_PER_EPISODE],
            dtype=np.float32
        )
        phase_indicator = np.array(
            [1.0 if self._phase == 'combat' else 0.0],
            dtype=np.float32
        )

        # Combat features (0 during deploy, updated during combat)
        combat_feats = (self._combat_features
                        if self._combat_features is not None
                        else np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32))

        # Hero ability status
        hero_status = self._hero_manager.get_status_vector()

        vector = np.concatenate([
            self._features,
            self._remaining_troops / 10.0,
            self._deploy_map,
            step_norm,
            combat_feats,
            hero_status,
            phase_indicator,
        ])

        return self._grid, vector

    def _get_mask(self):
        """Builds the action mask according to the current phase."""
        hero_mask = self._hero_manager.get_ability_mask()
        return compute_action_mask(
            self._remaining_troops,
            phase=self._phase,
            hero_ability_mask=hero_mask
        )

    # -----------------------------------------------------------------
    # NAVIGATION ADB
    # -----------------------------------------------------------------

    def _get_screen_state(self):
        img_pil = self._adb_screenshot()
        if img_pil is None:
            return None, 0.0, None
        state, confidence = self._classify_screen(img_pil, self.models)
        return state, confidence, img_pil

    def _navigate_to(self, target_state, timeout_retries=MAX_NAV_RETRIES):
        last_state = None
        stuck_count = 0
        MAX_STUCK = 4

        for attempt in range(timeout_retries):
            state, confidence, img_pil = self._get_screen_state()
            if state is None:
                time.sleep(1)
                continue

            if self.verbose and attempt % 3 == 0:
                print(f" State: {state} ({confidence:.0%}) "
                      f"[target: {target_state}]")

            if state == last_state:
                stuck_count += 1
            else:
                stuck_count = 0
            last_state = state

            if stuck_count >= MAX_STUCK:
                if self.verbose:
                    print(f" Stuck on '{state}' — waiting...")
                time.sleep(2.0)
                stuck_count = 0
                continue

            if confidence < 0.55 and state != target_state:
                time.sleep(1.0)
                continue

            if state == target_state:
                return True, img_pil

            # Navigate based on current state
            if state == 'village_home':
                self._adb_tap(*self._buttons['attaquer'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'recherche_adversaire':
                self._adb_tap(*self._buttons['trouver_partie'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'prep_attaque':
                self._adb_tap(*self._buttons['lancer_attaque'])
                time.sleep(WAIT_MATCHMAKING)
            elif state == 'resultats_attaque':
                self._return_to_village()
            elif state == 'chargement':
                time.sleep(2)
            elif state in ('gdc_ally', 'gdc_enemy', 'gdc_ended'):
                self._adb_tap(*self._ui['gdc_return'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'profil':
                self._adb_tap(*self._ui['close_profil'])
                time.sleep(0.5)
                self._adb_tap(1800, 500)
                time.sleep(0.5)
                self._adb_tap(30, 500)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'popup_offre':
                self._adb_tap(*self._ui['close_popup'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chat_clan':
                self._adb_tap(*self._ui['chat_close'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'menu_boutique':
                self._adb_tap(*self._ui['close_menu'])
                time.sleep(WAIT_NAVIGATION)
            else:
                time.sleep(WAIT_NAVIGATION)

        return False, None

    def _zoom_out(self):
        if ZOOM_AVAILABLE:
            try:
                _zoom_out_fn(scrolls=15)
            except Exception as e:
                if self.verbose:
                    print(f" WARNING: Zoom-out failed: {e}")

    def _find_green_button(self, img_pil):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        bottom_half = img_cv[h // 2:, :]
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 100, 120), (85, 255, 255))
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best_area, best_cx, best_cy = 0, None, None
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area > 5000 and bw > 150 and bw / max(bh, 1) > 1.5:
                if area > best_area:
                    best_area = area
                    best_cx = int(centroids[i][0])
                    best_cy = int(centroids[i][1]) + h // 2
        return (best_cx, best_cy) if best_cx else None

    def _return_to_village(self, max_retries=10):
        last_state = None
        stuck_count = 0
        for attempt in range(max_retries):
            state, conf, img_pil = self._get_screen_state()
            if state == last_state:
                stuck_count += 1
            else:
                stuck_count = 0
            last_state = state
            if stuck_count >= 3:
                time.sleep(2.0)
                stuck_count = 0
                continue
            if state == 'village_home':
                if self.verbose:
                    print(" Return to village confirmed")
                return True
            elif state in ('resultats_attaque', None) and img_pil is not None:
                btn_pos = self._find_green_button(img_pil)
                if btn_pos:
                    self._adb_tap(btn_pos[0], btn_pos[1])
                else:
                    for btn_y in [800, 760, 840, 720]:
                        self._adb_tap(960, btn_y)
                        time.sleep(0.3)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chargement':
                time.sleep(2)
            elif state in ('gdc_ally', 'gdc_enemy', 'gdc_ended'):
                self._adb_tap(*self._ui['gdc_return'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'profil':
                self._adb_tap(*self._ui['close_profil'])
                time.sleep(0.5)
                self._adb_tap(1800, 500)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chat_clan':
                self._adb_tap(*self._ui['chat_close'])
                time.sleep(0.5)
                self._adb_tap(30, 540)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'menu_boutique':
                self._adb_tap(*self._ui['close_menu'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'popup_offre':
                self._adb_tap(*self._ui['close_popup'])
                time.sleep(WAIT_NAVIGATION)
            else:
                self._adb_tap(30, 540) 
                time.sleep(WAIT_NAVIGATION)
        return False

    # -----------------------------------------------------------------
    # RETRAITE (FF)
    # -----------------------------------------------------------------

    def _surrender(self):
        """
        Surrenders the battle by pressing the white flag then confirming.

        Called when the smart retreat detects 0 living troops.
        Flow: white flag → wait for popup → confirm → results screen.

        Returns:
            success: bool (True if buttons were pressed successfully)
        """
        if self.verbose:
            print(" Surrendering combat...")

        # 1. Press the white flag (FF button)
        self._adb_tap(*self._ui['ff_button'])
        time.sleep(1.0)

        # 2. Press the confirmation button
        self._adb_tap(*self._ui['confirm_ff'])
        time.sleep(0.5)

        if self.verbose:
            print(" Retreat confirmed, waiting for results screen...")

        return True

    # -----------------------------------------------------------------
    # RESCAN BARRE DE TROUPES
    # -----------------------------------------------------------------

    # Automatic rescan frequency (every N steps in deploy phase)
    RESCAN_EVERY_N_STEPS = 8

    def _rescan_troop_bar(self):
        """
        Rescans the troop bar with a fresh screenshot.

        Uses icon SATURATION to determine whether a troop
        is still available (coloured) or exhausted (greyed out).
        """
        img = self._adb_screenshot()
        if img is None:
            return

        self._troop_finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Update counters via saturation
        ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}

        for troop_name_raw, (tx, ty, conf) in self._troop_finder.positions.items():
            troop_name = ALIAS_MAP.get(troop_name_raw, troop_name_raw)
            if troop_name not in TROOP_NAME_TO_IDX:
                continue

            idx = TROOP_NAME_TO_IDX[troop_name]
            is_active = self._is_slot_active(img_cv, tx, ty)

            if not is_active:
                # Greyed out = x0 → force counter to 0
                self._remaining_troops[idx] = 0
            elif self._remaining_troops[idx] <= 0:
                # Coloured but counter at 0 → out of sync → fix
                self._remaining_troops[idx] = 1.0

        # Troops missing from TroopFinder → fully deployed
        available_names = set()
        for name_raw in self._troop_finder.positions:
            available_names.add(ALIAS_MAP.get(name_raw, name_raw))
        for i, t in enumerate(TROOP_TYPES):
            if t['name'] not in available_names and self._remaining_troops[i] > 0:
                if t['role'] != 'spell':
                    self._remaining_troops[i] = 0

        # OCR counters to refine counts
        try:
            real_counts = read_troop_counts(img, self._troop_finder)
            for name, count in real_counts.items():
                real_name = ALIAS_MAP.get(name, name)
                if real_name in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[real_name]
                    self._remaining_troops[idx] = float(count)
        except Exception:
            pass

        if self.verbose:
            remaining = int(np.sum(self._remaining_troops))
            print(f" Rescan bar: {remaining} troops remaining")

    # -----------------------------------------------------------------
    # CLEANUP REMAINING TROOPS
    # -----------------------------------------------------------------

    def _cleanup_remaining_troops(self):
        """
        Deploys all troops still available in the bar.

        Detection by SATURATION: in CoC, exhausted troop icons (x0)
        are greyed out (saturation ≈ 0-18), while troops still
        available are coloured (saturation > 40).

        Far more reliable than internal counters.
        """
        if self.verbose:
            print("\n Cleanup: rescanning the bar...")

        img = self._adb_screenshot()
        if img is None:
            return

        self._troop_finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}
        SPELL_NAMES = {t['name'] for t in TROOP_TYPES if t['role'] == 'spell'}

        # OCR rescan counters
        try:
            real_counts = read_troop_counts(img, self._troop_finder)
            for name, count in real_counts.items():
                real_name = ALIAS_MAP.get(name, name)
                if real_name in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[real_name]
                    self._remaining_troops[idx] = float(count)
        except Exception:
            pass

        if self.verbose:
            remaining = int(np.sum([
                self._remaining_troops[i] for i, t in enumerate(TROOP_TYPES)
                if t['role'] not in ('spell',)
            ]))
            print(f" Rescan bar: {remaining} troops remaining")

        # Deploy positions (spread around the center)
        center_idx = NUM_POSITIONS // 2
        spread_positions = []
        for offset in [-2, -1, 0, 1, 2]:
            p_idx = (center_idx + offset) % NUM_POSITIONS
            if self._deploy_positions and p_idx < len(self._deploy_positions):
                spread_positions.append(self._deploy_positions[p_idx])
        if not spread_positions:
            spread_positions = [self._village_center or (960, 500)]

        # Identify COLOURED (non-greyed) troops in the bar
        troops_to_deploy = []
        for troop_name_raw, (tx, ty, conf) in self._troop_finder.positions.items():
            troop_name = ALIAS_MAP.get(troop_name_raw, troop_name_raw)

            if troop_name in SPELL_NAMES:
                continue
            if troop_name not in TROOP_NAME_TO_IDX:
                continue

            # Saturation test: coloured = still available
            if self._is_slot_active(img_cv, tx, ty):
                idx = TROOP_NAME_TO_IDX[troop_name]
                role = TROOP_TYPES[idx]['role']
                count = max(int(self._remaining_troops[idx]), 1)
                troops_to_deploy.append((troop_name_raw, troop_name, role, count))
                if self.verbose:
                    print(f" {troop_name} ({role}) x{count}")

        if not troops_to_deploy:
            if self.verbose:
                print(" Cleanup: nothing to deploy")
            return

        total = sum(c for _, _, _, c in troops_to_deploy)
        if self.verbose:
            print(f" → {total} troops to deploy")

        # Deploy in tactical order: tank → ranged → melee → siege → hero
        role_order = {'tank': 0, 'ranged': 1, 'melee': 2, 'siege': 3, 'hero': 4}
        troops_to_deploy.sort(key=lambda t: role_order.get(t[2], 99))

        deployed_count = 0
        MAX_ROUNDS = 5

        for troop_name_raw, troop_name, role, count in troops_to_deploy:
            if not self._troop_finder.select(troop_name_raw):
                continue

            time.sleep(DELAY_SWITCH_TROOP)
            idx = TROOP_NAME_TO_IDX[troop_name]
            tx, ty, _ = self._troop_finder.positions[troop_name_raw]
            taps_done = 0

            # Loop: tap in batches → check saturation → stop if greyed out
            for round_i in range(MAX_ROUNDS):
                # Tap 3 times
                for tap_i in range(3):
                    pos = spread_positions[(taps_done + tap_i) % len(spread_positions)]
                    self._adb_tap(pos[0], pos[1])
                    time.sleep(DELAY_DEPLOY)
                taps_done += 3

                # Check saturation (lightweight screenshot, no TroopFinder.update)
                check_img = self._adb_screenshot()
                if check_img is None:
                    break
                check_cv = cv2.cvtColor(np.array(check_img), cv2.COLOR_RGB2BGR)

                if not self._is_slot_active(check_cv, tx, ty):
                    # Slot greyed out → all deployed
                    break

            deployed_count += taps_done
            self._remaining_troops[idx] = 0
            self._last_troop_name = None

            if self.verbose:
                status = 'greyed' if taps_done < MAX_ROUNDS * 3 else 'max reached'
                print(f" {troop_name} → {taps_done} taps ({status})")

        if self.verbose:
            remaining = int(np.sum([
                self._remaining_troops[i] for i, t in enumerate(TROOP_TYPES)
                if t['role'] not in ('spell',)
            ]))
            print(f" Cleanup done: {deployed_count} actions"
                  f" ({remaining} still in counter)")

    # Saturation threshold to distinguish active (coloured) vs exhausted (greyed out)
    SLOT_SATURATION_THRESHOLD = 40

    def _is_slot_active(self, img_cv, x, y):
        """
        Checks whether a troop slot is coloured (active) or greyed out (x0).

        In CoC:
        - Available troops: coloured icon, average saturation > 40
        - Exhausted troops (x0): greyed icon, saturation < 20

        Args:
            img_cv: full BGR image
            x, y: ADB position of the icon centre

        Returns:
            True if the troop is still available
        """
        h, w = img_cv.shape[:2]
        ix = int(x * w / SCREEN_WIDTH)
        iy = int(y * h / SCREEN_HEIGHT)

        # Sampling zone: icon centre (avoids the x0 text at the top)
        sample_half_w = int(15 * w / SCREEN_WIDTH)
        sample_top = int(20 * h / SCREEN_HEIGHT)
        sample_bot = int(5 * h / SCREEN_HEIGHT)

        y1 = max(0, iy - sample_top)
        y2 = min(h, iy + sample_bot)
        x1 = max(0, ix - sample_half_w)
        x2 = min(w, ix + sample_half_w)

        region = img_cv[y1:y2, x1:x2]
        if region.size == 0:
            return False

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        avg_saturation = float(np.mean(hsv[:, :, 1]))

        return avg_saturation > self.SLOT_SATURATION_THRESHOLD

    # -----------------------------------------------------------------
    # EXÉCUTION D'ACTIONS
    # -----------------------------------------------------------------

    TROOP_ALIASES = {
        'lance_buche': ['lance_buche_vide'],
    }

    def _select_troop(self, troop_name):
        if troop_name == self._last_troop_name:
            return True
        if self._troop_finder.select(troop_name):
            self._last_troop_name = troop_name
            return True
        for alias in self.TROOP_ALIASES.get(troop_name, []):
            if self._troop_finder.select(alias):
                self._last_troop_name = troop_name
                return True
        if self.verbose:
            print(f" WARNING: {troop_name} not found in the bar")
        self._last_troop_name = None
        return False

    def _execute_action(self, action_idx):
        """
        Executes an action in the game via ADB.
        Handles both phases (deploy and combat).
        """
        action_type, troop_idx, pos_idx = decode_action(action_idx)

        if action_type == 'deploy':
            troop = TROOP_TYPES[troop_idx]
            troop_name = troop['name']
            is_spell = troop['role'] == 'spell'
            tap_pos = None
            deploy_success = False

            if self._select_troop(troop_name):
                time.sleep(DELAY_SWITCH_TROOP)
                deploy_success = True

                if is_spell:
                    # Smart spells via SpellCaster
                    combat_img = self._adb_screenshot()
                    if combat_img is not None:
                        # V4: if YOLO available, observe + target via YOLO
                        if self._combat_observer.has_yolo:
                            _, raw = self._combat_observer.observe(
                                combat_img, self._village_center, phase='combat')
                            targets = self._spell_caster.analyze_from_yolo(
                                raw, self._village_center)
                        else:
                            targets = self._spell_caster.analyze_battlefield(
                                combat_img, self._village_center)
                        spell_target_map = {'soin': 'heal', 'rage': 'rage', 'gel': 'freeze'}
                        target_key = spell_target_map.get(troop_name, 'heal')
                        x, y = targets[target_key]
                    else:
                        positions = self._spell_positions
                        if positions and pos_idx < len(positions):
                            x, y = positions[pos_idx]
                        else:
                            x, y = self._village_center or (960, 500)

                    self._adb_tap(x, y)
                    tap_pos = (x, y)
                    time.sleep(0.3)
                    self._last_troop_name = None
                else:
                    # Normal troops
                    positions = self._deploy_positions
                    if positions and pos_idx < len(positions):
                        x, y = positions[pos_idx]
                        self._adb_tap(x, y)
                        time.sleep(DELAY_DEPLOY)

                    # Track heroes for abilities
                    if troop['role'] == 'hero':
                        self._hero_manager.mark_deployed(troop_name)

            # --- Counter: decrement ONLY if deploy succeeded ---
            # Before this fix, the counter decremented even when _select_troop
            # returned False → the agent thought it had placed everything but the
            # bar still showed troops. The action mask closed while troops
            # remained to be deployed.
            if deploy_success:
                self._remaining_troops[troop_idx] = max(
                    0, self._remaining_troops[troop_idx] - 1
                )
            else:
                # The troop could not be selected. Rescan the bar
                # to update TroopFinder positions.
                self._deploy_failed_count = getattr(
                    self, '_deploy_failed_count', 0) + 1
                if self._deploy_failed_count >= 3:
                    self._rescan_troop_bar()
                    self._deploy_failed_count = 0

            if pos_idx is not None:
                self._deploy_map[pos_idx] += 0.2

            if is_spell and tap_pos:
                return f"{troop_name} → ({tap_pos[0]}, {tap_pos[1]})"
            elif is_spell:
                return f"{troop_name} → (selection failed)"
            else:
                return f"{troop_name} → pos {pos_idx}"

        elif action_type == 'wait_short':
            time.sleep(DELAY_WAIT_SHORT)
            self._last_troop_name = None
            return "wait 0.5s"

        elif action_type == 'wait_long':
            time.sleep(DELAY_WAIT_LONG)
            self._last_troop_name = None
            return "wait 2.0s"

        elif action_type == 'done':
            if self._phase == 'deploy':
                # Transition deploy → combat
                return "DONE (deploy → combat)"
            else:
                # End of combat phase
                return "DONE (end active combat)"

        elif action_type == 'ability':
            # NEW V3: Hero ability activation
            hero_name = HERO_NAMES[troop_idx]

            # If icon not yet detected, attempt a fresh scan
            if hero_name not in self._hero_manager._icon_positions:
                screenshot = self._adb_screenshot()
                if screenshot is not None and self._hero_manager.has_templates():
                    self._hero_manager.scan(screenshot)

            success = self._hero_manager.activate(hero_name, self._adb_tap)
            time.sleep(DELAY_ABILITY)
            if success:
                return f"{hero_name} ability activated"
            else:
                return f"WARNING: {hero_name} ability failed (icon not found)"

        elif action_type == 'wait_combat':
            # NEW V3: Observe the combat
            time.sleep(DELAY_WAIT_COMBAT)
            # Take a fresh screenshot and update combat features
            self._update_combat_observation()
            return f"observe ({DELAY_WAIT_COMBAT}s)"

        return "???"

    def _update_combat_observation(self):
        """
        Takes a mid-combat screenshot and updates:
        - Combat features (CombatObserver)
        - Ability icon positions (HeroAbilityManager.scan)
        - Smart retreat counter (consecutive 0-troop checks)
        """
        screenshot = self._adb_screenshot()
        if screenshot is None:
            return

        # Build the remaining spells dict
        spells_remaining = {}
        for i, t in enumerate(TROOP_TYPES):
            if t['role'] == 'spell':
                spells_remaining[t['name']] = int(self._remaining_troops[i])

        features, raw_data = self._combat_observer.observe(
            screenshot,
            village_center_adb=self._village_center,
            spells_remaining=spells_remaining,
            phase=self._phase
        )

        self._combat_features = features

        # --- Smart retreat ---
        # V4: if YOLO available, count detected troops directly
        # V3 fallback: GREEN bars only (red bars are ambiguous)
        if 'yolo_detections' in raw_data:
            yolo_troops = [d for d in raw_data['yolo_detections'] if d.is_troop]
            yolo_heroes = [d for d in raw_data['yolo_detections'] if d.is_hero]
            troops_alive = len(yolo_troops) + len(yolo_heroes)
        else:
            num_green = len(raw_data.get('green_positions', []))
            num_heroes = raw_data.get('num_heroes', 0)
            troops_alive = num_green + num_heroes

        if troops_alive <= GREEN_DEAD_THRESHOLD and self._phase == 'combat':
            self._no_troops_count += 1
            if self.verbose:
                print(f" Troops below threshold: {troops_alive} "
                      f"({self._no_troops_count}/{NO_TROOPS_CHECKS_THRESHOLD})")
        else:
            self._no_troops_count = 0

        # Scan hero ability icons (template matching)
        if self._phase == 'combat' and self._hero_manager.has_templates():
            self._hero_manager.scan(screenshot)

        # V4: update YOLO hero positions
        if 'hero_positions_named' in raw_data:
            self._hero_manager.update_battlefield_positions(
                raw_data['hero_positions_named'])

    # -----------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------

    def reset(self):
        """Starts a new episode."""
        self._episode_count += 1
        self._step_count = 0
        self._remaining_troops = np.zeros(NUM_TROOP_TYPES, dtype=np.float32)
        self._deploy_map = np.zeros(NUM_POSITIONS, dtype=np.float32)
        self._last_troop_name = None
        self._phase = 'deploy'
        self._combat_step_count = 0
        self._combat_features = np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32)
        self._no_troops_count = 0

        # Reset V3 modules
        self._hero_manager.reset()
        self._deploy_failed_count = 0

        # Reset reward shaping
        self._shaping_history = []
        self._tanks_deployed = 0
        self._troops_deployed = 0
        self._spells_deployed = 0
        self._last_deploy_pos = None
        self._step_rewards = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f" EPISODE #{self._episode_count} — Reset V3")
            print(f"{'='*60}")

        # Human-like behaviour between episodes
        if self._episode_count > 1:
            self._human_idle()

        # 1. Navigate to enemy village
        success, img_pil = self._navigate_to('phase_attaque')
        if not success:
            if self.verbose:
                print("ERROR: Unable to reach an enemy village")
            self._grid = np.zeros(
                (GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
            )
            self._features = np.zeros(VILLAGE_FEATURES, dtype=np.float32)
            self._deploy_positions = [(960, 500)] * NUM_POSITIONS
            self._spell_positions = self._generate_spell_positions((960, 500))
            self._village_center = (960, 500)
            return self._get_obs(), self._get_mask()

        # 2. Wait + zoom out
        if self.verbose:
            print(f"  Waiting for decorations ({WAIT_DECORATIONS}s)...")
        time.sleep(WAIT_DECORATIONS)
        self._zoom_out()

        # 3. Fresh screenshot
        img_pil = self._adb_screenshot()
        if img_pil is None:
            self._grid = np.zeros(
                (GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
            )
            self._features = np.zeros(VILLAGE_FEATURES, dtype=np.float32)
            self._deploy_positions = [(960, 500)] * NUM_POSITIONS
            self._spell_positions = self._generate_spell_positions((960, 500))
            self._village_center = (960, 500)
            return self._get_obs(), self._get_mask()

        # 4. Detect troops in the bar
        self._troop_finder.update(img_pil)
        self._remaining_troops = get_troop_counts_from_finder(self._troop_finder)

        # 4b. OCR counters
        try:
            real_counts = read_troop_counts(img_pil, self._troop_finder)
            ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}
            for name, count in real_counts.items():
                real_name = ALIAS_MAP.get(name, name)
                if real_name in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[real_name]
                    if self._remaining_troops[idx] > 0:
                        self._remaining_troops[idx] = float(count)
            if self.verbose and real_counts:
                print(f"  OCR counters: {dict(real_counts)}")
        except Exception as e:
            if self.verbose:
                print(f" WARNING: OCR counters failed: {e}")

        if self.verbose:
            detected = [TROOP_TYPES[i]['name'] for i in range(NUM_TROOP_TYPES)
                        if self._remaining_troops[i] > 0]
            total = int(np.sum(self._remaining_troops))
            print(f" {len(detected)} types ({total} units)")

        # 5. Deploy zone
        positions_all, center_adb, zone_ok = get_full_perimeter_positions(
            img_pil, num_points=NUM_POSITIONS
        )

        # If zone not detected well → retry with extra zoom-out
        if not zone_ok or not positions_all or len(positions_all) < NUM_POSITIONS // 2:
            if self.verbose:
                print(" Zone poorly detected, retrying with extra zoom-out...")
            self._zoom_out()
            time.sleep(1.0)
            img_retry = self._adb_screenshot()
            if img_retry is not None:
                img_pil = img_retry
                positions_all, center_adb, zone_ok = get_full_perimeter_positions(
                    img_pil, num_points=NUM_POSITIONS
                )
                # Also rescan troops on the new screenshot
                self._troop_finder.update(img_pil)

        if zone_ok and positions_all and len(positions_all) >= NUM_POSITIONS // 2:
            while len(positions_all) < NUM_POSITIONS:
                positions_all.append(positions_all[-1])
            self._deploy_positions = positions_all[:NUM_POSITIONS]
        else:
            if self.verbose:
                print(" WARNING: Zone not detected, using fallback positions")
            self._deploy_positions = self._generate_fallback_positions()
            center_adb = (960, 500)

        self._village_center = center_adb
        self._spell_positions = self._generate_spell_positions(center_adb)

        # 6. YOLO+CNN
        if self.verbose:
            print(" YOLO+CNN village analysis...")
        buildings = self._analyze_village(img_pil, self.models)
        if self.verbose:
            print(f" {len(buildings)} buildings detected")

        self._buildings = buildings
        state = encode_state(buildings)
        self._grid = state['grid']
        self._features = state['features']

        # V2 SpellCaster: register dangerous defense positions
        self._spell_caster.set_defense_positions(buildings)

        return self._get_obs(), self._get_mask()

    def _generate_fallback_positions(self):
        positions = []
        cx, cy = 960, 450
        radius = 350
        for i in range(NUM_POSITIONS):
            angle = 2 * np.pi * i / NUM_POSITIONS
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle) * 0.6)
            positions.append((max(80, min(1840, x)), max(80, min(850, y))))
        return positions

    def _generate_spell_positions(self, center_adb):
        cx, cy = center_adb
        positions = []
        for row in range(5):
            for col in range(4):
                x = cx + int((col - 1.5) * 80)
                y = cy + int((row - 2.0) * 50)
                positions.append((max(100, min(1820, x)), max(100, min(850, y))))
        return positions[:NUM_POSITIONS]

    # -----------------------------------------------------------------
    # REWARD SHAPING
    # -----------------------------------------------------------------

    def _compute_shaping_reward(self, action_idx):
        """
        Reward shaping V3.

        DEPLOY phase: identical to V2 (6 tactical rules).
        COMBAT phase: new rules for abilities and spells.
        """
        action_type, troop_idx, pos_idx = decode_action(action_idx)
        reward = 0.0

        if self._phase == 'deploy':
            # === Identical V2 rules ===
            if action_type == 'deploy':
                troop = TROOP_TYPES[troop_idx]
                role = troop['role']

                # Rule 1: Tanks first
                if role == 'tank' and self._troops_deployed < 4:
                    reward += 5.0

                # Rule 2: Spells before troops
                if role == 'spell':
                    troops_left = sum(
                        self._remaining_troops[i]
                        for i, t in enumerate(TROOP_TYPES)
                        if t['role'] not in ('spell',)
                        and self._remaining_troops[i] > 0
                    )
                    if troops_left > 3:
                        reward -= 8.0
                    elif troops_left > 0:
                        reward -= 3.0

                # Rule 3: Heroes before tanks
                if role == 'hero' and self._tanks_deployed == 0:
                    reward -= 3.0

                # Rule 4: Concentration
                if role not in ('spell',) and pos_idx is not None:
                    if self._last_deploy_pos is not None:
                        dist = abs(pos_idx - self._last_deploy_pos)
                        dist = min(dist, NUM_POSITIONS - dist)
                        if dist <= 3:
                            reward += 1.0
                        elif dist >= 8:
                            reward -= 1.0

                # Compteurs
                if role == 'tank':
                    self._tanks_deployed += 1
                if role == 'spell':
                    self._spells_deployed += 1
                else:
                    self._troops_deployed += 1
                    self._last_deploy_pos = pos_idx

            elif action_type == 'wait_long':
                # Rule 5: Strategic wait
                if self._tanks_deployed > 0 and self._troops_deployed < 6:
                    reward += 3.0

            elif action_type == 'done':
                # Rule 6: Undeployed troops
                troops_remaining = sum(
                    int(self._remaining_troops[i])
                    for i, t in enumerate(TROOP_TYPES)
                    if t['role'] not in ('spell',)
                )
                if troops_remaining > 0:
                    reward -= 2.0 * troops_remaining

        elif self._phase == 'combat':
            # === New V3 rules ===

            if action_type == 'ability':
                hero_name = HERO_NAMES[troop_idx]

                # Rule 7: Ability timing
                # The king should use his rage when he is low on HP
                # (approximated by time elapsed in combat)
                combat_progress = 0.0
                if self._combat_features is not None:
                    combat_progress = self._combat_features[1]

                if hero_name == 'roi':
                    # King uses his rage mid/late combat
                    if 0.3 <= combat_progress <= 0.8:
                        reward += REWARD_ABILITY_TIMING_GOOD
                    elif combat_progress < 0.1:
                        reward += REWARD_ABILITY_TIMING_BAD

                elif hero_name == 'reine':
                    # Queen uses her cloak while defenses remain
                    if 0.2 <= combat_progress <= 0.7:
                        reward += REWARD_ABILITY_TIMING_GOOD

                elif hero_name == 'grand_gardien':
                    # Grand Warden uses his tome when troops are taking damage
                    if self._combat_features is not None:
                        hurt_ratio = self._combat_features[10]
                        if hurt_ratio > 0.3:
                            reward += REWARD_ABILITY_TIMING_GOOD + 2.0
                        elif hurt_ratio < 0.1:
                            reward += REWARD_ABILITY_TIMING_BAD

                elif hero_name in ('championne', 'prince_gargouille'):
                    # Flexible timing, small bonus if used
                    if combat_progress > 0.2:
                        reward += 1.0

            elif action_type == 'deploy' and TROOP_TYPES[troop_idx]['role'] == 'spell':
                # Rule 8: Well-targeted spell during combat → small bonus
                reward += 1.0

            elif action_type == 'wait_combat':
                # Rule 9: Observing is neutral (no penalty)
                # But observing too much without acting → slight penalty
                if self._combat_step_count > 10:
                    reward -= 0.5

        self._shaping_history.append((action_type, troop_idx, pos_idx, reward))
        return reward

    # -----------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------

    def step(self, action_idx):
        """
        Executes an action.
        Handles the deploy → combat transition automatically.
        """
        self._step_count += 1

        # Shaping reward
        shaping = self._compute_shaping_reward(action_idx)
        self._step_rewards.append(shaping)

        # Execute
        action_desc = self._execute_action(action_idx)

        if self.verbose:
            phase_tag = "" if self._phase == 'deploy' else ""
            shaping_str = f" ({shaping:+.0f})" if shaping != 0 else ""
            print(f" {phase_tag} Step {self._step_count:2d}: "
                  f"{action_desc}{shaping_str}")

        action_type, _, _ = decode_action(action_idx)

        # --- Periodic bar rescan during deploy ---
        # Icons shift when troops are placed.
        # Without rescan, TroopFinder taps the wrong spot.
        if (self._phase == 'deploy'
                and self._step_count % self.RESCAN_EVERY_N_STEPS == 0
                and action_type != 'done'):
            self._rescan_troop_bar()

        # --- Transition deploy → combat ---
        if action_type == 'done' and self._phase == 'deploy':
            # Cleanup: deploy remaining troops before combat
            self._cleanup_remaining_troops()

            self._phase = 'combat'
            self._combat_step_count = 0
            self._combat_observer.start_combat()

            if self.verbose:
                print("\n   COMBAT PHASE ")
                print(f" Heroes deployed: {self._hero_manager.num_deployed()}")
                abilities = self._hero_manager.get_available_abilities()
                if abilities:
                    print(f" Available abilities: {abilities}")

            # First combat observation
            self._update_combat_observation()

            return self._get_obs(), self._get_mask(), shaping, False, {
                'step': self._step_count,
                'phase': 'combat'
            }

        # --- During combat phase ---
        if self._phase == 'combat':
            self._combat_step_count += 1

            # Check if combat is over
            is_battle_over = self._check_battle_end()

            # Conditions de fin de phase combat
            is_done = (
                is_battle_over
                or action_type == 'done'
                or self._combat_step_count >= MAX_COMBAT_STEPS
                or self._step_count >= MAX_STEPS_PER_EPISODE
            )

            if is_done:
                combat_reward, info = self._finish_episode()
                info['shaping_total'] = sum(self._step_rewards)
                info['combat_reward'] = combat_reward
                info['combat_steps'] = self._combat_step_count
                info['abilities_used'] = self._hero_manager.num_activated()
                return self._get_obs(), self._get_mask(), combat_reward, True, info
            else:
                return self._get_obs(), self._get_mask(), shaping, False, {
                    'step': self._step_count,
                    'combat_step': self._combat_step_count,
                    'phase': 'combat'
                }

        # --- During deploy phase ---
        is_done = False
        if self._step_count >= MAX_STEPS_PER_EPISODE:
            is_done = True
        # Do NOT force end when remaining=0 — the agent/heuristic
        # must reach ACTION_DONE to transition to combat phase.
        # Otherwise spells at end of deploy kill the combat phase.

        if is_done:
            # No combat phase if forced end → passive wait
            combat_reward, info = self._finish_episode()
            info['shaping_total'] = sum(self._step_rewards)
            info['combat_reward'] = combat_reward
            return self._get_obs(), self._get_mask(), combat_reward, True, info

        return self._get_obs(), self._get_mask(), shaping, False, {
            'step': self._step_count
        }

    def _check_battle_end(self):
        """
        Checks whether the battle is over.

        Two conditions:
        1. The screen shows results (normal end)
        2. No living troops detected N times in a row (smart retreat)
           → the battle will end on its own in a few seconds
        """
        # Condition 1: results screen
        state, confidence, _ = self._get_screen_state()
        if state == 'resultats_attaque' and confidence > 0.6:
            return True
        
        # Condition 2: smart retreat (consecutive 0-troop checks)
        if self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD:
            if self.verbose:
                print(f" Smart retreat: "
                      f"0 troops for {self._no_troops_count} checks")
            return True
        
        return False

    def get_step_rewards(self):
        return self._step_rewards

    # -----------------------------------------------------------------
    # FIN D'ÉPISODE
    # -----------------------------------------------------------------

    def _finish_episode(self):
        """Waits for battle end and computes the reward."""
        if self.verbose:
            remaining = int(np.sum(self._remaining_troops))
            print(f"\n Episode over!"
                  f" ({self._step_count} steps,"
                  f" {self._combat_step_count} combat,"
                  f" {remaining} remaining,"
                  f" {self._hero_manager.num_activated()} abilities)")

        # If in combat phase, battle may already be over;
        # otherwise wait passively
        result_img = self._wait_for_battle_end()

        if result_img is not None:
            results = read_attack_results(result_img, debug=False)
            stars = results['stars']
            percentage = results['percentage']
            success = results['success']
        else:
            if self.verbose:
                print(" WARNING: Unable to read results")
            stars = 0
            percentage = 0
            success = False

        reward = self._compute_reward(stars, percentage)

        if self.verbose:
            retreat_str = " (retreat)" if self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD else ""
            print(f"\n RESULTS{retreat_str}:")
            print(f" * Stars: {stars}/3")
            print(f" Percentage: {percentage}%")
            print(f" Reward: {reward:.0f}")

        if self.verbose:
            print(" Returning to village...")
        self._return_to_village()

        info = {
            'stars': stars,
            'percentage': percentage,
            'reward': reward,
            'success': success,
            'steps': self._step_count,
            'deploy_steps': self._step_count - self._combat_step_count,
            'combat_steps': self._combat_step_count,
            'troops_remaining': int(np.sum(self._remaining_troops)),
            'abilities_used': self._hero_manager.num_activated(),
            'episode': self._episode_count,
            'early_retreat': self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD,
        }

        return reward, info

    def _wait_for_battle_end(self):
        """
        Waits for the battle to end.

        Accelerated detection via low threshold:
          If green bars <= 2 for 3 consecutive checks → troops dead.
          → Surrender (white flag + confirmation) → results in ~5s.

        NOTE: green bars naturally decrease when troops take damage
        (green → orange). That is why a very low threshold (≤2) is used
        instead of a peak ratio — an injured troop (orange bar) is still
        alive and fighting.
        """
        if self.verbose:
            print(" Waiting for battle end...")

        # If smart retreat was triggered during combat phase, surrender immediately
        surrendered = False
        if self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD:
            self._surrender()
            surrendered = True
            min_wait = NO_TROOPS_MIN_WAIT
            if self.verbose:
                print(f" Active retreat → reduced wait ({min_wait:.0f}s min)")
        else:
            min_wait = 15.0 if self._phase == 'combat' else 30.0

        start_time = time.time()
        no_troops_consecutive = 0

        while time.time() - start_time < WAIT_BATTLE_MAX:
            elapsed = time.time() - start_time

            # 1. Check screen state
            state, confidence, img_pil = self._get_screen_state()

            if self.verbose and int(elapsed) % 10 == 0:
                print(f" {elapsed:.0f}s — screen: {state} ({confidence:.0%})")

            if state == 'resultats_attaque' and elapsed >= min_wait:
                if self.verbose:
                    print(f" Battle ended after {elapsed:.0f}s")
                time.sleep(WAIT_RESULT_SCREEN)
                final_img = self._adb_screenshot()
                return final_img if final_img else img_pil

            # 2. Scan GREEN bars only
            # (orange/red = injured troops OR enemy buildings)
            if img_pil is not None and not surrendered:
                try:
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    from clashai.combat.combat_observer import detect_troop_bars, detect_hero_bars
                    green_pos, _ = detect_troop_bars(img_cv)
                    hero_pos = detect_hero_bars(img_cv)
                    green_count = len(green_pos) + len(hero_pos)

                    if self.verbose:
                        print(f" Scan: {len(green_pos)} green, "
                              f"{len(hero_pos)} heroes "
                              f"→ alive={green_count}")

                    if green_count <= GREEN_DEAD_THRESHOLD:
                        no_troops_consecutive += 1
                        if self.verbose:
                            print(f" Below threshold "
                                  f"({no_troops_consecutive}/{NO_TROOPS_CHECKS_THRESHOLD})")
                    else:
                        no_troops_consecutive = 0

                    if no_troops_consecutive >= NO_TROOPS_CHECKS_THRESHOLD:
                        if self.verbose:
                            print(f" Troops dead "
                                  f"(green<={GREEN_DEAD_THRESHOLD} "
                                  f"x{NO_TROOPS_CHECKS_THRESHOLD})")
                        self._surrender()
                        surrendered = True
                        min_wait = NO_TROOPS_MIN_WAIT

                except Exception as e:
                    if self.verbose:
                        print(f" WARNING: Troop scan failed: {e}")

            time.sleep(WAIT_BATTLE_CHECK)

        state, _, img_pil = self._get_screen_state()
        if state == 'resultats_attaque':
            return img_pil
        return None

    def _compute_reward(self, stars, percentage):
        reward = (stars * REWARD_STAR_BONUS) + percentage
        if stars >= 1:
            reward += REWARD_FIRST_STAR_BONUS
        if stars == 0:
            reward += REWARD_ZERO_STAR_PENALTY
        if stars == 3 and percentage == 100:
            reward += REWARD_THREE_STAR_BONUS
        return float(reward)

    # -----------------------------------------------------------------
    # HEURISTIQUE (baseline)
    # -----------------------------------------------------------------

    def get_heuristic_sequence(self):
        """
        Heuristic sequence V3 — fully dynamic.

        Adapts automatically to actually available troops/spells.

        DEPLOY phase: troops only (tanks → funnel → ranged → melee → siege → heroes)
        COMBAT phase: spells (with fresh screenshot per cast) + hero abilities

        Spells are in the combat phase because:
        - SpellCaster takes a screenshot per spell → precise targeting
        - We can see troops fighting → we know where to place heal/rage/freeze
        - Freeze can target infernos near troops in real time
        """
        if self._buildings:
            best_dir = find_best_attack_side(
                self._buildings, verbose=self.verbose
            )
        else:
            best_dir = 0

        center_pos = int(best_dir / 8 * NUM_POSITIONS) % NUM_POSITIONS
        positions = [(center_pos + i - 2) % NUM_POSITIONS for i in range(5)]

        actions = []
        remaining = self._remaining_troops.copy()

        def add(name, pos):
            """Adds an action if the troop is available."""
            if name not in TROOP_NAME_TO_IDX:
                return False
            idx = TROOP_NAME_TO_IDX[name]
            if remaining[idx] > 0:
                actions.append(idx * NUM_POSITIONS + pos)
                remaining[idx] -= 1
                return True
            return False

        def add_all(name, pos_list):
            """Deploys all units of a type across the given positions."""
            if name not in TROOP_NAME_TO_IDX:
                return 0
            idx = TROOP_NAME_TO_IDX[name]
            count = 0
            i = 0
            while remaining[idx] > 0:
                p = pos_list[i % len(pos_list)]
                actions.append(idx * NUM_POSITIONS + p)
                remaining[idx] -= 1
                count += 1
                i += 1
            return count

        def add_one_spell(spell_name, pos):
            """Casts ONE spell if available."""
            return add(spell_name, pos)

        # ============================================================
        # Dynamic inventory
        # ============================================================
        tanks = [t['name'] for t in TROOP_TYPES 
                 if t['role'] == 'tank' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        ranged = [t['name'] for t in TROOP_TYPES 
                  if t['role'] == 'ranged' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        melee = [t['name'] for t in TROOP_TYPES 
                 if t['role'] == 'melee' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        heroes = [t['name'] for t in TROOP_TYPES 
                  if t['role'] == 'hero' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        sieges = [t['name'] for t in TROOP_TYPES 
                  if t['role'] == 'siege' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        spells = {}
        for t in TROOP_TYPES:
            if t['role'] == 'spell':
                idx = TROOP_NAME_TO_IDX[t['name']]
                if remaining[idx] > 0:
                    spells[t['name']] = int(remaining[idx])

        total_spells = sum(spells.values())
        sp = 10

        if self.verbose:
            print(f" Inventory: "
                  f"{len(tanks)} tanks, {len(ranged)} ranged, "
                  f"{len(melee)} melee, {len(heroes)} heroes, "
                  f"{len(sieges)} siege, {total_spells} spells {spells}")

        # ============================================================
        # DEPLOY PHASE: troops only
        # ============================================================

        # 1. TANKS at the edges
        for tank_name in tanks:
            idx = TROOP_NAME_TO_IDX[tank_name]
            n = int(remaining[idx])
            if n >= 2:
                add(tank_name, positions[0])
                add(tank_name, positions[4])
                add_all(tank_name, [positions[2]])
            elif n == 1:
                add(tank_name, positions[2])

        actions.append(ACTION_WAIT_LONG)

        # 2. FUNNEL — ranged at the edges
        funnel_count = 0
        for r_name in ranged:
            idx = TROOP_NAME_TO_IDX[r_name]
            if remaining[idx] >= 2 and funnel_count < 2:
                add(r_name, positions[0])
                add(r_name, positions[4])
                funnel_count += 1

        actions.append(ACTION_WAIT_SHORT)

        # 3. RANGED in a line
        for r_name in ranged:
            add_all(r_name, positions[1:4])

        actions.append(ACTION_WAIT_LONG)

        # 4. MELEE + SIEGE at the center
        for m_name in melee:
            add_all(m_name, [positions[2], positions[1], positions[3]])
        for s_name in sieges:
            add_all(s_name, [positions[2]])

        # 5. HEROES at the center
        for h_name in heroes:
            add(h_name, positions[2])

        # → DONE: transition to combat
        actions.append(ACTION_DONE)

        # ============================================================
        # COMBAT PHASE: spells + abilities (fresh screenshot)
        #
        # Each wait_combat = screenshot + SpellCaster recalculates
        # positions in real time. Spells are targeted on what the
        # AI SEES, not on static coordinates.
        #
        # Tactical sequence:
        # 1. Observe (troops engage)
        # 2. RAGE (DPS boost during engagement)
        # 3. Observe (see damage)
        # 4. FREEZE on inferno/eagle (protect troops)
        # 5. Hero abilities (GG first = invincibility)
        # 6. HEAL (troops have taken damage)
        # 7. Observe + alternate RAGE/HEAL remaining
        # 8. Remaining hero abilities
        # ============================================================
        from clashai.combat.agent import (ACTION_ABILITY_ROI, ACTION_ABILITY_REINE,
                              ACTION_ABILITY_GG, ACTION_ABILITY_CHAMP, ACTION_ABILITY_PG)

        ABILITY_ORDER = [
            ('grand_gardien', ACTION_ABILITY_GG),
            ('roi', ACTION_ABILITY_ROI),
            ('reine', ACTION_ABILITY_REINE),
            ('championne', ACTION_ABILITY_CHAMP),
            ('prince_gargouille', ACTION_ABILITY_PG),
        ]

        # Split abilities into two waves
        wave1_abilities = []
        wave2_abilities = []
        for hero_name, ability_action in ABILITY_ORDER:
            if hero_name in heroes:
                if hero_name in ('grand_gardien', 'roi'):
                    wave1_abilities.append(ability_action)
                else:
                    wave2_abilities.append(ability_action)

        # Sort spells by tactical priority
        # Freeze = urgent (protect from infernos), Rage = boost, Heal = sustain
        spell_queue = []
        # First: 1 rage (initial boost)
        if spells.get('rage', 0) > 0:
            spell_queue.append('rage')
            spells['rage'] -= 1
        # Then: all freezes (stop infernos)
        while spells.get('gel', 0) > 0:
            spell_queue.append('gel')
            spells['gel'] -= 1
        # Then alternate heal/rage
        while any(v > 0 for v in spells.values()):
            for spell_name in ['soin', 'rage']:
                if spells.get(spell_name, 0) > 0:
                    spell_queue.append(spell_name)
                    spells[spell_name] -= 1
                    break
            else:
                # Remaining spells (other types)
                for spell_name in list(spells.keys()):
                    if spells[spell_name] > 0:
                        spell_queue.append(spell_name)
                        spells[spell_name] -= 1
                        break
                else:
                    break

        # --- Build the combat sequence ---

        # 1. Observe (troops engage defenses)
        actions.append(ACTION_WAIT_COMBAT)

        # 2. First spell (rage boost) + defensive abilities
        spell_idx = 0
        if spell_idx < len(spell_queue):
            add_one_spell(spell_queue[spell_idx], sp)
            spell_idx += 1

        for ability in wave1_abilities:
            actions.append(ability)

        # 3. Observe damage
        actions.append(ACTION_WAIT_COMBAT)

        # 4. Freeze + heal (protect and heal)
        spells_this_round = 0
        while spell_idx < len(spell_queue) and spells_this_round < 2:
            add_one_spell(spell_queue[spell_idx], sp + spells_this_round)
            spell_idx += 1
            spells_this_round += 1

        # 5. Observe
        actions.append(ACTION_WAIT_COMBAT)

        # 6. Offensive abilities
        for ability in wave2_abilities:
            actions.append(ability)

        # 7. Remaining spells (with observe between each for targeting)
        while spell_idx < len(spell_queue):
            actions.append(ACTION_WAIT_COMBAT)
            add_one_spell(spell_queue[spell_idx], sp + (spell_idx % 4))
            spell_idx += 1

        actions.append(ACTION_DONE)

        return actions

    def close(self):
        if self.verbose:
            print(f"\nEnvironment V3 closed "
                  f"after {self._episode_count} episodes")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("ClashEnv V3 — test dry-run\n")

    from clashai.combat.agent import decode_action

    print("1. Test nouvelles actions V3 :")
    test_actions = [
        0, 19, 279, 280, 281, 282,
        283, 284, 285, 286, 287, 288
    ]
    for a in test_actions:
        t, ti, pi = decode_action(a)
        if t == 'deploy':
            name = TROOP_TYPES[ti]['name']
            print(f" Action {a:3d} → {t} {name} pos {pi}")
        elif t == 'ability':
            hero = HERO_NAMES[ti]
            print(f" Action {a:3d} → {t} {hero}")
        else:
            print(f" Action {a:3d} → {t}")

    print("\n2. Test masking par phase :")
    troops = get_initial_troop_counts()

    mask_deploy = compute_action_mask(troops, phase='deploy')
    print(f" Phase deploy : {int(mask_deploy.sum())} actions valides / {TOTAL_ACTIONS}")

    hero_mask = np.array([1, 1, 0, 0, 0], dtype=np.float32)
    mask_combat = compute_action_mask(troops, phase='combat', hero_ability_mask=hero_mask)
    print(f" Phase combat : {int(mask_combat.sum())} actions valides / {TOTAL_ACTIONS}")
    print(f" - Sorts : {int(mask_combat[:280].sum())}")
    print(f" - Abilities : {int(mask_combat[283:288].sum())}")
    print(f" - Wait combat : {int(mask_combat[288])}")
    print(f" - Done : {int(mask_combat[282])}")

    print("\nTest dry-run V3 terminé !")