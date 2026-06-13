# clashai/combat/hero/manager.py
# HeroAbilityManager — detect available abilities + activate them.

import os
import time

import cv2
import numpy as np

from clashai.config import HERO_NAMES, NUM_HEROES
from clashai.combat.hero.constants import (
    TEMPLATES_DIR, MATCH_THRESHOLD, HERO_ABILITY_NAMES,
    ABILITY_ZONE_TOP, ABILITY_ZONE_BOTTOM, ABILITY_ZONE_LEFT, ABILITY_ZONE_RIGHT,
    DEPLOY_TO_SCAN_DELAY, SCAN_COOLDOWN,
)
from clashai.combat.hero.template_match import _match_template_multiscale


def _adb_tap(x, y, delay=0.1):
    """Tap ADB — routes through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().tap(x, y, delay=delay)


class HeroAbilityManager:
    """
    Manages detection and activation of hero abilities during combat.

    Uses template matching to dynamically find ability icons on screen
    — no hardcoded positions.

    Workflow:
        1. During deploy, call mark_deployed() for each placed hero.
        2. During combat, call scan(screenshot) to detect icons.
        3. Call activate(hero_name) to tap the detected icon.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

        # Ability icon templates
        self._templates = {}
        self._load_templates()

        # Per-episode state
        self._deployed = {}
        self._deploy_time = {}
        self._activated = {}
        self._icon_positions = {}
        self._last_scan_time = 0

    def _load_templates(self):
        """Loads ability icon templates from hero_ability_templates/."""
        if not os.path.exists(TEMPLATES_DIR):
            if self.verbose:
                print("WARNING: hero_ability_templates/ directory not found")
                print(" Run: python scripts/rl/hero_ability.py --extract")
            return

        count = 0
        for hero in HERO_NAMES:
            filename = f"ability_{hero}.png"
            path = os.path.join(TEMPLATES_DIR, filename)
            if os.path.exists(path):
                tmpl = cv2.imread(path)
                if tmpl is not None:
                    self._templates[hero] = tmpl
                    count += 1

        if self.verbose:
            if count > 0:
                print(f"{count} ability templates loaded: "
                      f"{sorted(self._templates.keys())}")
            else:
                print(f"WARNING: No ability templates found in {TEMPLATES_DIR}")

    def reset(self):
        """Reset at the start of a new episode."""
        self._deployed = {name: False for name in HERO_NAMES}
        self._deploy_time = {}
        self._activated = {name: False for name in HERO_NAMES}
        self._icon_positions = {}
        self._last_scan_time = 0

    def mark_deployed(self, hero_name):
        """
        Marks a hero as deployed on the battlefield.
        Note: prince_gargouille is ignored (no ability).
        """
        if hero_name not in HERO_NAMES:
            return

        if not self._deployed.get(hero_name, False):
            self._deployed[hero_name] = True
            self._deploy_time[hero_name] = time.time()

            if self.verbose:
                ability = HERO_ABILITY_NAMES.get(hero_name, '?')
                print(f" {hero_name} deployed (ability: {ability})")

    def scan(self, screenshot_pil):
        """
        Scans a mid-combat screenshot to detect ability icons.
        Updates internal icon positions.

        Args:
            screenshot_pil: PIL Image of the ongoing combat

        Returns:
            found: list of hero names whose icon is visible
        """
        now = time.time()

        # Cooldown between scans
        if now - self._last_scan_time < SCAN_COOLDOWN:
            return list(self._icon_positions.keys())

        self._last_scan_time = now

        if not self._templates:
            return []

        # Convert and crop the ability zone
        screen = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                       ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]

        found = []

        for hero_name, template in self._templates.items():
            # Only search for deployed heroes that haven't been activated yet
            if not self._deployed.get(hero_name, False):
                continue
            if self._activated.get(hero_name, False):
                continue

            # Check post-deploy delay
            deploy_t = self._deploy_time.get(hero_name, now)
            if now - deploy_t < DEPLOY_TO_SCAN_DELAY:
                continue

            best_val, best_loc, best_tw, best_th = _match_template_multiscale(
                zone, template
            )

            if best_val >= MATCH_THRESHOLD and best_loc is not None:
                # Convert zone position → ADB coordinates
                x_adb = ABILITY_ZONE_LEFT + best_loc[0] + best_tw // 2
                y_adb = ABILITY_ZONE_TOP + best_loc[1] + best_th // 2
                self._icon_positions[hero_name] = (x_adb, y_adb, best_val)
                found.append(hero_name)

        if self.verbose and found:
            for name in found:
                x, y, conf = self._icon_positions[name]
                print(f" {name} ability detected "
                      f"at ({x}, {y}) conf={conf:.2f}")

        return found

    def get_available_abilities(self):
        """
        Returns heroes whose ability is available.

        Conditions:
            - Hero deployed
            - Ability not yet used
            - Icon detected by the last scan
        """
        available = []
        for name in HERO_NAMES:
            if not self._deployed.get(name, False):
                continue
            if self._activated.get(name, False):
                continue
            if name in self._icon_positions:
                available.append(name)
        return available

    def get_ability_mask(self):
        """
        Binary mask (NUM_HEROES,) for PPO action masking.
        1.0 = ability available and detected, 0.0 otherwise.
        """
        available = set(self.get_available_abilities())
        mask = np.zeros(NUM_HEROES, dtype=np.float32)
        for i, name in enumerate(HERO_NAMES):
            if name in available:
                mask[i] = 1.0
        return mask

    def activate(self, hero_name, adb_tap_fn=None):
        """
        Activates a hero's ability by tapping on its detected icon.

        Args:
            hero_name: str ('roi', 'reine', etc.)
            adb_tap_fn: callable(x, y) — if None, uses internal _adb_tap

        Returns:
            success: bool
        """
        tap_fn = adb_tap_fn or _adb_tap

        if hero_name not in HERO_NAMES:
            if self.verbose:
                print(f" WARNING: {hero_name} is not a hero with an ability")
            return False

        if not self._deployed.get(hero_name, False):
            if self.verbose:
                print(f" WARNING: {hero_name} not deployed")
            return False

        if self._activated.get(hero_name, False):
            if self.verbose:
                print(f" WARNING: {hero_name}'s ability already used")
            return False

        if hero_name not in self._icon_positions:
            if self.verbose:
                print(f" WARNING: {hero_name}'s icon not detected — "
                      f"re-run scan()")
            return False

        x, y, conf = self._icon_positions[hero_name]

        if self.verbose:
            ability = HERO_ABILITY_NAMES.get(hero_name, '?')
            print(f" {ability} ({hero_name}) "
                  f"→ tap ({x}, {y}) conf={conf:.2f}")

        tap_fn(x, y)
        time.sleep(0.3)

        self._activated[hero_name] = True
        # Remove from icon_positions to avoid tapping again
        del self._icon_positions[hero_name]

        return True

    def get_status_vector(self):
        """
        Status vector (NUM_HEROES,) for PPO observation.

        0.0  = not deployed
        0.25 = deployed, icon not yet searched
        0.5  = deployed, icon not found by scan
        0.75 = deployed, ability detected and available
        1.0  = ability activated
        """
        status = np.zeros(NUM_HEROES, dtype=np.float32)
        now = time.time()

        for i, name in enumerate(HERO_NAMES):
            if not self._deployed.get(name, False):
                status[i] = 0.0
            elif self._activated.get(name, False):
                status[i] = 1.0
            elif name in self._icon_positions:
                status[i] = 0.75
            else:
                deploy_t = self._deploy_time.get(name, now)
                if now - deploy_t < DEPLOY_TO_SCAN_DELAY:
                    status[i] = 0.25
                else:
                    status[i] = 0.5

        return status

    def num_deployed(self):
        return sum(1 for v in self._deployed.values() if v)

    def is_deployed(self, hero_name):
        return self._deployed.get(hero_name, False)

    def num_activated(self):
        return sum(1 for v in self._activated.values() if v)

    def has_templates(self):
        return len(self._templates) > 0

    # -----------------------------------------------------------------
    # V4: YOLO positions of heroes on the battlefield
    # -----------------------------------------------------------------
    def update_battlefield_positions(self, hero_positions_named: dict):
        """
        Updates hero positions on the battlefield via YOLO.

        Args:
            hero_positions_named: dict {hero_name: (x, y)} from CombatObserver
        """
        self._battlefield_positions = hero_positions_named
        # Automatically mark as deployed if YOLO detects the hero
        for name in hero_positions_named:
            if name in HERO_NAMES and not self._deployed.get(name, False):
                self.mark_deployed(name)
                if self.verbose:
                    print(f" {name} detected by YOLO → marked as deployed")

    def get_hero_position(self, hero_name):
        """Returns the YOLO position (x, y) of a hero, or None."""
        return getattr(self, '_battlefield_positions', {}).get(hero_name)

    def heroes_near_center(self, village_center, radius=250):
        """Returns heroes close to the village center (good time to activate ability)."""
        import math
        result = []
        positions = getattr(self, '_battlefield_positions', {})
        for name, (hx, hy) in positions.items():
            if name not in HERO_NAMES:
                continue
            dist = math.sqrt((hx - village_center[0])**2 + (hy - village_center[1])**2)
            if dist < radius:
                result.append((name, dist))
        return result

