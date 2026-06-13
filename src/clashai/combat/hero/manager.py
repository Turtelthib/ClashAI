# clashai/combat/hero/manager.py
# HeroAbilityManager — detect available abilities + activate them.

import time

import numpy as np

from clashai.config import HERO_NAMES, NUM_HEROES
from clashai.combat.hero.constants import (
    HERO_ABILITY_NAMES, CAPA_SUFFIX, DEPLOY_TO_SCAN_DELAY,
)


def _adb_tap(x, y, delay=0.1):
    """Tap ADB — routes through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().tap(x, y, delay=delay)


class HeroAbilityManager:
    """
    Manages detection and activation of hero abilities during combat.

    V4.4: availability is read from the YOLO troop bar detector (the same
    CNN that runs in PerceptionThread), not template matching. Once a hero
    is deployed, its slot in the bottom bar becomes a `<hero>_capa` ability
    button; the CNN detects it. A non-grayed `*_capa` detection = ability
    available; grayed = used / on cooldown.

    Workflow:
        1. During deploy, call mark_deployed() for each placed hero.
        2. During combat, call update_from_troop_bar(detections) each step.
        3. Call activate(hero_name) to tap the detected icon.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

        # Per-episode state
        self._deployed = {}
        self._deploy_time = {}
        self._activated = {}
        self._icon_positions = {}

    def reset(self):
        """Reset at the start of a new episode."""
        self._deployed = {name: False for name in HERO_NAMES}
        self._deploy_time = {}
        self._activated = {name: False for name in HERO_NAMES}
        self._icon_positions = {}

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

    def update_from_troop_bar(self, troop_bar_detections):
        """
        V4.4 — populate ability icon positions from the YOLO troop bar
        detector (PerceptionThread), replacing template matching.

        The CNN detects `<hero>_capa` classes — the ability button that
        replaces a hero's slot once deployed. A NON-grayed detection means
        the ability is available; grayed means used / on cooldown.

        Args:
            troop_bar_detections: list of dicts from TroopBarDetector.detect()
                (each has 'name', 'center', 'conf', 'is_grayed')

        Returns:
            found: list of hero names whose ability is currently available
        """
        if not troop_bar_detections:
            return list(self._icon_positions.keys())

        found = []
        for d in troop_bar_detections:
            name = d.get('name', '')
            if not name.endswith(CAPA_SUFFIX):
                continue
            hero_name = name[:-len(CAPA_SUFFIX)]   # 'roi_capa' -> 'roi'
            if hero_name not in HERO_NAMES:
                continue  # e.g. duc_draconique (not in the action space)

            if self._activated.get(hero_name, False):
                continue

            # The ability button only shows once the hero is deployed → the
            # CNN is the authoritative deploy signal for the ability.
            if not self._deployed.get(hero_name, False):
                self._deployed[hero_name] = True
                self._deploy_time.setdefault(hero_name, time.time())

            if d.get('is_grayed'):
                # Used / on cooldown → not selectable this step.
                self._icon_positions.pop(hero_name, None)
                continue

            cx, cy = d['center']
            self._icon_positions[hero_name] = (cx, cy, d.get('conf', 1.0))
            found.append(hero_name)

        if self.verbose and found:
            for name in found:
                x, y, conf = self._icon_positions[name]
                print(f" {name} ability (CNN) at ({x}, {y}) conf={conf:.2f}")

        return found

    def scan(self, screenshot_pil=None, troop_bar_detections=None):
        """
        DEPRECATED (V4.4) — template matching was removed in favor of the
        YOLO troop bar CNN. Use update_from_troop_bar() instead.

        Kept for back-compat: if `troop_bar_detections` is provided it
        delegates to update_from_troop_bar(); a bare screenshot argument
        (the old V3 call) is a no-op and returns the current positions.
        """
        if troop_bar_detections is not None:
            return self.update_from_troop_bar(troop_bar_detections)
        return list(self._icon_positions.keys())

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
        """DEPRECATED (V4.4) — template matching removed. Always False so
        legacy V3 guards (`if has_templates(): scan(...)`) become no-ops."""
        return False

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

