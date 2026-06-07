# clashai/combat/troop_manager.py
# Troop bar management for ClashAI.
#
# Responsibilities:
# - Scan the troop bar (TroopFinder + saturation check)
# - Select a troop by name or by role
# - Map V4 role → concrete troop (round-robin)
# - Cleanup: deploy all remaining troops
#
# Separated from the environment to keep files under 500 lines.

import time
import cv2
import numpy as np

from clashai.combat.action_space import (
    DEPLOY_ROLES, ROLE_TO_TROOPS, DEPLOY_SECTORS,
    SECTOR_OFFSETS, NUM_POSITIONS,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Re-imported from clashai/config/ (Phase A).
from clashai.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    DELAY_SWITCH_TROOP, DELAY_DEPLOY,
)  # noqa: E402

SLOT_SATURATION_THRESHOLD = 40
MAX_CLEANUP_ROUNDS = 5

# Template aliases
TROOP_ALIASES = {
    'lance_buche': ['lance_buche_vide'],
}
ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}


# =============================================================================
# TROOP MANAGER
# =============================================================================

class TroopManager:
    """
    Manages the troop bar: scan, selection, deploy by role.
    """

    def __init__(self, troop_finder, troop_types, troop_name_to_idx,
                 adb_screenshot_fn, adb_tap_fn, verbose=True):
        """
        Args:
            troop_finder: TroopFinder instance
            troop_types: list[dict] — V3 TROOP_TYPES
            troop_name_to_idx: dict {name: idx}
            adb_screenshot_fn: callable -> PIL Image
            adb_tap_fn: callable (x, y) -> None
            verbose: bool
        """
        self._finder = troop_finder
        self._troop_types = troop_types
        self._name_to_idx = troop_name_to_idx
        self._screenshot = adb_screenshot_fn
        self._tap = adb_tap_fn
        self.verbose = verbose

        # State
        self._last_troop_name = None
        self._deploy_failed_count = 0

        # Per-role round-robin (V4)
        self._role_cursors = {role: 0 for role in DEPLOY_ROLES}

    def reset(self):
        """Reset for a new episode."""
        self._last_troop_name = None
        self._deploy_failed_count = 0
        self._role_cursors = {role: 0 for role in DEPLOY_ROLES}

    # -----------------------------------------------------------------
    # Selection by name
    # -----------------------------------------------------------------

    def select_troop(self, troop_name):
        """Selects a troop by name from the bar."""
        if troop_name == self._last_troop_name:
            return True
        if self._finder.select(troop_name):
            self._last_troop_name = troop_name
            return True
        for alias in TROOP_ALIASES.get(troop_name, []):
            if self._finder.select(alias):
                self._last_troop_name = troop_name
                return True
        if self.verbose:
            print(f" WARNING: {troop_name} not found in troop bar")
        self._last_troop_name = None
        return False

    # -----------------------------------------------------------------
    # Selection by role (V4)
    # -----------------------------------------------------------------

    def select_next_for_role(self, role_name, remaining_troops):
        """
        Selects the next available troop for a given role.
        Round-robin among troops sharing the same role.

        Args:
            role_name: 'tank', 'ranged', 'melee', 'hero', 'siege'
            remaining_troops: array (N,) — counters

        Returns:
            (troop_idx, troop_name) or (None, None)
        """
        candidates = ROLE_TO_TROOPS.get(role_name, [])
        if not candidates:
            return None, None

        cursor = self._role_cursors.get(role_name, 0)
        n = len(candidates)

        # Search from cursor (round-robin)
        for offset in range(n):
            i = (cursor + offset) % n
            name = candidates[i]
            if name not in self._name_to_idx:
                continue
            idx = self._name_to_idx[name]
            if remaining_troops[idx] > 0:
                if self.select_troop(name):
                    self._role_cursors[role_name] = (i + 1) % n
                    return idx, name

        return None, None

    # -----------------------------------------------------------------
    # Saturation check
    # -----------------------------------------------------------------

    def is_slot_active(self, img_cv, x, y):
        """Checks if a slot is colored (available) or grayed out (exhausted)."""
        h, w = img_cv.shape[:2]
        ix = int(x * w / SCREEN_WIDTH)
        iy = int(y * h / SCREEN_HEIGHT)

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
        avg_sat = float(np.mean(hsv[:, :, 1]))
        return avg_sat > SLOT_SATURATION_THRESHOLD

    # -----------------------------------------------------------------
    # Bar rescan
    # -----------------------------------------------------------------

    def rescan(self, remaining_troops, read_counts_fn=None):
        """
        Rescans the troop bar with a fresh screenshot.

        Args:
            remaining_troops: array (N,) — modified in-place
            read_counts_fn: optional callable(img, finder) -> dict
        """
        img = self._screenshot()
        if img is None:
            return

        self._finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for name_raw, (tx, ty, conf) in self._finder.positions.items():
            name = ALIAS_MAP.get(name_raw, name_raw)
            if name not in self._name_to_idx:
                continue
            idx = self._name_to_idx[name]
            active = self.is_slot_active(img_cv, tx, ty)
            if not active:
                remaining_troops[idx] = 0
            elif remaining_troops[idx] <= 0:
                remaining_troops[idx] = 1.0

        # Troops gone from the finder
        available = {ALIAS_MAP.get(n, n) for n in self._finder.positions}
        for i, t in enumerate(self._troop_types):
            if t['name'] not in available and remaining_troops[i] > 0:
                if t['role'] != 'spell':
                    remaining_troops[i] = 0

        # Counters — pass current remaining_troops as prev_counts so OCR
        # uses them as dynamic upper bound (monotonic validation, no hardcoding)
        detector = getattr(self._finder, '_detector', None)
        if detector is not None:
            try:
                prev = {t['name']: int(remaining_troops[i])
                        for i, t in enumerate(self._troop_types)
                        if remaining_troops[i] > 0}
                counts = detector.to_counts(prev_counts=prev)
                for name, count in counts.items():
                    real_name = ALIAS_MAP.get(name, name)
                    if real_name in self._name_to_idx:
                        remaining_troops[self._name_to_idx[real_name]] = float(count)
            except Exception:
                pass
        elif read_counts_fn is not None:
            try:
                real = read_counts_fn(img, self._finder)
                for name, count in real.items():
                    real_name = ALIAS_MAP.get(name, name)
                    if real_name in self._name_to_idx:
                        remaining_troops[self._name_to_idx[real_name]] = float(count)
            except Exception:
                pass

        if self.verbose:
            total = int(np.sum(remaining_troops))
            print(f" Bar rescan: {total} troops remaining")

    # -----------------------------------------------------------------
    # Cleanup: deploy everything left
    # -----------------------------------------------------------------

    def cleanup(self, remaining_troops, deploy_positions, village_center):
        """
        Deploys all troops still in the bar (tap-until-gray).

        Args:
            remaining_troops: array — modified in-place
            deploy_positions: list[(x,y)] — deploy positions
            village_center: (x,y) — fallback
        """
        if self.verbose:
            from clashai.config.logging import section
            section("Cleanup: rescanning bar")

        img = self._screenshot()
        if img is None:
            return

        self._finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        spell_names = {t['name'] for t in self._troop_types if t['role'] == 'spell'}

        # Spread positions
        center_idx = NUM_POSITIONS // 2
        spread = []
        for off in [-2, -1, 0, 1, 2]:
            p = (center_idx + off) % NUM_POSITIONS
            if deploy_positions and p < len(deploy_positions):
                spread.append(deploy_positions[p])
        if not spread:
            spread = [village_center or (960, 500)]

        # Find colored (active) troops
        to_deploy = []
        for name_raw, (tx, ty, conf) in self._finder.positions.items():
            name = ALIAS_MAP.get(name_raw, name_raw)
            if name in spell_names or name not in self._name_to_idx:
                continue
            if self.is_slot_active(img_cv, tx, ty):
                idx = self._name_to_idx[name]
                role = self._troop_types[idx]['role']
                count = max(int(remaining_troops[idx]), 1)
                to_deploy.append((name_raw, name, role, count))
                if self.verbose:
                    print(f" {name} ({role}) x{count}")

        if not to_deploy:
            if self.verbose:
                print(" Cleanup: nothing to deploy")
            return

        total = sum(c for _, _, _, c in to_deploy)
        if self.verbose:
            print(f" -> {total} troops to deploy")

        # Tactical order
        role_order = {'tank': 0, 'ranged': 1, 'melee': 2, 'siege': 3, 'hero': 4}
        to_deploy.sort(key=lambda t: role_order.get(t[2], 99))

        deployed = 0
        for name_raw, name, role, count in to_deploy:
            if not self._finder.select(name_raw):
                continue
            time.sleep(DELAY_SWITCH_TROOP)
            idx = self._name_to_idx[name]
            tx, ty, _ = self._finder.positions[name_raw]
            taps = 0

            for _ in range(MAX_CLEANUP_ROUNDS):
                for tap_i in range(3):
                    pos = spread[(taps + tap_i) % len(spread)]
                    self._tap(pos[0], pos[1])
                    time.sleep(DELAY_DEPLOY)
                taps += 3

                check = self._screenshot()
                if check is None:
                    break
                check_cv = cv2.cvtColor(np.array(check), cv2.COLOR_RGB2BGR)
                if not self.is_slot_active(check_cv, tx, ty):
                    break

            deployed += taps
            remaining_troops[idx] = 0
            self._last_troop_name = None

            if self.verbose:
                from clashai.config.logging import pp
                status = 'grayed' if taps < MAX_CLEANUP_ROUNDS * 3 else 'max reached'
                pp(f" {name} -> {taps} taps ({status})", tag='cleanup')

        if self.verbose:
            from clashai.config.logging import pp
            remaining = int(np.sum([
                remaining_troops[i] for i, t in enumerate(self._troop_types)
                if t['role'] != 'spell'
            ]))
            pp(f" Cleanup done: {deployed} actions ({remaining} still in counter)",
               tag='done')

    # -----------------------------------------------------------------
    # Utility: position from V4 sector
    # -----------------------------------------------------------------

    @staticmethod
    def sector_to_position(sector_idx, center_pos):
        """Converts a V4 sector to an absolute position on the perimeter."""
        sector_name = DEPLOY_SECTORS[sector_idx]
        offset = SECTOR_OFFSETS[sector_name]
        return (center_pos + offset) % NUM_POSITIONS
