# clashai/combat/environment_v4/actions.py
# ActionsMixin — V4 action execution (deploy / spell / ability / wait / observe).

import time

from clashai.combat.action_space import DEPLOY_ROLES, HERO_NAMES, decode_action
from clashai.combat.troop_manager import TroopManager
from clashai.combat.troop_registry import load_spell_targets
from clashai.combat.legacy.agent import TROOP_TYPES, TROOP_NAME_TO_IDX
from clashai.config import (
    DELAY_SWITCH_TROOP, DELAY_DEPLOY,
    DELAY_WAIT_SHORT, DELAY_WAIT_LONG,
    DELAY_OBSERVE, DELAY_ABILITY,
)

# Spell targeting (data-driven): registry category → SpellCaster output key.
_SPELL_TARGETS = load_spell_targets()
_TARGET_TO_CASTER = {'cluster': 'rage', 'heal': 'heal', 'defense': 'freeze'}


class ActionsMixin:
    """Executes a decoded V4 action on the emulator."""

    def _execute_action(self, action_idx):
        """Executes a V4 action."""
        action_type, idx1, idx2 = decode_action(action_idx)

        if action_type == 'deploy':
            return self._execute_deploy(idx1, idx2)

        elif action_type == 'spell':
            return self._execute_spell(idx1)

        elif action_type == 'ability':
            return self._execute_ability(idx1)

        elif action_type == 'observe':
            time.sleep(DELAY_OBSERVE)
            self._update_combat_observation()
            return f"observe ({DELAY_OBSERVE}s)"

        elif action_type == 'wait_short':
            time.sleep(DELAY_WAIT_SHORT)
            self._troop_mgr._last_troop_name = None
            return "wait 0.5s"

        elif action_type == 'wait_long':
            time.sleep(DELAY_WAIT_LONG)
            self._troop_mgr._last_troop_name = None
            return "wait 2.0s"

        elif action_type == 'done':
            return "DONE (episode end)"

        return "???"

    def _execute_deploy(self, role_idx, sector_idx):
        """Deploys a troop of the given role at the given sector."""
        role_name = DEPLOY_ROLES[role_idx]

        # Respect the grayed signal DURING the deploy burst. _remaining_troops
        # is seeded from default_max (no real counter), so the heuristic/agent
        # over-queues deploys; without this, depleted troops get tapped on their
        # grayed icon until the fake counter drains. Cheap cache read (no GPU).
        self._sync_grayed_from_cache()

        # TroopManager selects the next troop for the role
        troop_idx, troop_name = self._troop_mgr.select_next_for_role(
            role_name, self._remaining_troops
        )

        if troop_idx is None:
            return f"WARNING: {role_name} exhausted"

        # Only sleep when a real troop switch happened (TroopFinder.select
        # already waits 0.15s on tap; if same troop is already selected,
        # no delay needed at all)
        if troop_name != self._troop_mgr._last_troop_name:
            time.sleep(DELAY_SWITCH_TROOP)

        # Convert sector → absolute position
        abs_pos = TroopManager.sector_to_position(sector_idx, self._center_pos)

        # V4.2: wraparound on the REAL number of found positions
        # (get_perimeter_from_buildings may return < NUM_POSITIONS).
        # NEVER fall back on _village_center which is inside the TH.
        if self._deploy_positions and len(self._deploy_positions) > 0:
            abs_pos = abs_pos % len(self._deploy_positions)
            x, y = self._deploy_positions[abs_pos]
        else:
            # Only possible fallback: no positions found at all.
            # Tap near the left edge (safe), not at the center.
            x, y = (100, 400)

        self._adb_tap(x, y)
        time.sleep(DELAY_DEPLOY)

        # Track heroes
        troop = TROOP_TYPES[troop_idx]
        if troop['role'] == 'hero':
            self._hero_manager.mark_deployed(troop_name)

        # Update counters
        self._remaining_troops[troop_idx] = max(
            0, self._remaining_troops[troop_idx] - 1
        )
        self._sector_map[sector_idx] += 0.2
        self._last_sector = sector_idx

        return f"{troop_name} → {DEPLOY_ROLES[role_idx]}@{sector_idx}"

    def _execute_spell(self, spell_name):
        """Casts a spell with automatic targeting."""
        if spell_name not in TROOP_NAME_TO_IDX:
            return f"WARNING: spell {spell_name} unknown"

        spell_idx = TROOP_NAME_TO_IDX[spell_name]
        if self._remaining_troops[spell_idx] <= 0:
            return f"WARNING: {spell_name} exhausted"

        # Select the spell in the bar
        if not self._troop_mgr.select_troop(spell_name):
            return f"WARNING: {spell_name} not found"

        time.sleep(DELAY_SWITCH_TROOP)

        # Auto-targeting via SpellCaster
        combat_img = self._adb_screenshot()
        if combat_img is not None:
            if self._combat_observer.has_yolo:
                _, raw = self._combat_observer.observe(
                    combat_img, self._village_center, phase='combat')
                targets = self._spell_caster.analyze_from_yolo(
                    raw, self._village_center)
            else:
                targets = self._spell_caster.analyze_battlefield(
                    combat_img, self._village_center)

            # Data-driven targeting: each spell declares a category
            # (cluster/heal/defense) → mapped to a SpellCaster output.
            category = _SPELL_TARGETS.get(spell_name, 'cluster')
            key = _TARGET_TO_CASTER.get(category, 'rage')
            x, y = targets.get(key) or targets.get('rage') or (
                self._village_center or (960, 500))
        else:
            x, y = self._village_center or (960, 500)

        self._adb_tap(x, y)
        time.sleep(0.3)

        self._remaining_troops[spell_idx] = max(
            0, self._remaining_troops[spell_idx] - 1
        )
        self._troop_mgr._last_troop_name = None

        return f"{spell_name} → ({x}, {y})"

    def _execute_ability(self, hero_idx):
        """Activates a hero's ability."""
        hero_name = HERO_NAMES[hero_idx]

        # Refresh from the troop bar CNN if the icon has not been found yet
        if hero_name not in self._hero_manager._icon_positions:
            bar_det = self.models.get('troop_bar_detector') if self.models else None
            screenshot = self._adb_screenshot()
            if bar_det is not None and screenshot is not None:
                self._hero_manager.update_from_troop_bar(bar_det.detect(screenshot))

        success = self._hero_manager.activate(hero_name, self._adb_tap)
        time.sleep(DELAY_ABILITY)

        if success:
            return f"{hero_name} ability activated"
        return f"WARNING: {hero_name} ability failed"
