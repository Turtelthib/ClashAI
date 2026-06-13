# clashai/combat/environment_v4/heuristic.py
# HeuristicMixin — scripted V4 action sequence (one action per unit).

from clashai.combat.action_space import (
    HERO_NAMES, build_role_inventory, build_spell_inventory,
)
from clashai.combat.legacy.agent import TROOP_TYPES


class HeuristicMixin:
    """Scripted attack sequence used when no RL checkpoint is loaded."""

    def get_heuristic_sequence(self):
        """
        Heuristic sequence V4 — 1 action per unit.

        Each troop = 1 deploy(role, sector) action.
        Sectors cycle to spread troops evenly.
        """
        from clashai.combat.action_space import encode_action as enc

        FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT = 0, 1, 2, 3, 4
        TANK, RANGED, MELEE, HERO, SIEGE = 0, 1, 2, 3, 4

        actions = []
        role_inv, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
        spell_inv = build_spell_inventory(self._remaining_troops, TROOP_TYPES)

        if self.verbose:
            print(f" V4 inventory: {dict(role_inv)} | spells: {dict(spell_inv)}")

        # 1. TANKS — spread at extremes then center
        tank_sectors = [FAR_LEFT, FAR_RIGHT, CENTER, LEFT, RIGHT]
        for i in range(role_inv.get('tank', 0)):
            actions.append(enc('deploy', TANK, tank_sectors[i % len(tank_sectors)]))

        actions.append(enc('wait_long'))

        # 2. FUNNEL — 2 ranged aux extrémités
        funnel = min(role_inv.get('ranged', 0), 2)
        funnel_secs = [FAR_LEFT, FAR_RIGHT]
        for i in range(funnel):
            actions.append(enc('deploy', RANGED, funnel_secs[i]))

        actions.append(enc('wait_short'))

        # 3. RANGED — remaining in a line (left, center, right)
        ranged_remaining = max(0, role_inv.get('ranged', 0) - funnel)
        ranged_sectors = [LEFT, CENTER, RIGHT]
        for i in range(ranged_remaining):
            actions.append(enc('deploy', RANGED, ranged_sectors[i % len(ranged_sectors)]))

        actions.append(enc('wait_long'))

        # 4. MELEE — at center
        melee_sectors = [CENTER, LEFT, RIGHT]
        for i in range(role_inv.get('melee', 0)):
            actions.append(enc('deploy', MELEE, melee_sectors[i % len(melee_sectors)]))

        # 5. SIEGE — center (V4.1: siege BEFORE heroes)
        for _ in range(role_inv.get('siege', 0)):
            actions.append(enc('deploy', SIEGE, CENTER))

        # 6. HEROES — centre
        for _ in range(role_inv.get('hero', 0)):
            actions.append(enc('deploy', HERO, CENTER))

        # V4.2: no intermediate done — done = end of episode
        # Spells and abilities follow directly after deploy.
        actions.append(enc('observe'))

        # Spells in priority order — rage, freeze, heal (tactical order)
        spell_priority = ['rage', 'gel', 'soin']
        for spell_name in spell_priority:
            count = spell_inv.get(spell_name, 0)
            for _ in range(count):
                actions.append(enc('observe'))
                actions.append(enc('spell', spell_name))

        # Abilities — deployed heroes only (skip championne/PG if absent)
        for i, hero_name in enumerate(HERO_NAMES):
            if self._hero_manager.is_deployed(hero_name):
                actions.append(enc('observe'))
                actions.append(enc('ability', i))

        actions.append(enc('observe'))
        actions.append(enc('done'))

        return actions
