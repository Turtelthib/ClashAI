# clashai/combat/environment_v4/observation.py
# ObservationMixin — V4 observation (grid + 54-dim vector) and action mask.

import time

import numpy as np

from clashai.combat.action_space import (
    DEPLOY_ROLES, SPELL_NAMES, compute_action_mask,
)
from clashai.combat.agent_v4 import (
    ROLE_FEATURES, SPELL_FEATURES, COMBAT_FEATURES_SIZE,
)
from clashai.combat.legacy.agent import TROOP_TYPES, TROOP_NAME_TO_IDX


class ObservationMixin:
    """V4 observation (54 dims) + 37-action mask."""

    def _get_obs(self):
        """Builds the V4 observation: grid + 54-dim vector."""
        # Elapsed time normalized over 180s (CoC 3-min timer) — more stable than step/MAX
        elapsed = time.time() - self._episode_start_time
        time_norm = np.array([min(elapsed / 180.0, 1.0)], dtype=np.float32)

        combat_feats = (self._combat_features
                        if self._combat_features is not None
                        else np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32))

        hero_status = self._hero_manager.get_status_vector()

        # V4: role counts instead of individual troop counts
        role_counts = np.zeros(ROLE_FEATURES, dtype=np.float32)
        for i, role in enumerate(DEPLOY_ROLES):
            for troop in TROOP_TYPES:
                if troop['role'] == role and troop['name'] in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[troop['name']]
                    role_counts[i] += self._remaining_troops[idx]
        role_counts = role_counts / 10.0

        # V4: spell counts
        spell_counts = np.zeros(SPELL_FEATURES, dtype=np.float32)
        for i, spell_name in enumerate(SPELL_NAMES):
            if spell_name in TROOP_NAME_TO_IDX:
                spell_counts[i] = self._remaining_troops[TROOP_NAME_TO_IDX[spell_name]]
        spell_counts = spell_counts / 3.0

        vector = np.concatenate([
            self._features,
            role_counts,
            spell_counts,
            self._sector_map,
            time_norm,
            combat_feats,
            hero_status,
        ])

        return self._grid, vector

    def _get_mask(self):
        """V4 action mask."""
        hero_mask = self._hero_manager.get_ability_mask()
        return compute_action_mask(
            self._remaining_troops,
            TROOP_TYPES,
            hero_ability_mask=hero_mask,
        )
