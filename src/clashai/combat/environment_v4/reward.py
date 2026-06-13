# clashai/combat/environment_v4/reward.py
# RewardMixin — V4.2 reward shaping, dispatched on action_type.

from clashai.combat.action_space import DEPLOY_ROLES, HERO_NAMES, decode_action
from clashai.combat.reward_shaping import (
    compute_deploy_reward, compute_combat_reward,
    compute_leftover_penalty, compute_spell_leftover_penalty,
)
from clashai.combat.legacy.agent import TROOP_TYPES


class RewardMixin:
    """Centralized V4.2 reward shaping."""

    def _compute_shaping_reward(self, action_idx):
        """Reward shaping V4.2 — dispatches on action_type instead of self._phase."""
        action_type, idx1, idx2 = decode_action(action_idx)

        if action_type == 'deploy':
            reward = compute_deploy_reward(
                action_type, idx1, idx2,
                self._tanks_deployed, self._troops_deployed,
                self._last_sector, self._combat_features,
            )
            if idx1 is not None:
                role = DEPLOY_ROLES[idx1]
                if role == 'tank':
                    self._tanks_deployed += 1
                self._troops_deployed += 1

        elif action_type in ('spell', 'ability', 'observe', 'wait_short', 'wait_long'):
            spell_name = idx1 if action_type == 'spell' else None
            hero_idx = idx1 if action_type == 'ability' else None
            reward = compute_combat_reward(
                action_type, spell_name, hero_idx,
                self._combat_features,
                self._combat_step_count,
                HERO_NAMES,
            )
            # Building destruction bonus detected on this step (observe only)
            if action_type == 'observe' and self._buildings_destroyed_total > 0:
                new_destroyed = self._buildings_destroyed_total - getattr(
                    self, '_last_rewarded_destroyed', 0
                )
                if new_destroyed > 0:
                    reward += 2.0 * new_destroyed
                    self._last_rewarded_destroyed = self._buildings_destroyed_total

        elif action_type == 'done':
            reward = compute_leftover_penalty(self._remaining_troops, TROOP_TYPES)
            reward += compute_spell_leftover_penalty(self._remaining_troops, TROOP_TYPES)

        else:
            reward = 0.0

        self._step_rewards.append(reward)
        return reward
