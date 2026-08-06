# clashai/combat/agent_v4/
# PPO Agent V4 for ClashAI (Phase 3 split of agent_v4.py).
#
# 37 actions (role × sector + auto-targeted spells), compacted 54-dim
# observation, ~400K-param CNN+MLP shared → actor/critic network.
#
# Modules:
#   constants.py — observation-space dims + PPO hyperparameters
#   network.py   — ActorCriticV4 (nn.Module)
#   buffer.py    — RolloutBuffer (GAE advantages/returns)
#   bc.py        — BehavioralCloningMixin (imitation pretraining)
#   agent.py     — PPOAgentV4 (select_action / update / save / load)
#   __main__.py  — smoke test
#
# Same-name package → `from clashai.combat.agent_v4 import PPOAgentV4, ...`
# keeps working unchanged.

from clashai.combat.agent_v4.agent import PPOAgentV4
from clashai.combat.agent_v4.buffer import RolloutBuffer
from clashai.combat.agent_v4.constants import (
    BATCH_SIZE,
    CLIP_EPSILON,
    COMBAT_FEATURES_SIZE,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    GRID_CHANNELS,
    GRID_SIZE,
    HERO_STATUS_SIZE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    PHASE_SIZE,
    PPO_EPOCHS,
    ROLE_FEATURES,
    SECTOR_MAP_SIZE,
    SPELL_FEATURES,
    STEP_FEATURES,
    VALUE_COEF,
    VECTOR_SIZE,
    VILLAGE_FEATURES,
)
from clashai.combat.agent_v4.network import ActorCriticV4

__all__ = [
    'PPOAgentV4', 'ActorCriticV4', 'RolloutBuffer',
    'ROLE_FEATURES', 'SPELL_FEATURES', 'SECTOR_MAP_SIZE', 'STEP_FEATURES',
    'HERO_STATUS_SIZE', 'PHASE_SIZE', 'VECTOR_SIZE', 'COMBAT_FEATURES_SIZE',
    'GAMMA', 'GAE_LAMBDA', 'CLIP_EPSILON', 'ENTROPY_COEF', 'VALUE_COEF',
    'MAX_GRAD_NORM', 'LEARNING_RATE', 'PPO_EPOCHS', 'BATCH_SIZE',
    'GRID_CHANNELS', 'GRID_SIZE', 'VILLAGE_FEATURES',
]
