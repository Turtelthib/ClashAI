# clashai/combat/agent_v4/constants.py
# Observation-space dimensions + PPO hyperparameters for the V4 agent.

from clashai.combat.action_space import NUM_ROLES, NUM_SECTORS, NUM_HEROES, SPELL_NAMES
from clashai.combat.combat_observer import COMBAT_FEATURES_SIZE
# Re-imported from clashai/config/rl.py (Phase A) — re-exported for back-compat.
from clashai.config import GRID_CHANNELS, GRID_SIZE, VILLAGE_FEATURES  # noqa: F401

# V4: compacted vector
ROLE_FEATURES = NUM_ROLES
SPELL_FEATURES = len(SPELL_NAMES)
SECTOR_MAP_SIZE = NUM_SECTORS
STEP_FEATURES = 1
HERO_STATUS_SIZE = NUM_HEROES
PHASE_SIZE = 0

VECTOR_SIZE = (VILLAGE_FEATURES + ROLE_FEATURES + SPELL_FEATURES
               + SECTOR_MAP_SIZE + STEP_FEATURES
               + COMBAT_FEATURES_SIZE + HERO_STATUS_SIZE)
# 20 + 5 + 3 + 5 + 1 + 15 + 5 = 54
# Note: checkpoints V4.1 incompatibles (nn.Linear(55→54)).
# Old checkpoints in weights/rl/ will be unusable.

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.02
VALUE_COEF = 0.05
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 3e-4
PPO_EPOCHS = 4
BATCH_SIZE = 16
