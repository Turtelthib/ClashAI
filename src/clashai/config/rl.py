# clashai/config/rl.py
# RL constants shared between V3 and V4.
#
# Per-version hyperparameters (LEARNING_RATE, ENTROPY_COEF, BATCH_SIZE…)
# stay defined in the respective agent module (`agent.py` for V3,
# `agent_v4.py` for V4) — they're tuned per architecture and intentionally
# differ. Only constants that are truly shared (grid encoding shape, hero
# list, deploy perimeter size) live here.

# -----------------------------------------------------------------------------
# Hero list (shared between V3, V4, hero_ability, action_space)
# -----------------------------------------------------------------------------
# Previously defined in 3 places. Single source here.
# Note: `duc_draconique` exists as a 6th hero in the troop bar detector's
# UNIQUE_HEROES set but doesn't yet have an ability template — kept out of
# this list until that's wired up.
HERO_NAMES = ['roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille']
NUM_HEROES = len(HERO_NAMES)


# -----------------------------------------------------------------------------
# Deploy perimeter
# -----------------------------------------------------------------------------
# Number of equally-spaced deployment positions around the village
# perimeter. Used by deploy_zone.get_perimeter_from_walls() to sample the
# hull and by the action space to map sector indices.
NUM_POSITIONS = 20


# -----------------------------------------------------------------------------
# Building grid encoding (state_encoder produces a (GRID_CHANNELS, GRID_SIZE, GRID_SIZE) tensor)
# -----------------------------------------------------------------------------
# Same value in V3 agent.py + V4 agent_v4.py + state_encoder.py — unified.
GRID_CHANNELS = 12
GRID_SIZE = 40
VILLAGE_FEATURES = 20


# -----------------------------------------------------------------------------
# V4 action space total (37) — re-exported for convenience.
# -----------------------------------------------------------------------------
# Actual definition stays in action_space.py because it's computed from the
# combination of DEPLOY_ROLES × NUM_SECTORS + spells + abilities + ctrl.
TOTAL_ACTIONS_V4 = 37
