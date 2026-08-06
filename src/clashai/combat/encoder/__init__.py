# clashai/combat/encoder/
# Transform YOLO+CNN building detections into the RL state
# (grid tensor + village feature vector).
#
# Split into focused modules (Phase 3):
#   constants.py   — categories, channels, grid config, defense stats
#   grid.py        — buildings_to_grid + danger heatmaps
#   features.py    — extract_features, encode_state, attack-direction coords
#   attack_side.py — find_best_attack_side + state summary
#
# Public API re-exported so callers keep using:
#   from clashai.combat.state_encoder import encode_state, find_best_attack_side

from clashai.combat.encoder.attack_side import (
    find_best_attack_side,
    print_state_summary,
)
from clashai.combat.encoder.constants import (
    CATEGORIES,
    CC_CLASSES,
    CHANNEL_NAMES,
    CLASS_TO_CHANNEL,
    DEFENSE_STATS,
    EAGLE_CLASSES,
    INFERNO_CLASSES,
    NUM_CHANNELS,
    NUM_VILLAGE_FEATURES,
    SCATTER_CLASSES,
)
from clashai.combat.encoder.features import (
    encode_state,
    extract_features,
    get_attack_direction_coords,
)
from clashai.combat.encoder.grid import buildings_to_grid

__all__ = [
    'CATEGORIES', 'CHANNEL_NAMES', 'CLASS_TO_CHANNEL',
    'NUM_CHANNELS', 'NUM_VILLAGE_FEATURES', 'DEFENSE_STATS',
    'INFERNO_CLASSES', 'EAGLE_CLASSES', 'SCATTER_CLASSES', 'CC_CLASSES',
    'buildings_to_grid',
    'extract_features', 'encode_state', 'get_attack_direction_coords',
    'find_best_attack_side', 'print_state_summary',
]
