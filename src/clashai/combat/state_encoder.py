# clashai/combat/state_encoder.py
# Back-compat shim — the implementation moved to the `encoder/` package
# (Phase 3 split). Re-exports the public API so existing imports keep
# working:
#   from clashai.combat.state_encoder import encode_state, find_best_attack_side

from clashai.combat.encoder import (  # noqa: F401
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
    buildings_to_grid,
    encode_state,
    extract_features,
    find_best_attack_side,
    get_attack_direction_coords,
    print_state_summary,
)

__all__ = [
    'CATEGORIES', 'CHANNEL_NAMES', 'CLASS_TO_CHANNEL',
    'NUM_CHANNELS', 'NUM_VILLAGE_FEATURES', 'DEFENSE_STATS',
    'INFERNO_CLASSES', 'EAGLE_CLASSES', 'SCATTER_CLASSES', 'CC_CLASSES',
    'buildings_to_grid',
    'extract_features', 'encode_state', 'get_attack_direction_coords',
    'find_best_attack_side', 'print_state_summary',
]
