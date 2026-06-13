# clashai/combat/hero/constants.py
# Hero ability display names + CNN mapping constants.

from clashai.config import HERO_NAMES, NUM_HEROES  # noqa: F401 (re-exported)

# Troop bar CNN suffix marking a hero's ability button: 'roi' -> 'roi_capa'.
CAPA_SUFFIX = '_capa'

# Grace delay after deploy before the ability is considered "searched"
# (used only by get_status_vector for the 0.25 vs 0.5 distinction).
DEPLOY_TO_SCAN_DELAY = 5.0

HERO_ABILITY_NAMES = {
    'roi': 'Rage Royale',
    'reine': 'Cloak Royal',
    'grand_gardien': 'Tome Éternel',
    'championne': 'Seeking Shield',
    'prince_gargouille': 'Visage Noir',
}
