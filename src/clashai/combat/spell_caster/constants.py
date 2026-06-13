# clashai/combat/spell_caster/constants.py
# HSV health-bar thresholds + freeze-target priorities for spell targeting.

# Re-imported from clashai/config/screen.py (Phase A) — re-exported.
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: F401

# --- Green health bars (healthy troops) ---
HP_BAR_H_MIN = 45
HP_BAR_H_MAX = 85
HP_BAR_S_MIN = 100
HP_BAR_V_MIN = 120

# --- Red/orange health bars (injured troops) ---
HP_RED_H_MIN = 0
HP_RED_H_MAX = 10
HP_RED_S_MIN = 120
HP_RED_V_MIN = 120

HP_ORANGE_H_MIN = 10
HP_ORANGE_H_MAX = 25
HP_ORANGE_S_MIN = 100
HP_ORANGE_V_MIN = 120

# --- Health bar size ---
HP_BAR_MIN_AREA = 30
HP_BAR_MAX_AREA = 800
HP_BAR_MIN_RATIO = 1.5

# --- UI exclusion zones ---
UI_EXCLUSION_Y = 0.60
UI_EXCLUSION_TOP = 0.08

# --- Defense classes to target with freeze ---
FREEZE_PRIORITY_CLASSES = [
    'tour_enfer_mono',
    'tour_enfer_multiple',
    'aigle_artilleur',
    'catapulte_erratique',
    'arcX_sol', 'arcX_sol_air',
    'monolithe',
]

# Priority weights (higher = higher priority for freeze)
FREEZE_PRIORITY_WEIGHTS = {
    'tour_enfer_mono': 10.0,
    'tour_enfer_multiple': 10.0,
    'aigle_artilleur': 7.0,
    'catapulte_erratique': 5.0,
    'arcX_sol': 4.0,
    'arcX_sol_air': 4.0,
    'monolithe': 6.0,
}

# Max distance to consider a defense threatening to troops (ADB pixels)
FREEZE_MAX_RANGE = 600
