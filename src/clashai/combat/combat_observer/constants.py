# clashai/combat/combat_observer/constants.py
# HSV thresholds for health bars, bar geometry, UI zones, clustering params.

# Re-imported from clashai/config/screen.py (Phase A) — re-exported.
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: F401

# --- Green health bars (healthy troops) ---
HP_GREEN_H_RANGE = (45, 85)
HP_GREEN_S_MIN = 100
HP_GREEN_V_MIN = 120

# --- Red/orange health bars (injured troops) ---
HP_RED_H_RANGE = (0, 15)
HP_RED_S_MIN = 120
HP_RED_V_MIN = 120

HP_ORANGE_H_RANGE = (15, 30)
HP_ORANGE_S_MIN = 100
HP_ORANGE_V_MIN = 120

# --- Health bar size ---
HP_BAR_MIN_AREA = 30
HP_BAR_MAX_AREA = 800
HP_BAR_MIN_RATIO = 1.5

# --- Hero health bars (larger than normal troop bars) ---
HERO_BAR_MIN_AREA = 200
HERO_BAR_MAX_AREA = 2000
HERO_BAR_MIN_RATIO = 2.0

# --- UI exclusion zones ---
UI_BOTTOM_Y = 0.60
UI_TOP_Y = 0.08
UI_LEFT_X = 0.02
UI_RIGHT_X = 0.98

# --- Clustering ---
CLUSTER_RADIUS = 150
MIN_CLUSTER_SIZE = 2

# Number of output features
COMBAT_FEATURES_SIZE = 15
