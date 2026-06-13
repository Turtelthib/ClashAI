# clashai/combat/combat_observer/
# Mid-combat battlefield observation for ClashAI V4 (Phase 3 split).
#
# Two perception modes:
#   - YOLO troops (V4): exact position of each troop/hero by class
#   - HSV health bars (V3 fallback): clusters of green/red bars
#
# Modules:
#   constants.py    — HSV thresholds, bar geometry, UI zones, clustering params
#   health_bars.py  — _detect_bars + detect_troop/hurt/hero_bars
#   clustering.py   — _cluster_positions (BFS)
#   observer.py     — CombatObserver (feature vector + raw data)
#   __main__.py     — smoke test
#
# Same-name package → `from clashai.combat.combat_observer import
# CombatObserver, COMBAT_FEATURES_SIZE` keeps working unchanged.

from clashai.combat.combat_observer.constants import (
    COMBAT_FEATURES_SIZE, ADB_WIDTH, ADB_HEIGHT,
    HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
    HP_RED_H_RANGE, HP_RED_S_MIN, HP_RED_V_MIN,
    HP_ORANGE_H_RANGE, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN,
    HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO,
    HERO_BAR_MIN_AREA, HERO_BAR_MAX_AREA, HERO_BAR_MIN_RATIO,
    UI_BOTTOM_Y, UI_TOP_Y, UI_LEFT_X, UI_RIGHT_X,
    CLUSTER_RADIUS, MIN_CLUSTER_SIZE,
)
from clashai.combat.combat_observer.health_bars import (
    _detect_bars, detect_troop_bars, detect_hurt_bars, detect_hero_bars,
)
from clashai.combat.combat_observer.clustering import _cluster_positions
from clashai.combat.combat_observer.observer import CombatObserver

__all__ = [
    'CombatObserver', 'COMBAT_FEATURES_SIZE',
    '_cluster_positions',
    '_detect_bars', 'detect_troop_bars', 'detect_hurt_bars', 'detect_hero_bars',
    'ADB_WIDTH', 'ADB_HEIGHT',
    'CLUSTER_RADIUS', 'MIN_CLUSTER_SIZE',
]
