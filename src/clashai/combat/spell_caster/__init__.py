# clashai/combat/spell_caster/
# SpellCaster V2 — intelligent heal/rage/freeze targeting (Phase 3 bonus split).
#
# - FREEZE: nearest dangerous defense (inferno / eagle…) to the troop cluster
# - HEAL:   cluster of injured troops (red/orange health bars)
# - RAGE:   in front of troops, toward the village center
#
# Modules:
#   constants.py   — HSV thresholds, bar geometry, freeze priorities
#   health_bars.py — detect_health_bars (green / red+orange, color param)
#   clustering.py  — cluster_positions (BFS, keeps member points)
#   caster.py      — SpellCaster (analyze_battlefield / analyze_from_yolo)
#   __main__.py    — test CLI
#
# Same-name package → `from clashai.combat.spell_caster import SpellCaster`
# keeps working unchanged.

from clashai.combat.spell_caster.constants import (
    ADB_WIDTH, ADB_HEIGHT,
    FREEZE_PRIORITY_CLASSES, FREEZE_PRIORITY_WEIGHTS, FREEZE_MAX_RANGE,
)
from clashai.combat.spell_caster.health_bars import detect_health_bars
from clashai.combat.spell_caster.clustering import cluster_positions
from clashai.combat.spell_caster.caster import SpellCaster

__all__ = [
    'SpellCaster',
    'detect_health_bars', 'cluster_positions',
    'FREEZE_PRIORITY_CLASSES', 'FREEZE_PRIORITY_WEIGHTS', 'FREEZE_MAX_RANGE',
    'ADB_WIDTH', 'ADB_HEIGHT',
]
