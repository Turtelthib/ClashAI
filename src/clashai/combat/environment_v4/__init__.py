# clashai/combat/environment_v4/
# V4 RL environment for ClashAI (Phase 3 split of environment_v4.py).
#
# ClashEnvV4 inherits from ClashEnvV3 (ADB navigation) and is assembled from
# domain mixins so each concern stays in a focused module:
#   core.py        — __init__ / reset / step / episode-end / screen hook
#   observation.py — _get_obs (54 dims) / _get_mask (37 actions)
#   actions.py     — deploy / spell / ability / wait / observe
#   reward.py      — V4.2 reward shaping
#   observe.py     — perception sync (PerceptionThread cache + blocking fallback)
#   capture.py     — annotated episode captures + debug overlays
#   heuristic.py   — scripted action sequence (no-checkpoint mode)
#   env.py         — assembled ClashEnvV4 class
#
# Same-name package → `from clashai.combat.environment_v4 import ClashEnvV4`
# keeps working unchanged.

from clashai.combat.environment_v4.env import ClashEnvV4

__all__ = ['ClashEnvV4']
