# clashai/combat/hero/
# Hero special-ability management during combat.
#
# Split into focused modules (Phase 3):
#   constants.py      — ability zone, threshold, ability names, delays
#   template_match.py — multi-scale template matching helper
#   manager.py        — HeroAbilityManager (detect + activate)
#   cli.py            — setup/debug CLI (capture zone, test scan)
#
# NOTE: template matching here is slated for replacement by the YOLO troop
# bar CNN (which already detects roi_capa / reine_capa / … ). See ROADMAP.
#
# Public API re-exported so callers keep using:
#   from clashai.combat.hero_ability import HeroAbilityManager, HERO_NAMES

from clashai.config import HERO_NAMES, NUM_HEROES
from clashai.combat.hero.constants import (
    HERO_ABILITY_NAMES, MATCH_THRESHOLD,
    ABILITY_ZONE_TOP, ABILITY_ZONE_BOTTOM, ABILITY_ZONE_LEFT, ABILITY_ZONE_RIGHT,
)
from clashai.combat.hero.manager import HeroAbilityManager

__all__ = [
    'HeroAbilityManager',
    'HERO_NAMES', 'NUM_HEROES',
    'HERO_ABILITY_NAMES', 'MATCH_THRESHOLD',
    'ABILITY_ZONE_TOP', 'ABILITY_ZONE_BOTTOM',
    'ABILITY_ZONE_LEFT', 'ABILITY_ZONE_RIGHT',
]
