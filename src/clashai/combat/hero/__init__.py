# clashai/combat/hero/
# Hero special-ability management during combat.
#
# V4.4: ability availability is read from the YOLO troop bar CNN (which
# detects `<hero>_capa` ability buttons), replacing the old template
# matching. The same detector already runs in PerceptionThread, so the
# manager just consumes its detections — no extra inference.
#
# Modules:
#   constants.py — ability display names + CNN mapping constants
#   manager.py   — HeroAbilityManager (consume detections + activate)
#   cli.py       — debug CLI (run the CNN, list available abilities)
#
# Public API re-exported so callers keep using:
#   from clashai.combat.hero_ability import HeroAbilityManager, HERO_NAMES

from clashai.config import HERO_NAMES, NUM_HEROES
from clashai.combat.hero.constants import HERO_ABILITY_NAMES, CAPA_SUFFIX
from clashai.combat.hero.manager import HeroAbilityManager

__all__ = [
    'HeroAbilityManager',
    'HERO_NAMES', 'NUM_HEROES',
    'HERO_ABILITY_NAMES', 'CAPA_SUFFIX',
]
