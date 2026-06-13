# clashai/combat/hero_ability.py
# Back-compat shim — implementation moved to the `hero/` package
# (Phase 3 split). Re-exports the public API so existing imports keep
# working:
#   from clashai.combat.hero_ability import HeroAbilityManager, HERO_NAMES

from clashai.combat.hero import (  # noqa: F401
    HeroAbilityManager,
    HERO_NAMES, NUM_HEROES,
    HERO_ABILITY_NAMES, MATCH_THRESHOLD,
    ABILITY_ZONE_TOP, ABILITY_ZONE_BOTTOM, ABILITY_ZONE_LEFT, ABILITY_ZONE_RIGHT,
)

__all__ = [
    'HeroAbilityManager',
    'HERO_NAMES', 'NUM_HEROES',
    'HERO_ABILITY_NAMES', 'MATCH_THRESHOLD',
    'ABILITY_ZONE_TOP', 'ABILITY_ZONE_BOTTOM',
    'ABILITY_ZONE_LEFT', 'ABILITY_ZONE_RIGHT',
]
