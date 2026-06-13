# clashai/combat/hero_ability.py
# Back-compat shim — implementation moved to the `hero/` package
# (Phase 3 split + V4.4 CNN migration). Re-exports the public API so
# existing imports keep working:
#   from clashai.combat.hero_ability import HeroAbilityManager, HERO_NAMES

from clashai.combat.hero import (  # noqa: F401
    HeroAbilityManager,
    HERO_NAMES, NUM_HEROES,
    HERO_ABILITY_NAMES, CAPA_SUFFIX,
)

__all__ = [
    'HeroAbilityManager',
    'HERO_NAMES', 'NUM_HEROES',
    'HERO_ABILITY_NAMES', 'CAPA_SUFFIX',
]
