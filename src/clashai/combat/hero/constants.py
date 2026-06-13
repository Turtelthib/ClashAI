# clashai/combat/hero/constants.py
# Ability-scan zone, match threshold, hero ability display names, delays.

from clashai.paths import HERO_TEMPLATES_DIR
from clashai.config import MATCH_SCALES, HERO_NAMES, NUM_HEROES

TEMPLATES_DIR = HERO_TEMPLATES_DIR

ABILITY_ZONE_TOP = 850
ABILITY_ZONE_BOTTOM = 1080
ABILITY_ZONE_LEFT = 0
ABILITY_ZONE_RIGHT = 1920

MATCH_THRESHOLD = 0.50  # local — hero_ability tuned at 0.50

HERO_ABILITY_NAMES = {
    'roi': 'Rage Royale',
    'reine': 'Cloak Royal',
    'grand_gardien': 'Tome Éternel',
    'championne': 'Seeking Shield',
    'prince_gargouille': 'Visage Noir',
}

# Delay after deploy before icons appear; cooldown between scans.
DEPLOY_TO_SCAN_DELAY = 5.0
SCAN_COOLDOWN = 2.0
