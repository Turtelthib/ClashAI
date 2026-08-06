# clashai/config/__init__.py
# Centralised configuration package for ClashAI.
#
# Use:
#   from clashai.config import SCREEN_WIDTH, DELAY_DEPLOY
# or:
#   from clashai.config.timing import DELAY_OBSERVE, WAIT_BATTLE_MAX
#
# Convention: only constants that are truly shared between 2+ modules live
# here. Context-specific constants (deploy_zone HSV ranges, spell_caster
# priorities, …) stay in their respective module.
#
# Sub-modules:
#   screen      : canonical 1920x1080 game frame
#   timing      : DELAY_*, WAIT_*, ADB_DELAY_* shared by env/brain/perception
#   perception  : YOLO confidence thresholds, MATCH_SCALES, screen CNN threshold
#   rl          : GRID_CHANNELS, HERO_NAMES, NUM_POSITIONS, etc. (V3+V4)
#   brain       : orchestrator intervals / priorities / clan-castle / chat
#   window      : EMULATOR_WINDOW_KEYWORDS, EXCLUDED_TITLE_SUBSTRINGS

# Most-used constants re-exported at package level for ergonomy.
# (Modules can still import directly from `clashai.config.<sub>` for the rest.)

from clashai.config.brain import (
    ATTACKS_BEFORE_CHAT_CHECK,
    CHAT_CHECK_INTERVAL,
    DEFAULT_BOT_NAME,
    IDLE_BETWEEN_ATTACKS,
    IDLE_BETWEEN_ATTACKS_MAX,
    MAX_COMMAND_AGE_MINUTES,
    MAX_HISTORY,
    MONITOR_INTERVAL,
    PRIORITY_FARM_ATTACK,
    PRIORITY_GDC_COMMAND,
    PRIORITY_IDLE,
    REQUEST_COOLDOWN,
)
from clashai.config.perception import (
    BUILDING_CONFIDENCE_THRESHOLD,
    DIGIT_MATCH_THRESHOLD,
    MATCH_SCALES,
    MATCH_THRESHOLD_DEFAULT,
    SCREEN_CONFIDENCE_THRESHOLD,
    YOLO_CONF_DEFAULT,
    YOLO_IOU_DEFAULT,
)
from clashai.config.rl import (
    GRID_CHANNELS,
    GRID_SIZE,
    HERO_NAMES,
    NUM_HEROES,
    NUM_POSITIONS,
    TOTAL_ACTIONS_V4,
    VILLAGE_FEATURES,
)
from clashai.config.screen import (
    ADB_HEIGHT,
    ADB_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from clashai.config.timing import (
    ADB_DELAY_MATCHMAKING,
    ADB_DELAY_NAVIGATION,
    ADB_DELAY_SCREENSHOT,
    ADB_DELAY_TAP,
    DELAY_ABILITY,
    DELAY_DEPLOY,
    DELAY_OBSERVE,
    DELAY_SWITCH_TROOP,
    DELAY_WAIT_COMBAT,
    DELAY_WAIT_LONG,
    DELAY_WAIT_SHORT,
    WAIT_BATTLE_CHECK,
    WAIT_BATTLE_MAX,
    WAIT_DECORATIONS,
    WAIT_MATCHMAKING,
    WAIT_NAVIGATION,
    WAIT_RESULT_SCREEN,
)
from clashai.config.window import (
    EMULATOR_WINDOW_KEYWORDS,
    EXCLUDED_TITLE_SUBSTRINGS,
    title_is_excluded,
)
