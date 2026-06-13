# clashai/navigation/gdc/constants.py
# UI button positions, target-list zone, scroll + wait timings.

# Re-imported from clashai/config (Phase A) — kept importable for back-compat.
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: F401


def _get_ui_pos(name):
    """UI button position from ui_positions.json (calibrate_ui), with fallback."""
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position(name)
    except ImportError:
        defaults = {
            'chat_open': (47, 400),
            'chat_close_tap': (1400, 400),
            'gdc_open': (47, 560),
            'gdc_war_ended_see_map': (960, 700),
            'gdc_enemy_map': (1700, 540),
            'gdc_ally_map': (200, 540),
            'gdc_attack_target': (900, 660),
            'gdc_village_next': (1050, 680),
            'gdc_village_prev': (100, 680),
            'gdc_return_home': (80, 780),
            'attack_button': (80, 830),
            'start_attack': (960, 700),
            'open_profil': (40, 50),
            'close_profil': (1270, 90),
            'close_menu': (1340, 95),
            'close_popup': (1300, 100),
            'return_home': (960, 800),
        }
        return defaults.get(name, (960, 400))


# Zone where enemy target numbers appear (the enemy list with their #).
TARGET_LIST_ZONE = {
    'left': 100,
    'right': 1820,
    'top': 150,
    'bottom': 850,
}

# Approximate number of CW targets visible at once on screen.
VISIBLE_TARGETS_PER_SCREEN = 5

# Scroll speed for navigating the list.
SCROLL_DISTANCE = 400
SCROLL_DURATION = 300

# Wait times between actions.
WAIT_NAVIGATION = 1.5
WAIT_MENU_LOAD = 2.0
WAIT_SCROLL = 1.0
WAIT_TARGET_LOAD = 2.0
WAIT_MATCHMAKING = 4.0

MAX_RETRIES = 15
