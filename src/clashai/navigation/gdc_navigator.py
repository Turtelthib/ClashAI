# clashai/navigation/gdc_navigator.py
# Back-compat shim — implementation moved to the `gdc/` package (Phase 3
# split). Re-exports the public API so existing imports keep working:
#   from clashai.navigation.gdc_navigator import GdCNavigator

from clashai.navigation.gdc import (  # noqa: F401
    GdCNavigator, GdCOrchestrator,
    _detect_target_numbers,
    _adb_screenshot, _adb_tap, _adb_swipe,
    _get_ui_pos, TARGET_LIST_ZONE, VISIBLE_TARGETS_PER_SCREEN,
    SCROLL_DISTANCE, SCROLL_DURATION,
    WAIT_NAVIGATION, WAIT_MENU_LOAD, WAIT_SCROLL,
    WAIT_TARGET_LOAD, WAIT_MATCHMAKING, MAX_RETRIES,
    ADB_WIDTH, ADB_HEIGHT,
)

__all__ = [
    'GdCNavigator', 'GdCOrchestrator',
    '_detect_target_numbers',
    '_adb_screenshot', '_adb_tap', '_adb_swipe',
    '_get_ui_pos', 'TARGET_LIST_ZONE', 'VISIBLE_TARGETS_PER_SCREEN',
    'SCROLL_DISTANCE', 'SCROLL_DURATION',
    'WAIT_NAVIGATION', 'WAIT_MENU_LOAD', 'WAIT_SCROLL',
    'WAIT_TARGET_LOAD', 'WAIT_MATCHMAKING', 'MAX_RETRIES',
    'ADB_WIDTH', 'ADB_HEIGHT',
]
