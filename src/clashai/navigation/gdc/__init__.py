# clashai/navigation/gdc/
# Clan War (GdC) navigation + orchestration (Phase 3 split of gdc_navigator.py).
#
# Modules:
#   constants.py    — UI positions, target zone, scroll/wait timings
#   adb_io.py       — _adb_screenshot / _adb_tap / _adb_swipe
#   ocr.py          — _detect_target_numbers (enemy CW list OCR)
#   navigator.py    — GdCNavigator (navigate → select → launch attack)
#   orchestrator.py — GdCOrchestrator (chat → nav → V4 agent → home)
#   __main__.py     — CLI (--attack / --navigate / --monitor)
#
# Public API re-exported so callers keep using:
#   from clashai.navigation.gdc_navigator import GdCNavigator

from clashai.navigation.gdc.constants import (
    _get_ui_pos, TARGET_LIST_ZONE, VISIBLE_TARGETS_PER_SCREEN,
    SCROLL_DISTANCE, SCROLL_DURATION,
    WAIT_NAVIGATION, WAIT_MENU_LOAD, WAIT_SCROLL,
    WAIT_TARGET_LOAD, WAIT_MATCHMAKING, MAX_RETRIES,
    ADB_WIDTH, ADB_HEIGHT,
)
from clashai.navigation.gdc.adb_io import _adb_screenshot, _adb_tap, _adb_swipe
from clashai.navigation.gdc.ocr import _detect_target_numbers
from clashai.navigation.gdc.navigator import GdCNavigator
from clashai.navigation.gdc.orchestrator import GdCOrchestrator

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
