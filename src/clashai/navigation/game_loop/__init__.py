# clashai/navigation/game_loop/
# Central perception + ADB glue for the ClashAI agent.
#
# Split into focused modules (Phase 3):
#   constants.py  — DEVICE, thresholds, YOLO settings, torchvision transforms
#   models.py     — load_models() (CNNs + YOLO + PerceptionThread)
#   analysis.py   — classify_screen, analyze_village, get_village_summary
#   adb_io.py     — adb_screenshot / tap / swipe / key / check_connection
#   controller.py — BUTTONS, handle_state, run_test, run_live, main (CLI)
#
# Public API re-exported so the ~23 callers keep using:
#   from clashai.navigation.game_loop import load_models, adb_screenshot, ...
#   from clashai.navigation import game_loop; game_loop.classify_screen(...)

from clashai.navigation.game_loop.constants import (
    DEVICE,
    SCREEN_CONFIDENCE_THRESHOLD, BUILDING_CONFIDENCE_THRESHOLD,
    YOLO_CONF, YOLO_IOU, YOLO_BUILDINGS_IMGSZ,
    screen_transform, building_transform,
)
from clashai.navigation.game_loop.models import load_models
from clashai.navigation.game_loop.analysis import (
    classify_screen, analyze_village, get_village_summary,
)
from clashai.navigation.game_loop.adb_io import (
    adb_screenshot, adb_tap, adb_swipe, adb_key, adb_check_connection,
)
from clashai.navigation.game_loop.controller import (
    BUTTONS, handle_state, run_test, run_live, main,
)

__all__ = [
    'DEVICE',
    'SCREEN_CONFIDENCE_THRESHOLD', 'BUILDING_CONFIDENCE_THRESHOLD',
    'YOLO_CONF', 'YOLO_IOU', 'YOLO_BUILDINGS_IMGSZ',
    'screen_transform', 'building_transform',
    'load_models',
    'classify_screen', 'analyze_village', 'get_village_summary',
    'adb_screenshot', 'adb_tap', 'adb_swipe', 'adb_key', 'adb_check_connection',
    'BUTTONS', 'handle_state', 'run_test', 'run_live', 'main',
]
