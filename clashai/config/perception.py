# clashai/config/perception.py
# Centralised perception thresholds.
#
# Per-model `YOLO_*_IMGSZ` constants stay defined in their respective
# detector modules (`troop_bar_detector.py`, `troop_detector.py`,
# `deploy_zone.py`, `game_loop.py`) so a single retrain only requires
# bumping one constant. They're re-exported here for discoverability and
# uniform access from outside code.

# -----------------------------------------------------------------------------
# YOLO inference image sizes (re-exported from detector modules)
# -----------------------------------------------------------------------------
# These match the imgsz used at training time. Set explicitly at predict
# because Ultralytics defaults to 640 if not specified, which silently
# halves resolution for models trained at 1280/1600.

# Use late imports so this module never fails because a downstream module
# is itself broken at import time.
def _import_imgsz():
    from clashai.navigation.game_loop import YOLO_BUILDINGS_IMGSZ
    from clashai.perception.troop_bar_detector import YOLO_IMGSZ as YOLO_TROOP_BAR_IMGSZ
    from clashai.perception.troop_detector import YOLO_TROOPS_IMGSZ
    from clashai.perception.deploy_zone import YOLO_WALLS_IMGSZ
    return {
        'YOLO_BUILDINGS_IMGSZ':  YOLO_BUILDINGS_IMGSZ,
        'YOLO_TROOP_BAR_IMGSZ':  YOLO_TROOP_BAR_IMGSZ,
        'YOLO_TROOPS_IMGSZ':     YOLO_TROOPS_IMGSZ,
        'YOLO_WALLS_IMGSZ':      YOLO_WALLS_IMGSZ,
    }


# -----------------------------------------------------------------------------
# Confidence thresholds (shared across modules)
# -----------------------------------------------------------------------------

# Screen-state CNN (`classify_screen()` in game_loop) — minimum confidence
# below which we ignore the prediction and re-poll.
SCREEN_CONFIDENCE_THRESHOLD = 0.60

# Building CNN (legacy per-bbox classifier — supplements YOLO buildings).
BUILDING_CONFIDENCE_THRESHOLD = 0.50

# Generic YOLO confidence + IoU defaults (game_loop applies these to YOLO
# buildings; per-detector overrides exist).
YOLO_CONF_DEFAULT = 0.25
YOLO_IOU_DEFAULT = 0.50


# -----------------------------------------------------------------------------
# Template matching (cv2.matchTemplate)
# -----------------------------------------------------------------------------

# Identical multi-scale list previously triplicated in
# hero_ability.py / troop_finder.py / clan_castle.py — single source here.
MATCH_SCALES = [1.0, 0.9, 1.1, 0.85, 1.15]

# Each consumer keeps its own MATCH_THRESHOLD because they were tuned
# differently:
#   troop_finder        → 0.45 (templates are noisy)
#   hero_ability        → 0.50
#   clan_castle         → 0.60 (high-contrast UI templates)
# This default is for new code that hasn't been tuned yet.
MATCH_THRESHOLD_DEFAULT = 0.50

# Reward / counter digit templates use a separate threshold.
DIGIT_MATCH_THRESHOLD = 0.60
