# clashai/perception/deploy/constants.py
# Constants for deploy-zone detection (HSV ranges, offsets, directions).

import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# ADB resolution — re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# UI exclusion zones (in ADB coordinates 1920×1080)
# Taps in these zones trigger buttons instead of deploying troops
# YOLO walls segmentation trained at imgsz=640 (see
# tools/train/train_yolo_walls_seg.py DEFAULT_IMG_SIZE). Set explicitly so a
# future retrain at 1280/1600 only requires bumping this constant.
YOLO_WALLS_IMGSZ = 640

UI_EXCLUSION_ZONES = [
    # Top: player info + resources
    (0, 0, 280, 230),
    (1450, 0, 1920, 160),
    # Bottom: troop bar only (real UI, not the village)
    (0, 735, 1920, 1080),
    # Side buttons (smaller than the old large rectangle)
    (0, 590, 210, 730),
    (1220, 560, 1510, 730),
]

# Minimum margin from screen edges
SCREEN_MARGIN = 60

# Distance (in ADB pixels) between the hull and the deployment positions
DEPLOY_OFFSET = 35

# V4.2 — Parameters for get_perimeter_from_buildings (YOLO-only, no HSV)
# Artificial expansion of each bbox to simulate the CoC collision zone
# (~1.5 tile at medium zoom). The hull of expanded bboxes covers the real red zone.
BUILDING_PADDING = 40

# Minimum distance between a final position and any building center.
# Must be > half max building size + margin. Most buildings are
# 60-80px wide → 70px guarantees we never tap on a sprite.
MIN_BUILDING_DIST = 40

# Final offset from the hull, adaptive to zoom level.
# Smaller than DEPLOY_OFFSET because padding already does most of the work.
OFFSET_BY_ZOOM = {'dezoome': 30, 'moyen': 20, 'zoome': 10}

# Radial push cap AFTER exiting the hull to avoid landing in
# water/rocks when an off-center building forces a longer push.
# 40px ≈ 1 CoC tile : if no valid spot at ≤ 40px from hull → discard ray.
MAX_RADIAL_PUSH = 40

# Directions (index → label)
DIRECTION_LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']

# Corresponding angles (in radians, 0 = right, counter-clockwise)
# N=up, E=right, S=down, O=left
DIRECTION_ANGLES = {
    0: np.pi / 2,
    1: np.pi / 4,
    2: 0,
    3: -np.pi / 4,
    4: -np.pi / 2,
    5: -3 * np.pi / 4,
    6: np.pi,
    7: 3 * np.pi / 4,
}

# V4.2 — Local HSV validation as a complement to YOLO:
# when a building is not detected by YOLO (worker huts,
# isolated walls…), we check the color of the candidate pixel.
# The CoC red overlay shifts the grass Hue (~33 green) toward ~15 (orange).
HSV_CHECK_RADIUS = 4
HSV_RED_H_MAX = 28
HSV_RED_SAT_MIN = 50
HSV_RED_RATIO_THRESHOLD = 0.5
