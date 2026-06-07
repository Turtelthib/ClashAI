# clashai/perception/deploy/
# Deployment-zone detection — where to drop troops outside the red line.
#
# Split into focused modules (Phase 3):
#   constants.py — HSV ranges, offsets, direction angles, YOLO_WALLS_IMGSZ
#   boundary.py  — HSV detection of the village boundary
#   positions.py — hull geometry -> deploy positions (+ fallback)
#   yolo_zone.py — YOLO walls + building-bbox perimeter
#   debug.py     — visual debug + get_smart_deploy_positions facade
#
# Public API re-exported so callers keep using:
#   from clashai.perception.deploy_zone import get_perimeter_from_buildings, ...

from clashai.perception.deploy.constants import YOLO_WALLS_IMGSZ
from clashai.perception.deploy.boundary import detect_village_boundary
from clashai.perception.deploy.positions import (
    compute_deploy_positions,
    get_village_center_adb,
    get_full_perimeter_positions,
)
from clashai.perception.deploy.yolo_zone import (
    get_perimeter_from_walls,
    get_perimeter_from_buildings,
)
from clashai.perception.deploy.debug import (
    save_deploy_debug_image,
    get_smart_deploy_positions,
    debug_deploy_zone,
)

__all__ = [
    'YOLO_WALLS_IMGSZ',
    'detect_village_boundary',
    'compute_deploy_positions',
    'get_village_center_adb',
    'get_full_perimeter_positions',
    'get_perimeter_from_walls',
    'get_perimeter_from_buildings',
    'save_deploy_debug_image',
    'get_smart_deploy_positions',
    'debug_deploy_zone',
]
