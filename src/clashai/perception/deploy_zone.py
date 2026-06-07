# clashai/perception/deploy_zone.py
# Back-compat shim — the implementation moved to the `deploy/` package
# (Phase 3 split). This module re-exports the public API so existing
# imports keep working:
#   from clashai.perception.deploy_zone import get_perimeter_from_buildings

from clashai.perception.deploy import (  # noqa: F401
    YOLO_WALLS_IMGSZ,
    detect_village_boundary,
    compute_deploy_positions,
    get_village_center_adb,
    get_full_perimeter_positions,
    get_perimeter_from_walls,
    get_perimeter_from_buildings,
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
