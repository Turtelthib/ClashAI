# clashai/navigation/gdc/adb_io.py
# ADB I/O for GdC navigation: screenshot (canonical WGC-routed) + tap + swipe.

# Canonical screenshot impl (Phase B.1) — WGC (fast, occlusion-proof) + ADB fallback.
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot  # noqa: F401


def _adb_tap(x, y, delay=0.15):
    """Phase C.1: routed through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().tap(x, y, delay=delay)


def _adb_swipe(x1, y1, x2, y2, duration_ms=300):
    """Phase C.1: routed through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().swipe(x1, y1, x2, y2, duration_ms=duration_ms, delay=0.5)
