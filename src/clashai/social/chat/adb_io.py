# clashai/social/chat/adb_io.py
# ADB I/O for chat: screenshot (canonical WGC-routed) + tap.

# Canonical screenshot impl (Phase B.1) — routes through WGC (fast,
# occlusion-proof) with ADB fallback.
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot  # noqa: F401


def _adb_tap(x, y, delay=0.1):
    """Phase C.1: routed through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().tap(x, y, delay=delay)
