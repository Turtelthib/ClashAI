# clashai/config/screen.py
# Canonical screen / ADB resolution.
#
# All perception models (YOLO buildings, troop bar, walls seg, screen CNN)
# and tap coordinate logic assume the game frame is 1920x1080 — that's what
# ADB `screencap -p` returns natively, and what `ScreenCapture._normalize_to_canonical()`
# resizes WGC/PrintWindow/dxcam/mss frames to.
#
# These values were duplicated across 15+ files (SCREEN_WIDTH, ADB_WIDTH,
# CANONICAL_W, etc. all = 1920). This module is the single source of truth.

# Game frame canonical resolution (matches ADB screencap output)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Aliases kept for back-compat with code that historically used ADB_*.
# New code should prefer SCREEN_WIDTH / SCREEN_HEIGHT.
ADB_WIDTH = SCREEN_WIDTH
ADB_HEIGHT = SCREEN_HEIGHT
