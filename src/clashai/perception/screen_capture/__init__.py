# clashai/perception/screen_capture/
# Direct emulator window capture (WGC > PrintWindow > dxcam > mss > ADB).
#
# Split into focused modules (Phase 3):
#   capture.py        — ScreenCapture orchestrator + backends + push pipeline
#   window_detect.py  — locate the emulator HWND
#   gdi_capture.py    — Win32 PrintWindow capture of one HWND
#   normalize.py      — crop+resize a raw capture to the canonical 1920x1080
#
# Public API is re-exported here so callers keep using:
#   from clashai.perception.screen_capture import ScreenCapture, get_capture

from clashai.perception.screen_capture.capture import (
    ScreenCapture,
    get_capture,
    grab_frame,
)
from clashai.perception.screen_capture.window_detect import find_emulator_bbox

__all__ = ['ScreenCapture', 'get_capture', 'grab_frame', 'find_emulator_bbox']
