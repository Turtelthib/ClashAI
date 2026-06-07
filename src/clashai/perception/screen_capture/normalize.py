# clashai/perception/screen_capture/normalize.py
# Normalise a raw window capture into the canonical game frame.
#
# Windows-side backends (WGC / PrintWindow / dxcam / mss) capture the
# whole window — title bar + borders — at the OS-side (possibly DPI-scaled)
# resolution. Downstream models (screen CNN, YOLO, button coords) all
# expect the ADB-native game frame. This module crops the client area and
# resizes to the canonical resolution.

import ctypes
import ctypes.wintypes

from PIL import Image

from clashai.config import SCREEN_WIDTH, SCREEN_HEIGHT

# Canonical resolution downstream code expects (matches ADB screencap).
CANONICAL_W = SCREEN_WIDTH
CANONICAL_H = SCREEN_HEIGHT


def normalize_to_canonical(img, hwnd):
    """
    Crop the game client area out of a full-window capture and resize to
    CANONICAL_W x CANONICAL_H.

    Uses GetClientRect + ClientToScreen to locate the client area inside
    the window (logical px), then scales by the capture's actual pixel
    size to handle DPI scaling. Falls back to a plain resize if hwnd is
    None or any Win32 call fails.
    """
    if hwnd is None:
        return img.resize((CANONICAL_W, CANONICAL_H), Image.LANCZOS)

    user32 = ctypes.windll.user32
    try:
        win_rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(win_rect))
        win_w = win_rect.right - win_rect.left
        win_h = win_rect.bottom - win_rect.top

        cli_rect = ctypes.wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(cli_rect))
        cli_w = cli_rect.right - cli_rect.left
        cli_h = cli_rect.bottom - cli_rect.top

        pt = ctypes.wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))

        off_x = pt.x - win_rect.left
        off_y = pt.y - win_rect.top

        img_w, img_h = img.size
        scale_x = img_w / win_w if win_w > 0 else 1.0
        scale_y = img_h / win_h if win_h > 0 else 1.0

        crop_box = (
            max(0, int(off_x * scale_x)),
            max(0, int(off_y * scale_y)),
            min(img_w, int((off_x + cli_w) * scale_x)),
            min(img_h, int((off_y + cli_h) * scale_y)),
        )
        cropped = img.crop(crop_box)
    except Exception:
        cropped = img

    if cropped.size != (CANONICAL_W, CANONICAL_H):
        cropped = cropped.resize((CANONICAL_W, CANONICAL_H), Image.LANCZOS)
    return cropped
