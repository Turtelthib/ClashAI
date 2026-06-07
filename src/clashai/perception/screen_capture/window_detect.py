# clashai/perception/screen_capture/window_detect.py
# Locate the emulator window among all top-level / child windows.
#
# All functions are stateless. They filter out editor/browser tabs that
# merely contain the keyword as text, and path-like titles (adbproxy.exe).

import ctypes
import ctypes.wintypes

import numpy as np

from clashai.config import (
    EMULATOR_WINDOW_KEYWORDS,
    title_is_excluded as _title_is_excluded,
)
from clashai.perception.screen_capture.gdi_capture import printwindow_single


def find_emulator_bbox():
    """
    Returns (bbox_dict, title, hwnd) for the emulator window, where
    bbox_dict = {left, top, width, height}. Returns (None, None, None)
    if not found. Picks the largest matching top-level window.
    """
    user32 = ctypes.windll.user32
    found = []

    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM,
    )

    def _cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            if _title_is_excluded(title):
                return True
            if '\\' in title or '.exe' in title.lower() or '.dll' in title.lower():
                return True
            for kw in EMULATOR_WINDOW_KEYWORDS:
                if kw.lower() in title.lower():
                    rect = ctypes.wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w >= 400 and h >= 300:
                        found.append((hwnd, rect.left, rect.top, rect.right, rect.bottom, title))
                    break
        return True

    user32.EnumWindows(EnumProc(_cb), 0)
    if not found:
        return None, None, None
    hwnd, left, top, right, bottom, title = max(
        found, key=lambda x: (x[3] - x[1]) * (x[4] - x[2])
    )
    return ({'left': left, 'top': top, 'width': right - left, 'height': bottom - top},
            title, hwnd)


def find_hwnd():
    """
    Find the emulator main window handle (largest visible matching window).
    Filters out background processes (invisible windows, path-like titles).
    Returns the HWND or None.
    """
    user32 = ctypes.windll.user32
    found = []
    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def _cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n < 3:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if '\\' in title or '.exe' in title.lower() or '.dll' in title.lower():
            return True
        if _title_is_excluded(title):
            return True
        for kw in EMULATOR_WINDOW_KEYWORDS:
            if kw.lower() in title.lower():
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w >= 400 and h >= 300:
                    found.append((hwnd, w * h, title))
                break
        return True

    user32.EnumWindows(EnumProc(_cb), 0)
    if not found:
        return None
    found.sort(key=lambda x: -x[1])
    return found[0][0]


def pick_best_render_hwnd(parent_hwnd, verbose=False):
    """
    Among parent_hwnd and every descendant, probe each with PrintWindow
    and pick the HWND whose capture has the highest pixel variance — that's
    where the game actually renders (the parent often has a transparent
    client area when the surface is a Crosvm/DirectX child window).
    """
    user32 = ctypes.windll.user32
    candidates = [parent_hwnd]

    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def _cb(hwnd, _):
        rect = ctypes.wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w >= 300 and h >= 300:
            candidates.append(hwnd)
        user32.EnumChildWindows(hwnd, EnumProc(_cb), 0)
        return True

    user32.EnumChildWindows(parent_hwnd, EnumProc(_cb), 0)

    best_hwnd = parent_hwnd
    best_score = -1.0
    for hwnd in candidates:
        img = printwindow_single(hwnd)
        if img is None:
            continue
        score = float(np.asarray(img).std())
        if verbose:
            print(f"  probe hwnd={hwnd} -> variance={score:.1f}")
        if score > best_score:
            best_score = score
            best_hwnd = hwnd

    return best_hwnd
