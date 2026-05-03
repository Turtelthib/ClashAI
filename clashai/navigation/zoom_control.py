# clashai/navigation/zoom_control.py
# Zoom control via Windows SendMessage — completely isolated to the emulator.
#
# SendMessage(hwnd, WM_MOUSEWHEEL, ...) sends directly to the window handle,
# bypassing focus and cursor position entirely — same isolation as ADB taps.
#
# If the main window doesn't respond, it automatically tries child windows
# (some emulators render the game in a child OpenGL/DirectX viewport).
#
# CTRL_SCROLL = True   →  Ctrl+scroll (LDPlayer, BlueStacks, MuMu…)
# CTRL_SCROLL = False  →  plain scroll (Google Play Games)
#
# Currently: Google Play Games (localhost:6520) → CTRL_SCROLL = False

import time
import sys
import ctypes
import ctypes.wintypes


# =============================================================================
# CONFIGURATION
# =============================================================================

CTRL_SCROLL = False

EMULATOR_WINDOW_KEYWORDS = [
    'Google Play', 'play games',
    'LDPlayer', 'LD Player', 'LDMultiPlayer',
    'BlueStacks',
    'MuMu',
    'Nox',
    'MEmu',
    'Clash of Clans',
]

ZOOM_OUT_SCROLLS = 15
SCROLL_DELTA     = -120
SCROLL_DELAY     = 0.05

WM_MOUSEWHEEL = 0x020A
MK_CONTROL    = 0x0008


# =============================================================================
# INTERNAL
# =============================================================================

_user32 = ctypes.windll.user32


def _find_emulator_hwnd():
    """Returns (hwnd, title) of the emulator main window, or None."""
    found = []

    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM,
    )

    def _cb(hwnd, _):
        n = _user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            _user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            for kw in EMULATOR_WINDOW_KEYWORDS:
                if kw.lower() in title.lower():
                    found.append((hwnd, title))
                    return False
        return True

    _user32.EnumWindows(EnumProc(_cb), 0)
    return found[0] if found else None


def _make_wm_mousewheel_params(delta, use_ctrl, hwnd):
    """Returns (wParam, lParam) for WM_MOUSEWHEEL targeting hwnd's center."""
    rect = ctypes.wintypes.RECT()
    _user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2

    keys = MK_CONTROL if use_ctrl else 0
    wParam = ctypes.c_uint(((delta & 0xFFFF) << 16) | keys)
    lParam = ctypes.c_long((cy << 16) | (cx & 0xFFFF))
    return wParam, lParam


def _get_child_hwnds(hwnd):
    """Returns all direct child window handles."""
    children = []
    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM,
    )
    def _cb(child, _):
        children.append(child)
        return True
    _user32.EnumChildWindows(hwnd, EnumProc(_cb), 0)
    return children


def _send_scroll(hwnd, delta, use_ctrl, scrolls):
    """
    Sends WM_MOUSEWHEEL directly to hwnd via SendMessage.
    If the main window doesn't respond (no effect), also tries child windows.
    SendMessage is synchronous and targets the handle directly —
    no cursor movement, no focus required.
    """
    wParam, lParam = _make_wm_mousewheel_params(delta, use_ctrl, hwnd)

    # Try main window
    for _ in range(scrolls):
        _user32.SendMessageW(hwnd, WM_MOUSEWHEEL, wParam, lParam)
        time.sleep(SCROLL_DELAY)

    # Also send to child windows (game viewport in some emulators)
    children = _get_child_hwnds(hwnd)
    if children:
        child_wParam, child_lParam = _make_wm_mousewheel_params(delta, use_ctrl, children[0])
        for child in children:
            for _ in range(scrolls):
                _user32.SendMessageW(child, WM_MOUSEWHEEL, child_wParam, child_lParam)
                time.sleep(SCROLL_DELAY)


# =============================================================================
# PUBLIC API
# =============================================================================

def zoom_out(scrolls=None):
    """Zooms out the game in the emulator — fully isolated, no cursor needed."""
    if scrolls is None:
        scrolls = ZOOM_OUT_SCROLLS

    result = _find_emulator_hwnd()
    if result is None:
        print(" WARNING: Emulator window not found — zoom_out skipped")
        print(f" Add the window title to EMULATOR_WINDOW_KEYWORDS in zoom_control.py")
        return False

    hwnd, title = result
    _send_scroll(hwnd, SCROLL_DELTA, CTRL_SCROLL, scrolls)
    print(f" Dézoom effectué ({scrolls} scrolls) → {title}")
    return True


def zoom_in(scrolls=5):
    """Zooms in the game in the emulator — fully isolated, no cursor needed."""
    result = _find_emulator_hwnd()
    if result is None:
        return False

    hwnd, _ = result
    _send_scroll(hwnd, -SCROLL_DELTA, CTRL_SCROLL, scrolls)
    return True


# =============================================================================
# MAIN (test)
# =============================================================================

if __name__ == "__main__":
    result = _find_emulator_hwnd()
    if result:
        hwnd, title = result
        children = _get_child_hwnds(hwnd)
        print(f"Emulator: '{title}' (hwnd={hwnd}, {len(children)} child windows)")
    else:
        print("WARNING: No emulator window found.")
        print(f"Keywords: {EMULATOR_WINDOW_KEYWORDS}")

    if '--test' in sys.argv:
        zoom_out(scrolls=10)
    elif '--zoom-in' in sys.argv:
        zoom_in(scrolls=5)
