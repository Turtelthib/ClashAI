# scripts/rl/zoom_control.py
# Zoom control via Windows API (ctypes mouse_event).
#
# pyautogui.scroll does NOT work on the Google Play Games emulator.
# However, ctypes.windll.user32.mouse_event with MOUSEEVENTF_WHEEL
# works perfectly.
#
# Usage:
# from clashai.navigation.zoom_control import zoom_out
# zoom_out() # Zooms out to maximum before each attack

import time
import sys
import ctypes


# =============================================================================
# CONFIGURATION
# =============================================================================

# Name of the emulator window (to find it automatically)
EMULATOR_WINDOW_KEYWORDS = ['mulateur', 'Google Play', 'play games']

# Fallback if the window is not found
FALLBACK_CENTER_X = 1334
FALLBACK_CENTER_Y = 764

# Scroll parameters
ZOOM_OUT_SCROLLS = 15
SCROLL_DELTA = -120
SCROLL_DELAY = 0.08

# Windows API constants
MOUSEEVENTF_WHEEL = 0x0800


# =============================================================================
# FIND EMULATOR WINDOW
# =============================================================================

def _find_emulator_center():
    """
    Automatically finds the center of the emulator window.
    Uses pygetwindow if available, otherwise falls back.
    """
    try:
        import pygetwindow as gw
        windows = gw.getAllTitles()

        for keyword in EMULATOR_WINDOW_KEYWORDS:
            matches = [w for w in windows if keyword.lower() in w.lower()]
            if matches:
                win = gw.getWindowsWithTitle(matches[0])[0]
                cx = win.left + win.width // 2
                cy = win.top + win.height // 2
                return cx, cy

    except ImportError:
        pass
    except Exception:
        pass

    return FALLBACK_CENTER_X, FALLBACK_CENTER_Y


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def zoom_out(scrolls=None):
    """
    Zooms out to maximum by simulating the mouse wheel via Windows API.

    Args:
        scrolls: number of scrolls (default: ZOOM_OUT_SCROLLS)

    Returns:
        True if zoom-out was performed
    """
    if scrolls is None:
        scrolls = ZOOM_OUT_SCROLLS

    # Find the emulator center
    center_x, center_y = _find_emulator_center()

    # Save the current cursor position
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    original_x, original_y = pt.x, pt.y

    try:
        # Place the cursor at the emulator center
        ctypes.windll.user32.SetCursorPos(center_x, center_y)
        time.sleep(0.2)

        # Scroll to zoom out
        for _ in range(scrolls):
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_WHEEL, 0, 0, SCROLL_DELTA, 0
            )
            time.sleep(SCROLL_DELAY)

        # Small delay for the game to finish the zoom animation
        time.sleep(0.3)

        print(f" Dézoom effectué ({scrolls} scrolls)")
        return True

    except Exception as e:
        print(f" WARNING: Erreur dézoom : {e}")
        return False

    finally:
        # Restore the cursor to its original position
        ctypes.windll.user32.SetCursorPos(original_x, original_y)


def zoom_in(scrolls=5):
    """Zoome (scroll vers le haut). Utile pour les tests."""
    center_x, center_y = _find_emulator_center()

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    original_x, original_y = pt.x, pt.y

    try:
        ctypes.windll.user32.SetCursorPos(center_x, center_y)
        time.sleep(0.2)

        for _ in range(scrolls):
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_WHEEL, 0, 0, 120, 0
            )
            time.sleep(SCROLL_DELAY)

        return True
    finally:
        ctypes.windll.user32.SetCursorPos(original_x, original_y)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if '--test' in sys.argv:
        print("Test dézoom...")
        zoom_out(scrolls=10)
        print("Terminé ! Vérifie que le jeu a dézoomé.")

    elif '--zoom-in' in sys.argv:
        print("Test zoom in...")
        zoom_in(scrolls=5)
        print("Terminé !")

    else:
        cx, cy = _find_emulator_center()
        print("zoom_control.py — Dézoom via Windows API")
        print(f" Centre émulateur : ({cx}, {cy})")
        print(f" Scrolls : {ZOOM_OUT_SCROLLS}")
        print()
        print(" --test Tester le dézoom")
        print(" --zoom-in Tester le zoom")