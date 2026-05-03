# clashai/perception/screen_capture.py
# Direct emulator window capture — bypasses ADB for perception.
#
# Captures the emulator window directly via Windows screen APIs.
# ~5-10ms per frame vs ~150ms for ADB screencap PNG.
# Taps still go through ADB (only way to send inputs to Android).
#
# Priority: dxcam (GPU, fastest) > mss (CPU, lightweight) > ADB (fallback)
#
# Usage:
#   from clashai.perception.screen_capture import ScreenCapture
#   cap = ScreenCapture()
#   img = cap.grab()   # PIL.Image, same interface as adb_screenshot()

import ctypes
import ctypes.wintypes
import numpy as np
from PIL import Image

from clashai.navigation.zoom_control import EMULATOR_WINDOW_KEYWORDS


# =============================================================================
# WINDOW DETECTION
# =============================================================================

def find_emulator_bbox():
    """
    Returns (left, top, right, bottom) of the emulator window in screen coords.
    Returns None if not found.
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
            for kw in EMULATOR_WINDOW_KEYWORDS:
                if kw.lower() in title.lower():
                    rect = ctypes.wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w >= 400 and h >= 300:
                        found.append((rect.left, rect.top, rect.right, rect.bottom, title))
                    break  # Avoid duplicate entries from multiple matching keywords
        return True  # Always continue to collect all matches

    user32.EnumWindows(EnumProc(_cb), 0)
    if not found:
        return None, None
    # Pick the largest matching window (avoids tiny ghost windows)
    left, top, right, bottom, title = max(found, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
    return {'left': left, 'top': top, 'width': right - left, 'height': bottom - top}, title


# =============================================================================
# SCREEN CAPTURE
# =============================================================================

class ScreenCapture:
    """
    Direct window capture — much faster than ADB screencap.

    Automatically picks the best available backend:
      1. dxcam  — GPU capture via DXGI (fastest, ~2-5ms)
      2. mss    — CPU capture via Win32 (fast, ~5-15ms)
      3. ADB    — fallback if window not found

    Interface matches adb_screenshot(): returns PIL.Image (RGB).
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._backend = None
        self._bbox = None
        self._dxcam = None
        self._mss = None
        self._title = None

        self._init_backend()

    def _init_backend(self):
        bbox, title = find_emulator_bbox()

        if bbox is None:
            print("WARNING: ScreenCapture — emulator window not found, falling back to ADB")
            self._backend = 'adb'
            return

        self._bbox = bbox
        self._title = title

        # Try dxcam first
        try:
            import dxcam
            region = (bbox['left'], bbox['top'],
                      bbox['left'] + bbox['width'],
                      bbox['top'] + bbox['height'])
            self._dxcam = dxcam.create(output_color='RGB')
            self._dxcam_region = region
            self._backend = 'dxcam'
            if self.verbose:
                print(f"ScreenCapture: dxcam backend ({title})")
            return
        except Exception:
            pass

        # Fall back to mss
        try:
            import mss as mss_lib
            self._mss_lib = mss_lib
            self._backend = 'mss'
            if self.verbose:
                print(f"ScreenCapture: mss backend ({title})")
            return
        except Exception:
            pass

        print("WARNING: ScreenCapture — dxcam and mss unavailable, falling back to ADB")
        self._backend = 'adb'

    def grab(self):
        """
        Captures the emulator window and returns a PIL.Image (RGB).
        Same interface as adb_screenshot().
        """
        if self._backend == 'dxcam':
            return self._grab_dxcam()
        elif self._backend == 'mss':
            return self._grab_mss()
        else:
            return self._grab_adb()

    def _grab_dxcam(self):
        try:
            frame = self._dxcam.grab(region=self._dxcam_region)
            if frame is None:
                return self._grab_mss()
            return Image.fromarray(frame)
        except Exception as e:
            print(f"WARNING: dxcam grab failed ({e}), falling back to mss")
            self._backend = 'mss'
            return self._grab_mss()

    def _grab_mss(self):
        try:
            with self._mss_lib.mss() as sct:
                frame = sct.grab(self._bbox)
                img = Image.frombytes('RGB', frame.size, frame.bgra, 'raw', 'BGRX')
                return img
        except Exception as e:
            print(f"WARNING: mss grab failed ({e}), falling back to ADB")
            self._backend = 'adb'
            return self._grab_adb()

    def _grab_adb(self):
        import subprocess
        import io
        from clashai.paths import ADB_DEVICE
        try:
            r = subprocess.run(
                ['adb', '-s', ADB_DEVICE, 'exec-out', 'screencap', '-p'],
                capture_output=True, timeout=8
            )
            if r.returncode == 0 and len(r.stdout) > 100:
                return Image.open(io.BytesIO(r.stdout)).convert('RGB')
        except Exception as e:
            print(f"WARNING: ADB screenshot failed: {e}")
        return None

    @property
    def backend(self):
        return self._backend

    def is_direct(self):
        return self._backend in ('dxcam', 'mss')

    def benchmark(self, n=10):
        """Measures average capture time over n frames."""
        import time
        times = []
        for _ in range(n):
            t0 = time.time()
            self.grab()
            times.append((time.time() - t0) * 1000)
        avg = sum(times) / len(times)
        print(f"ScreenCapture [{self._backend}]: avg={avg:.1f}ms over {n} frames")
        return avg


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_instance = None

def get_capture() -> ScreenCapture:
    """Returns the module-level ScreenCapture singleton."""
    global _instance
    if _instance is None:
        _instance = ScreenCapture()
    return _instance


def grab_frame() -> Image.Image:
    """Convenience function — grabs one frame from the emulator."""
    return get_capture().grab()
