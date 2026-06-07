# clashai/perception/screen_capture/capture.py
# ScreenCapture — direct emulator window capture, bypassing ADB.
#
# ~5-10ms per frame vs ~150ms for ADB screencap PNG. Taps still go
# through ADB (only way to send inputs to Android).
#
# Backend priority: WGC > PrintWindow > dxcam > mss > ADB
#   - WGC (Windows.Graphics.Capture) is the only backend that works when
#     the emulator is occluded AND hardware-accelerated (Google Play
#     Games, BlueStacks, LDPlayer…). Needs the `windows-capture` package.
#   - dxcam/mss read the physical screen → break if anything is in front
#     of the emulator window.
#
# Pure helpers live in sibling modules:
#   window_detect.py — find the emulator HWND
#   gdi_capture.py   — PrintWindow GDI capture of one HWND
#   normalize.py     — crop+resize a raw capture to the canonical frame

import atexit
import ctypes
import threading
import time

import numpy as np
from PIL import Image

from clashai.perception.screen_capture.window_detect import (
    find_emulator_bbox, find_hwnd, pick_best_render_hwnd,
)
from clashai.perception.screen_capture.gdi_capture import printwindow_single
from clashai.perception.screen_capture.normalize import (
    normalize_to_canonical, CANONICAL_W, CANONICAL_H,
)


class ScreenCapture:
    """
    Direct window capture with automatic backend selection.

    Interface matches adb_screenshot(): grab() returns a 1920x1080 RGB
    PIL.Image. Also exposes a push pipeline (subscribe_to_frames) for the
    V5.3 event-driven perception.
    """

    CANONICAL_W = CANONICAL_W
    CANONICAL_H = CANONICAL_H

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._backend = None
        self._bbox = None
        self._dxcam = None
        self._mss = None
        self._title = None
        self._wgc = None
        self._wgc_control = None
        self._wgc_latest = None
        self._wgc_lock = threading.Lock()
        self._hwnd = None

        # V5.3 push pipeline
        self._frame_callbacks = []
        self._frame_callbacks_lock = threading.Lock()
        self._fallback_poll_thread = None
        self._fallback_poll_stop = threading.Event()
        self._fallback_poll_fps = 30

        self._init_backend()

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------

    def _init_backend(self):
        bbox, title, hwnd = find_emulator_bbox()

        if bbox is None:
            print("WARNING: ScreenCapture — emulator window not found, falling back to ADB")
            self._backend = 'adb'
            return

        self._bbox = bbox
        self._title = title
        # Keep the picked hwnd for normalize even if WGC/PrintWindow fail
        # and we fall back to mss/dxcam.
        self._hwnd = hwnd

        parent = find_hwnd()
        if parent:
            self._hwnd = parent

        # WGC first — only backend for occluded hardware-accelerated emulators.
        if parent and self._init_wgc(parent):
            self._backend = 'wgc'
            if self.verbose:
                print(f"ScreenCapture: WGC backend ({title})")
            return

        # PrintWindow fallback — GDI windows, occluded OK; not DirectX.
        if parent:
            self._hwnd = pick_best_render_hwnd(parent, verbose=self.verbose)
            self._backend = 'printwindow'
            if self.verbose:
                print(f"ScreenCapture: PrintWindow backend ({title}) — using hwnd={self._hwnd}")
            return

        # dxcam (GPU, fast — only if window is on top).
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

        # mss (only works when emulator is visible).
        try:
            import mss as mss_lib
            self._mss_lib = mss_lib
            self._backend = 'mss'
            if self.verbose:
                print(f"ScreenCapture: mss backend ({title})")
            return
        except Exception:
            pass

        print("WARNING: ScreenCapture — falling back to ADB")
        self._backend = 'adb'

    # ------------------------------------------------------------------
    # Public grab
    # ------------------------------------------------------------------

    def grab(self):
        """Capture the emulator at the canonical 1920x1080 game resolution."""
        if self._backend == 'wgc':
            img = self._grab_wgc()
        elif self._backend == 'printwindow':
            img = self._grab_printwindow()
        elif self._backend == 'dxcam':
            img = self._grab_dxcam()
        elif self._backend == 'mss':
            img = self._grab_mss()
        else:
            return self._grab_adb()  # already 1920x1080
        if img is None:
            return None
        return normalize_to_canonical(img, self._hwnd)

    # ------------------------------------------------------------------
    # V5.3 push pipeline — frame subscription
    # ------------------------------------------------------------------

    def subscribe_to_frames(self, callback):
        """
        Register a callback fired on every new captured frame.
        Signature: callback(img_pil) — a normalized 1920x1080 RGB image.

        WGC fires on the Rust capture thread (native fps). Other backends
        spin a ~30fps fallback polling thread on first subscription.
        Subscribers must be FAST (read latest, enqueue, return).
        """
        with self._frame_callbacks_lock:
            if callback not in self._frame_callbacks:
                self._frame_callbacks.append(callback)
        if self._backend != 'wgc' and self._fallback_poll_thread is None:
            self._start_fallback_polling()

    def unsubscribe_from_frames(self, callback) -> bool:
        with self._frame_callbacks_lock:
            try:
                self._frame_callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def num_frame_subscribers(self) -> int:
        with self._frame_callbacks_lock:
            return len(self._frame_callbacks)

    def _fire_frame_callbacks(self, img_pil):
        with self._frame_callbacks_lock:
            callbacks = list(self._frame_callbacks)
        for cb in callbacks:
            try:
                cb(img_pil)
            except Exception as e:
                cb_name = getattr(cb, '__name__', repr(cb))
                print(f"WARNING: frame subscriber {cb_name} raised: {e}")

    def _fire_frame_callbacks_from_bgra(self, bgra):
        try:
            rgb = bgra[:, :, [2, 1, 0]]
            img = normalize_to_canonical(Image.fromarray(rgb), self._hwnd)
        except Exception as e:
            print(f"WARNING: BGRA->PIL conversion failed: {e}")
            return
        self._fire_frame_callbacks(img)

    def _start_fallback_polling(self):
        if self._fallback_poll_thread is not None:
            return
        interval = 1.0 / max(1, self._fallback_poll_fps)

        def _loop():
            if self.verbose:
                print(f"ScreenCapture: fallback polling thread started "
                      f"(~{self._fallback_poll_fps}fps) for backend={self._backend}")
            while not self._fallback_poll_stop.is_set():
                t0 = time.time()
                try:
                    img = self.grab()
                    if img is not None:
                        self._fire_frame_callbacks(img)
                except Exception as e:
                    print(f"WARNING: fallback polling grab failed: {e}")
                sleep = interval - (time.time() - t0)
                if sleep > 0:
                    self._fallback_poll_stop.wait(timeout=sleep)

        self._fallback_poll_thread = threading.Thread(
            target=_loop, name='ScreenCaptureFallbackPoll', daemon=True,
        )
        self._fallback_poll_thread.start()

    def stop_fallback_polling(self):
        self._fallback_poll_stop.set()
        if self._fallback_poll_thread is not None:
            self._fallback_poll_thread.join(timeout=1.0)
            self._fallback_poll_thread = None

    # ------------------------------------------------------------------
    # WGC backend
    # ------------------------------------------------------------------

    def _init_wgc(self, hwnd):
        """Start a Windows.Graphics.Capture session against `hwnd`. Keeps
        the latest BGRA frame in self._wgc_latest. Returns True on success."""
        try:
            from windows_capture import WindowsCapture
        except ImportError:
            if self.verbose:
                print("ScreenCapture: windows-capture not installed, skipping WGC")
            return False

        try:
            wgc = WindowsCapture(
                cursor_capture=False,
                draw_border=False,
                window_hwnd=hwnd,
            )

            @wgc.event
            def on_frame_arrived(frame, capture_control):
                buf = frame.frame_buffer
                buf_copy = buf.copy()
                with self._wgc_lock:
                    self._wgc_latest = buf_copy
                if self._frame_callbacks:
                    self._fire_frame_callbacks_from_bgra(buf_copy)

            @wgc.event
            def on_closed():
                pass

            self._wgc_control = wgc.start_free_threaded()
            self._wgc = wgc

            # Stop the Rust capture thread before interpreter finalisation
            # (avoids "Fatal Python error: remaining threads" on Ctrl+C).
            atexit.register(self._stop_wgc)

            deadline = time.time() + 2.0
            while time.time() < deadline:
                with self._wgc_lock:
                    if self._wgc_latest is not None:
                        return True
                time.sleep(0.05)

            if self.verbose:
                print("ScreenCapture: WGC started but no frame arrived within 2s")
            return False
        except Exception as e:
            if self.verbose:
                print(f"ScreenCapture: WGC init failed ({e})")
            return False

    def _stop_wgc(self):
        """Idempotent WGC shutdown, registered via atexit."""
        ctrl = self._wgc_control
        if ctrl is None:
            return
        self._wgc_control = None
        try:
            ctrl.stop()
        except Exception:
            pass
        try:
            ctrl.wait()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-backend grab
    # ------------------------------------------------------------------

    def _grab_wgc(self):
        with self._wgc_lock:
            buf = self._wgc_latest
        if buf is None:
            return self._grab_adb()
        return Image.fromarray(buf[:, :, [2, 1, 0]])

    def _grab_printwindow(self):
        """PrintWindow capture using the HWND picked at init. Works behind
        other windows; not when minimized."""
        if ctypes.windll.user32.IsIconic(self._hwnd):
            print("WARNING: emulator window is minimized — restore it (don't close, just un-minimize)")
            return self._grab_adb()
        img = printwindow_single(self._hwnd)
        if img is None:
            return self._grab_adb()
        return img

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
                return Image.frombytes('RGB', frame.size, frame.bgra, 'raw', 'BGRX')
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

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backend(self):
        return self._backend

    def is_direct(self):
        return self._backend in ('dxcam', 'mss')

    def benchmark(self, n=10):
        """Average capture time over n frames."""
        times = []
        for _ in range(n):
            t0 = time.time()
            self.grab()
            times.append((time.time() - t0) * 1000)
        avg = sum(times) / len(times)
        print(f"ScreenCapture [{self._backend}]: avg={avg:.1f}ms over {n} frames")
        return avg


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

_instance = None


def get_capture() -> ScreenCapture:
    """Returns the module-level ScreenCapture singleton."""
    global _instance
    if _instance is None:
        _instance = ScreenCapture()
    return _instance


def grab_frame() -> Image.Image:
    """Convenience: grab one frame from the emulator."""
    return get_capture().grab()
