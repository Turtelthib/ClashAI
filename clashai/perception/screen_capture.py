# clashai/perception/screen_capture.py
# Direct emulator window capture — bypasses ADB for perception.
#
# Captures the emulator window directly via Windows screen APIs.
# ~5-10ms per frame vs ~150ms for ADB screencap PNG.
# Taps still go through ADB (only way to send inputs to Android).
#
# Priority: WGC > PrintWindow > dxcam > mss > ADB
#   - WGC (Windows.Graphics.Capture) is the only backend that works when the
#     emulator is occluded *and* uses hardware-accelerated rendering (Google
#     Play Games, BlueStacks, LDPlayer…). Needs the `windows-capture` package.
#   - dxcam/mss read the physical screen, so they break if anything is in
#     front of the emulator window.
#
# Usage:
#   from clashai.perception.screen_capture import ScreenCapture
#   cap = ScreenCapture()
#   img = cap.grab()   # PIL.Image, same interface as adb_screenshot()

import atexit
import ctypes
import ctypes.wintypes
import threading
import time
import numpy as np
from PIL import Image

# Window detection constants re-imported from clashai/config/window.py (Phase A).
from clashai.config import (
    EMULATOR_WINDOW_KEYWORDS,
    EXCLUDED_TITLE_SUBSTRINGS,
    title_is_excluded as _title_is_excluded,
)




# =============================================================================
# WINDOW DETECTION
# =============================================================================

def find_emulator_bbox():
    """
    Returns (bbox_dict, title, hwnd) for the emulator window.
    Returns (None, None, None) if not found.
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
            # Skip editor / browser tabs that contain the keyword as text,
            # and path-like titles (adbproxy.exe etc.)
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
                    break  # Avoid duplicate entries from multiple matching keywords
        return True  # Always continue to collect all matches

    user32.EnumWindows(EnumProc(_cb), 0)
    if not found:
        return None, None, None
    # Pick the largest matching window (avoids tiny ghost windows)
    hwnd, left, top, right, bottom, title = max(
        found, key=lambda x: (x[3] - x[1]) * (x[4] - x[2])
    )
    return ({'left': left, 'top': top, 'width': right - left, 'height': bottom - top},
            title, hwnd)


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
        self._wgc = None
        self._wgc_control = None  # CaptureControl returned by start_free_threaded
        self._wgc_latest = None
        self._wgc_lock = threading.Lock()
        self._hwnd = None

        # V5.3 push pipeline: any number of callbacks fire on each new
        # frame. For WGC, callbacks fire on the Rust capture thread
        # (native push). For non-WGC backends, a fallback polling thread
        # is started on first subscription and emulates push at ~30fps.
        self._frame_callbacks = []
        self._frame_callbacks_lock = threading.Lock()
        self._fallback_poll_thread = None
        self._fallback_poll_stop = threading.Event()
        self._fallback_poll_fps = 30

        self._init_backend()

    def _init_backend(self):
        bbox, title, hwnd = find_emulator_bbox()

        if bbox is None:
            print("WARNING: ScreenCapture — emulator window not found, falling back to ADB")
            self._backend = 'adb'
            return

        self._bbox = bbox
        self._title = title
        # Keep the hwnd of whatever window find_emulator_bbox picked — used by
        # _normalize_to_canonical to crop the title bar even when WGC/PrintWindow
        # init fails and we fall back to mss/dxcam.
        self._hwnd = hwnd

        # Find a HWND we can hand to WGC / PrintWindow. We keep both the parent
        # and (for fallback PrintWindow) the best-render child window.
        parent = self._find_hwnd()
        if parent:
            self._hwnd = parent

        # WGC (Windows.Graphics.Capture) first — only backend that works for
        # hardware-accelerated emulators like Google Play Games even when
        # they're behind another window. Requires the `windows-capture` pkg.
        if parent and self._init_wgc(parent):
            self._backend = 'wgc'
            if self.verbose:
                print(f"ScreenCapture: WGC backend ({title})")
            return

        # PrintWindow fallback — works for occluded windows that render via
        # GDI, but NOT for DirectX/Vulkan surfaces (Google Play Games will
        # return the screen content here, hence WGC above).
        if parent:
            self._hwnd = self._pick_best_render_hwnd(parent)
            self._backend = 'printwindow'
            if self.verbose:
                print(f"ScreenCapture: PrintWindow backend ({title}) — using hwnd={self._hwnd}")
            return

        # Fall back to dxcam (GPU, fast — only works if window is on top)
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

        # Fall back to mss (only works when emulator is visible)
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

    def _find_hwnd(self):
        """
        Find the emulator main window handle.
        Filters out background processes (invisible windows, path-like titles).
        Picks the largest visible window matching the keywords.
        """
        user32 = ctypes.windll.user32
        found = []
        EnumProc = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        def _cb(hwnd, _):
            # Skip invisible and minimized windows
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n < 3:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            # Skip path-like titles (background processes like adbproxy.exe)
            if '\\' in title or '.exe' in title.lower() or '.dll' in title.lower():
                return True
            # Skip editor / browser windows that contain the keyword as text
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
        # Pick the largest matching visible window
        found.sort(key=lambda x: -x[1])
        return found[0][0]

    def _pick_best_render_hwnd(self, parent_hwnd):
        """
        Among parent_hwnd and every descendant, probe each with a single
        PrintWindow call and pick the HWND whose capture has the highest
        pixel variance — that's the window where the game is actually
        rendered (parent often has a transparent client area when the
        rendering surface is a Crosvm/DirectX child window).
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
            img = self._printwindow_single(hwnd)
            if img is None:
                continue
            arr = np.asarray(img)
            # Variance + mean — a uniformly black/white frame fails on both;
            # a real game frame has both high variance and a non-extreme mean.
            score = float(arr.std())
            if self.verbose:
                print(f"  probe hwnd={hwnd} → variance={score:.1f}")
            if score > best_score:
                best_score = score
                best_hwnd = hwnd

        return best_hwnd

    def _printwindow_single(self, hwnd):
        """One-shot PrintWindow capture of a specific HWND. Returns PIL.Image
        or None. Used both by the probe loop and (with self._hwnd) by grab()."""
        user32 = ctypes.windll.user32
        gdi32  = ctypes.windll.gdi32

        if user32.IsIconic(hwnd):
            return None

        rect = ctypes.wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        width  = rect.right  - rect.left
        height = rect.bottom - rect.top
        if width < 100 or height < 100:
            return None

        hwnd_dc = user32.GetWindowDC(hwnd)
        if not hwnd_dc:
            return None
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        old_bitmap = gdi32.SelectObject(mem_dc, bitmap)

        try:
            PW_RENDERFULLCONTENT = 0x02
            ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
            if not ok:
                user32.PrintWindow(hwnd, mem_dc, 0)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ('biSize', ctypes.c_uint32), ('biWidth', ctypes.c_int32),
                    ('biHeight', ctypes.c_int32), ('biPlanes', ctypes.c_uint16),
                    ('biBitCount', ctypes.c_uint16), ('biCompression', ctypes.c_uint32),
                    ('biSizeImage', ctypes.c_uint32), ('biXPelsPerMeter', ctypes.c_int32),
                    ('biYPelsPerMeter', ctypes.c_int32), ('biClrUsed', ctypes.c_uint32),
                    ('biClrImportant', ctypes.c_uint32),
                ]
            bmi = BITMAPINFOHEADER()
            bmi.biSize        = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth       = width
            bmi.biHeight      = -height
            bmi.biPlanes      = 1
            bmi.biBitCount    = 32
            bmi.biCompression = 0

            buf = (ctypes.c_ubyte * (width * height * 4))()
            scanned = gdi32.GetDIBits(mem_dc, bitmap, 0, height, buf, ctypes.byref(bmi), 0)
            if scanned == 0:
                return None
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            return Image.fromarray(arr[:, :, [2, 1, 0]])
        except Exception:
            return None
        finally:
            gdi32.SelectObject(mem_dc, old_bitmap)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)

    # Canonical resolution that all downstream code (CNN screen classifier,
    # YOLO buildings, button positions, troop bar) expects. ADB returns this
    # natively; the Windows-side backends (WGC / PrintWindow / dxcam / mss)
    # capture the whole window incl. title bar at the OS-side resolution and
    # need to be normalised to match.
    CANONICAL_W, CANONICAL_H = 1920, 1080

    def grab(self):
        """
        Captures the emulator window and returns a PIL.Image (RGB) at the
        canonical 1920x1080 game-content resolution (matches ADB output).
        Works even when the emulator is in the background.
        """
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
        return self._normalize_to_canonical(img)

    def _normalize_to_canonical(self, img):
        """
        Convert a raw window capture into the canonical 1920x1080 game frame
        that downstream code expects (matches ADB's `screencap -p` output).

        The raw capture covers the FULL window (with the Windows title bar
        on top + a few pixels of border) at the OS-side resolution, which
        may be DPI-scaled (e.g. 2560x1528 on a 200% display while the
        logical window is 1280x764). We:
          1. Use GetClientRect + ClientToScreen to find where the game's
             client area sits inside the window in logical pixels.
          2. Scale that rectangle to the captured image's actual pixel size.
          3. Crop to that region → game content only (no title bar).
          4. Resize to 1920x1080 → matches ADB.
        """
        if self._hwnd is None:
            return img.resize((self.CANONICAL_W, self.CANONICAL_H), Image.LANCZOS)

        user32 = ctypes.windll.user32
        try:
            win_rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(self._hwnd, ctypes.byref(win_rect))
            win_w = win_rect.right - win_rect.left
            win_h = win_rect.bottom - win_rect.top

            cli_rect = ctypes.wintypes.RECT()
            user32.GetClientRect(self._hwnd, ctypes.byref(cli_rect))
            cli_w = cli_rect.right - cli_rect.left
            cli_h = cli_rect.bottom - cli_rect.top

            pt = ctypes.wintypes.POINT(0, 0)
            user32.ClientToScreen(self._hwnd, ctypes.byref(pt))

            # Offsets of the client area inside the window, in logical px
            off_x = pt.x - win_rect.left
            off_y = pt.y - win_rect.top

            img_w, img_h = img.size
            # If the captured image is larger than the window rect, the OS
            # is doing DPI scaling — apply the same factor to the crop box.
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

        if cropped.size != (self.CANONICAL_W, self.CANONICAL_H):
            cropped = cropped.resize((self.CANONICAL_W, self.CANONICAL_H), Image.LANCZOS)
        return cropped

    # ------------------------------------------------------------------
    # V5.3 push pipeline — frame subscription
    # ------------------------------------------------------------------

    def subscribe_to_frames(self, callback):
        """
        Register a callback fired on every new captured frame.

        Signature: callback(img_pil) where img_pil is a 1920x1080 RGB
        PIL.Image (already normalized — matches what grab() returns).

        On WGC backend, callbacks fire on the Rust capture thread at the
        emulator's native frame rate (typically 30-60 fps). On other
        backends, a fallback polling thread is started on first
        subscription and emulates push at ~30fps.

        Subscribers must be FAST (microseconds): read latest frame,
        enqueue work, return. Any blocking will stall the capture
        pipeline. Exceptions are caught and logged so a buggy subscriber
        cannot kill the producer thread.
        """
        with self._frame_callbacks_lock:
            if callback not in self._frame_callbacks:
                self._frame_callbacks.append(callback)
        # For non-WGC backends, kick off the polling fallback so the
        # subscriber actually receives frames.
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
        """Dispatch a ready-to-use PIL.Image to every frame subscriber."""
        with self._frame_callbacks_lock:
            callbacks = list(self._frame_callbacks)
        for cb in callbacks:
            try:
                cb(img_pil)
            except Exception as e:
                # Don't let a buggy subscriber crash the WGC thread.
                cb_name = getattr(cb, '__name__', repr(cb))
                print(f"WARNING: frame subscriber {cb_name} raised: {e}")

    def _fire_frame_callbacks_from_bgra(self, bgra):
        """WGC arrives with raw BGRA. Convert + normalise once, then
        dispatch the same PIL.Image to every subscriber."""
        try:
            rgb = bgra[:, :, [2, 1, 0]]
            img = Image.fromarray(rgb)
            img = self._normalize_to_canonical(img)
        except Exception as e:
            print(f"WARNING: BGRA→PIL conversion failed: {e}")
            return
        self._fire_frame_callbacks(img)

    def _start_fallback_polling(self):
        """For backends without a native push event (PrintWindow, dxcam,
        mss, ADB), spin a daemon thread that calls grab() at
        ~_fallback_poll_fps and fires subscriber callbacks."""
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
        """Stop the fallback polling thread (no-op if not running)."""
        self._fallback_poll_stop.set()
        if self._fallback_poll_thread is not None:
            self._fallback_poll_thread.join(timeout=1.0)
            self._fallback_poll_thread = None

    def _init_wgc(self, hwnd):
        """
        Spin up a Windows.Graphics.Capture session against `hwnd` and keep
        the latest frame in self._wgc_latest (BGRA numpy array, written from
        the WGC background thread). Returns True on success.

        WGC is the only Windows API that handles DirectX-rendered emulators
        when their window is occluded — PrintWindow returns the screen
        contents for those.
        """
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
                # frame.frame_buffer is a (H, W, 4) BGRA uint8 view that gets
                # reused for the next frame — copy before releasing the lock.
                buf = frame.frame_buffer
                buf_copy = buf.copy()
                with self._wgc_lock:
                    self._wgc_latest = buf_copy
                # V5.3 push pipeline: notify subscribers natively, no
                # extra polling thread needed. Conversion to PIL +
                # normalisation happens in _fire_frame_callbacks_from_bgra
                # so subscribers receive the canonical 1920x1080 RGB.
                if self._frame_callbacks:
                    self._fire_frame_callbacks_from_bgra(buf_copy)

            @wgc.event
            def on_closed():
                pass

            self._wgc_control = wgc.start_free_threaded()
            self._wgc = wgc

            # Register a cleanup so the Rust capture thread is asked to stop
            # before Python finalises the interpreter — without this we hit
            # `Fatal Python error: PyInterpreterState_Delete: remaining
            # threads` on Ctrl+C.
            atexit.register(self._stop_wgc)

            # Wait briefly for the first frame so the first grab() works
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
        """Idempotent shutdown of the WGC capture thread, registered via
        atexit. Bare try/except — by the time we run during interpreter
        finalisation, modules may already be partially gone."""
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

    def _grab_wgc(self):
        with self._wgc_lock:
            buf = self._wgc_latest
        if buf is None:
            return self._grab_adb()
        # BGRA → RGB (drop alpha)
        return Image.fromarray(buf[:, :, [2, 1, 0]])

    def _grab_printwindow(self):
        """
        Captures the emulator window contents using Win32 PrintWindow API.
        Works even when the window is behind another window (e.g. VS Code).

        Uses the HWND selected at init (parent or whichever descendant had
        the highest pixel variance — see _pick_best_render_hwnd).

        Does NOT work when the window is minimized (Windows stops rendering it).
        """
        user32 = ctypes.windll.user32
        if user32.IsIconic(self._hwnd):
            print("WARNING: emulator window is minimized — restore it (don't close, just un-minimize)")
            return self._grab_adb()
        img = self._printwindow_single(self._hwnd)
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
