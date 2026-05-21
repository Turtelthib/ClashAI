# clashai/adb/client.py
# Thin wrapper around the `adb` binary.
#
# Provides typed methods for the operations the bot actually needs (tap,
# swipe, keyevent, screencap_raw, generic shell + run). Every call
# automatically uses `-s <device>` so multi-emulator setups work, and a
# configurable post-call delay lets the emulator settle before the next
# action.
#
# Other modules should never call `subprocess.run(["adb", ...])` directly
# anymore — instead get the singleton via `get_client()` and use its
# methods.

import io
import subprocess
import time
from typing import Optional

from PIL import Image

from clashai.paths import ADB_DEVICE
from clashai.config import (
    ADB_DELAY_TAP, ADB_DELAY_SCREENSHOT,
)
from clashai.adb.exceptions import (
    ADBError, ADBNotFoundError,
    ADBTimeoutError, ADBNotConnectedError,
)


# Default subprocess timeout (seconds) for short ADB commands.
DEFAULT_TIMEOUT = 5

# Whether the constant after-tap delay is enforced after every tap/swipe
# call (legacy game_loop behaviour). Set to False to bypass for custom
# pacing — the caller can sleep itself.
ENFORCE_TAP_DELAY = True


class ADBClient:
    """
    Stateless wrapper over the `adb` binary, bound to a single device.

    Methods always include `-s <device>` so the same client can coexist
    with other emulators on the same host. Most methods raise the
    appropriate `ADBError` subclass on failure; the `_strict` flag is
    `False` by default for convenience methods (`tap`, `keyevent`…), which
    swallow timeouts and return `False`/`None` to preserve the historic
    fire-and-forget semantics.
    """

    def __init__(self, device: str = ADB_DEVICE,
                 tap_delay: float = ADB_DELAY_TAP):
        self.device = device
        self.tap_delay = tap_delay

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _run(self, args, *, timeout: float = DEFAULT_TIMEOUT,
             capture_output: bool = True, text: bool = False,
             _strict: bool = True) -> subprocess.CompletedProcess:
        """Run an adb command. Prepends `adb` (the `-s <device>` flag is
        the caller's responsibility because some commands like `adb
        connect` / `adb devices` reject `-s`)."""
        try:
            return subprocess.run(
                ['adb'] + list(args),
                capture_output=capture_output,
                text=text,
                timeout=timeout,
            )
        except FileNotFoundError as e:
            raise ADBNotFoundError(
                "adb executable not found in PATH. Install Android Platform Tools."
            ) from e
        except subprocess.TimeoutExpired as e:
            if not _strict:
                # Mimic the legacy fire-and-forget behaviour.
                return subprocess.CompletedProcess(args, returncode=1, stdout=b'', stderr=b'')
            raise ADBTimeoutError(
                f"ADB command timed out after {timeout}s: {' '.join(args)}"
            ) from e

    def _device_run(self, *args, timeout: float = DEFAULT_TIMEOUT,
                    capture_output: bool = True,
                    _strict: bool = True) -> subprocess.CompletedProcess:
        """Run an adb command bound to the configured device (`-s <device>`)."""
        return self._run(
            ['-s', self.device] + list(args),
            timeout=timeout,
            capture_output=capture_output,
            _strict=_strict,
        )

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------

    def check_connection(self, verbose: bool = True) -> bool:
        """Returns True if `adb devices` lists the configured device.

        Does not raise on a missing device (returns False instead) — this
        is the lifecycle check the brain runs at startup."""
        try:
            r = self._run(['devices'], text=True)
        except ADBNotFoundError:
            if verbose:
                print("ERROR: ADB binary not found in PATH.")
            return False
        except ADBTimeoutError:
            if verbose:
                print("ERROR: ADB devices command timed out.")
            return False

        output = (r.stdout or '').replace('\r', '')
        lines = output.strip().split('\n')
        devices = [l.strip() for l in lines if '\tdevice' in l or ' device' in l]
        if any(self.device in d for d in devices):
            if verbose:
                print(f"ADB connected: {self.device}")
            return True
        if verbose:
            if devices:
                connected = [d.split()[0] for d in devices]
                print(f"WARNING: {self.device} not found. Connected: {connected}")
            else:
                print(f"ERROR: No ADB device detected. Run: adb connect {self.device}")
        return False

    def connect(self) -> subprocess.CompletedProcess:
        """`adb connect <device>` — used at brain startup."""
        return self._run(['connect', self.device], text=True)

    def tap(self, x: int, y: int, delay: Optional[float] = None) -> bool:
        """Tap at (x, y). Returns True on subprocess success."""
        r = self._device_run(
            'shell', f'input tap {x} {y}',
            _strict=False,
        )
        if ENFORCE_TAP_DELAY:
            time.sleep(self.tap_delay if delay is None else delay)
        return r.returncode == 0

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 300,
              delay: Optional[float] = None) -> bool:
        """Swipe from (x1, y1) to (x2, y2) over `duration_ms`."""
        r = self._device_run(
            'shell', f'input swipe {x1} {y1} {x2} {y2} {duration_ms}',
            _strict=False,
        )
        if ENFORCE_TAP_DELAY:
            time.sleep(self.tap_delay if delay is None else delay)
        return r.returncode == 0

    def keyevent(self, code: int, delay: Optional[float] = None) -> bool:
        """Send a key event (e.g. KEYCODE_BACK=4)."""
        r = self._device_run(
            'shell', f'input keyevent {code}',
            _strict=False,
        )
        if ENFORCE_TAP_DELAY:
            time.sleep(self.tap_delay if delay is None else delay)
        return r.returncode == 0

    def input_text(self, text: str, delay: float = 0.3) -> bool:
        """`adb shell input text <s>` — type text into the focused field.

        The caller is responsible for sanitising / escaping input. The
        usual `adb input text` pitfalls (spaces must be `%s`, special
        chars need quoting) are NOT applied here.
        """
        r = self._device_run(
            'shell', 'input', 'text', text,
            _strict=False,
        )
        time.sleep(delay)
        return r.returncode == 0

    def shell(self, command: str, timeout: float = DEFAULT_TIMEOUT,
              text: bool = True) -> subprocess.CompletedProcess:
        """Run a generic `adb shell <command>` for cases we don't have a
        dedicated method (e.g. `wm size`, `dumpsys window`...)."""
        return self._run(
            ['-s', self.device, 'shell', command],
            timeout=timeout, text=text,
        )

    def screencap_raw(self, timeout: float = 8) -> Optional[bytes]:
        """Raw PNG bytes from `adb exec-out screencap -p`. Returns None on
        failure — that's the historic semantic and many callers rely on
        it for fallback chains.

        This is the *fallback* path; production capture goes through
        `clashai.perception.screen_capture` (WGC) and only falls back here
        if WGC is unavailable."""
        try:
            r = subprocess.run(
                ['adb', '-s', self.device, 'exec-out', 'screencap', '-p'],
                capture_output=True, timeout=timeout,
            )
            if r.returncode != 0 or len(r.stdout) < 100:
                return None
            return r.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def screencap(self, timeout: float = 8) -> Optional[Image.Image]:
        """`screencap_raw()` decoded into a PIL.Image (RGB)."""
        raw = self.screencap_raw(timeout=timeout)
        if raw is None:
            return None
        return Image.open(io.BytesIO(raw)).convert('RGB')


# -----------------------------------------------------------------------------
# Singleton helper
# -----------------------------------------------------------------------------

_singleton: Optional[ADBClient] = None


def get_client() -> ADBClient:
    """Returns a process-wide ADBClient bound to ADB_DEVICE. Cheap to call
    (no setup) — kept as a singleton so future per-client state (retry
    counters, telemetry) lives in one place."""
    global _singleton
    if _singleton is None:
        _singleton = ADBClient()
    return _singleton
