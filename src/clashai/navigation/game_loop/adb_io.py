# clashai/navigation/game_loop/adb_io.py
# ADB I/O convenience wrappers (route through clashai.adb.ADBClient).
# adb_screenshot() prefers direct WGC capture, falls back to ADB.


def adb_check_connection():
    """Checks that ADB is connected to the configured device."""
    from clashai.adb import get_client
    return get_client().check_connection()


def adb_screenshot():
    """
    Captures the emulator screen and returns a PIL Image (1920x1080).

    Priority: direct window capture via screen_capture (WGC, ~5-15ms) → ADB
    PNG (~150ms) fallback. The direct backend is initialised once and
    reused across calls (see clashai/perception/screen_capture.py).
    """
    from clashai.perception.screen_capture import get_capture
    try:
        img = get_capture().grab()
        if img is not None:
            return img.convert('RGB')
    except Exception as e:
        print(f"WARNING: Direct capture failed ({e}), falling back to ADB")

    # ADB fallback — works even if WGC fails or the window is unavailable.
    # Routes through ADBClient (Phase C.1).
    from clashai.adb import get_client
    return get_client().screencap()


# Phase C.1: ADB I/O routed through clashai.adb.ADBClient. These wrappers
# are kept as thin convenience functions so existing callers
# (env, brain, agents, tools) don't need to change. New code should prefer
# `from clashai.adb import get_client; client.tap(...)`.

def adb_tap(x, y):
    """Performs a tap at position (x, y)."""
    from clashai.adb import get_client
    get_client().tap(x, y)


def adb_swipe(x1, y1, x2, y2, duration_ms=300):
    """Performs a swipe."""
    from clashai.adb import get_client
    get_client().swipe(x1, y1, x2, y2, duration_ms=duration_ms)


def adb_key(keycode):
    """Envoie une touche (ex: KEYCODE_BACK)."""
    from clashai.adb import get_client
    get_client().keyevent(keycode)
