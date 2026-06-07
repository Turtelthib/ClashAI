# tools/debug/inspect_emulator_window.py
# Diagnostic: inspect the emulator window's child window hierarchy and
# PrintWindow-capture each candidate, so we can see which HWND actually
# contains the rendered game (parent vs Crosvm/DirectX child).
#
# Run:
#   uv run python tools/debug/inspect_emulator_window.py
#
# Outputs:
#   _inspect_<hwnd>_<class>.png  — one per candidate HWND
#
# Then open each PNG: the one that actually shows the game is the HWND we
# need to use in ScreenCapture.

import ctypes
import ctypes.wintypes
import numpy as np
from PIL import Image

user32 = ctypes.windll.user32
gdi32  = ctypes.windll.gdi32


def _find_parent_hwnd():
    """Phase B.2 thin shim — delegates to the canonical implementation
    in clashai.perception.screen_capture (same match logic, exclusion list,
    and 400x300 minimum size).

    Returns (hwnd, title) for back-compat with this tool's callers, or
    (None, None) if no emulator window is found.
    """
    from clashai.perception.screen_capture import find_emulator_bbox
    bbox, title, hwnd = find_emulator_bbox()
    return (hwnd, title) if hwnd is not None else (None, None)


def _enum_children_recursive(parent_hwnd):
    """Return list of (hwnd, class_name, title, width, height) for every
    descendant of parent_hwnd, walked recursively."""
    out = []
    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def _info(hwnd):
        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buf, 256)
        cls = cls_buf.value

        n = user32.GetWindowTextLengthW(hwnd)
        title = ''
        if n > 0:
            tbuf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, tbuf, n + 1)
            title = tbuf.value

        rect = ctypes.wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        return cls, title, w, h

    def _cb(hwnd, _):
        cls, title, w, h = _info(hwnd)
        out.append((hwnd, cls, title, w, h))
        # Recurse into this child's own children
        user32.EnumChildWindows(hwnd, EnumProc(_cb), 0)
        return True

    user32.EnumChildWindows(parent_hwnd, EnumProc(_cb), 0)
    return out


def _grab_printwindow(hwnd, width, height):
    """Capture a single hwnd via PrintWindow + PW_RENDERFULLCONTENT.
    Returns PIL.Image (RGB) or None."""
    if width < 50 or height < 50:
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
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0

        buf = (ctypes.c_ubyte * (width * height * 4))()
        scanned = gdi32.GetDIBits(mem_dc, bitmap, 0, height, buf, ctypes.byref(bmi), 0)
        if scanned == 0:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        return Image.fromarray(arr[:, :, [2, 1, 0]])
    finally:
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)


def _safe_filename_chunk(s):
    return ''.join(c if c.isalnum() else '_' for c in s)[:40] or 'unnamed'


def main():
    parent_hwnd, parent_title = _find_parent_hwnd()
    if parent_hwnd is None:
        print("No emulator window found.")
        return

    print(f"Parent window: hwnd={parent_hwnd}  title={parent_title!r}")

    # Parent info
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(parent_hwnd, ctypes.byref(rect))
    pw, ph = rect.right - rect.left, rect.bottom - rect.top
    print(f"  Parent client rect: {pw}x{ph}")

    candidates = [(parent_hwnd, 'PARENT', parent_title, pw, ph)]
    candidates += _enum_children_recursive(parent_hwnd)

    print(f"\nFound {len(candidates)} candidate windows (parent + children):")
    print(f"{'HWND':>10}  {'Class':<28} {'Size':>12}  Title")
    print('-' * 80)
    for hwnd, cls, title, w, h in candidates:
        print(f"{hwnd:>10}  {cls[:28]:<28} {w:>5}x{h:<5}   {title[:30]!r}")

    print("\nCapturing each candidate via PrintWindow...")
    for hwnd, cls, title, w, h in candidates:
        img = _grab_printwindow(hwnd, w, h)
        if img is None:
            print(f"  hwnd={hwnd}  {cls}  -> skipped (size {w}x{h})")
            continue
        arr = np.array(img)
        # Variance is a rough proxy for "did we capture real content"
        var = arr.std()
        fname = f"_inspect_{hwnd}_{_safe_filename_chunk(cls)}.png"
        img.save(fname)
        print(f"  hwnd={hwnd}  {cls:<24}  variance={var:6.1f}  -> {fname}")

    print("\nOpen the _inspect_*.png files. The one showing the actual game")
    print("(without your terminal/VS Code in front) is the HWND we need to use.")
    print("Tell me its hwnd + class name.")


if __name__ == "__main__":
    main()
