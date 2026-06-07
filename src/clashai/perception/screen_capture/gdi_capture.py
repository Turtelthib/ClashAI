# clashai/perception/screen_capture/gdi_capture.py
# Pure Win32 GDI PrintWindow capture of a single HWND.
#
# Captures a window's contents even when it's behind another window, via
# PrintWindow + PW_RENDERFULLCONTENT (works for GDI-rendered windows; for
# DirectX surfaces use the WGC backend instead).

import ctypes
import ctypes.wintypes

import numpy as np
from PIL import Image


def printwindow_single(hwnd):
    """One-shot PrintWindow capture of a specific HWND.

    Returns a PIL.Image (RGB) or None (minimized window, too small, or
    GDI failure). Stateless — used both by the render-HWND probe and by
    the ScreenCapture printwindow backend.
    """
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    if user32.IsIconic(hwnd):
        return None

    rect = ctypes.wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
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
    except Exception:
        return None
    finally:
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)
