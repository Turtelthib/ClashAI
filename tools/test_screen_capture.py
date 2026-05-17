# tools/test_screen_capture.py
# Manual sanity check for ScreenCapture / PrintWindow backend.
#
# Run:
#   uv run python tools/test_screen_capture.py
#
# Then arrange your windows in 3 configurations and press Enter between each:
#   1. Emulator window visible on top
#   2. Emulator window behind another (e.g. VS Code in front, emulator behind)
#   3. (optional) Emulator window partially hidden
#
# Outputs:
#   _test_capture_1_visible.png
#   _test_capture_2_occluded.png
#   _test_capture_3_extra.png
#
# Open each PNG and verify you see the *full emulator window contents* — not
# the desktop, not VS Code, not a black image.

from clashai.perception.screen_capture import ScreenCapture


def main():
    cap = ScreenCapture(verbose=True)
    print(f"\nBackend selected: {cap.backend}")
    print(f"Window: {cap._title}")
    print(f"HWND: {getattr(cap, '_hwnd', None)}\n")

    scenarios = [
        ("1_visible",  "Bring the emulator window to the FRONT, then press Enter"),
        ("2_occluded", "Put another window (VS Code, browser) IN FRONT of the emulator, then press Enter"),
        ("3_extra",    "Any other arrangement you want to test, then press Enter (or Ctrl+C to stop)"),
    ]

    for tag, instruction in scenarios:
        try:
            input(f"\n>>> {instruction}: ")
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
            return

        img = cap.grab()
        if img is None:
            print(f"  [{tag}] grab returned None — capture failed")
            continue
        path = f"_test_capture_{tag}.png"
        img.save(path)
        print(f"  [{tag}] saved {path}  ({img.size[0]}x{img.size[1]})")

    print("\nDone. Open the three PNGs and check each one shows the emulator contents.")


if __name__ == "__main__":
    main()
