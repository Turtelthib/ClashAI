# tools/capture_troop_bar.py
# Dataset capture tool for the troop bar YOLO model.
#
# Captures full screenshots and saves them to data_source/troop_bar/.
# Run during:
#   - prep_attaque screen (all slots colored, clean view)
#   - During combat (progressive depletion, some grayed)
#
# Usage:
#   uv run python tools/capture_troop_bar.py
#   uv run python tools/capture_troop_bar.py --interval 1.0
#   uv run python tools/capture_troop_bar.py --max 200

import os
import sys
import time
import argparse
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

OUTPUT_DIR = os.path.join(project_root, 'data_source', 'troop_bar')


def main():
    parser = argparse.ArgumentParser(description='Capture troop bar dataset')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Seconds between captures (default: 2.0)')
    parser.add_argument('--max', type=int, default=500,
                        help='Max captures (default: 500)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from clashai.perception.screen_capture import get_capture
    cap = get_capture()
    print(f"Backend: {cap.backend}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Interval: {args.interval}s | Max: {args.max}")
    print()
    print("Instructions:")
    print("  1. Go to prep_attaque screen — capture ~100 shots (all slots visible)")
    print("  2. Start an attack — capture ~100 shots during deploy (slots depleting)")
    print("  3. Press Ctrl+C when done")
    print()

    count = 0
    try:
        while count < args.max:
            t0 = time.time()

            img = cap.grab()
            if img is None:
                print("WARNING: capture failed, retrying...")
                time.sleep(1.0)
                continue

            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
            fname = f'bar_{ts}_{count:04d}.png'
            path = os.path.join(OUTPUT_DIR, fname)
            img.save(path)
            count += 1

            elapsed = time.time() - t0
            print(f"[{count:3d}/{args.max}] {fname} ({img.size[0]}x{img.size[1]}, {elapsed*1000:.0f}ms)")

            sleep = max(0.0, args.interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass

    print(f"\nDone — {count} screenshots saved to {OUTPUT_DIR}")
    print("Next: upload to Roboflow, annotate, export YOLO Detection format to datasets/troop_bar/")
    print("Then: uv run python tools/train_yolo_troop_bar.py")


if __name__ == '__main__':
    main()
