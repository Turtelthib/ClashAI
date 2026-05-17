# tools/capture_screen_state.py
# Captures screenshots for a specific screen state class.
# Saves to datasets/dataset_screen/<state>/
#
# Usage:
#   uv run python tools/capture_screen_state.py --state recherche_adversaire
#   uv run python tools/capture_screen_state.py --state gdc_ended
#
# Navigate to the target screen in CoC, then press ENTER to capture.
# Press Q to quit.

import os
import sys
import time
import argparse
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

VALID_STATES = [
    'chargement', 'chat_clan', 'gdc_ally', 'gdc_ended', 'gdc_enemy',
    'menu_boutique', 'phase_attaque', 'prep_attaque', 'profil',
    'recherche_adversaire', 'resultats_attaque', 'village_home',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True, choices=VALID_STATES,
                        help='Screen state to capture')
    parser.add_argument('--auto', type=float, default=0,
                        help='Auto-capture every N seconds (0 = manual)')
    args = parser.parse_args()

    out_dir = os.path.join(project_root, 'datasets', 'dataset_screen', args.state)
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"State: {args.state}")
    print(f"Output: {out_dir}")
    print(f"Existing images: {existing}")
    print()

    from clashai.perception.screen_capture import get_capture
    cap = get_capture()
    print(f"Capture backend: {cap.backend}")
    print()

    if args.auto > 0:
        print(f"Auto-capture every {args.auto}s — navigate to '{args.state}' screen")
        print("Press Ctrl+C to stop")
        count = 0
        try:
            while True:
                img = cap.grab()
                if img:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
                    path = os.path.join(out_dir, f'{args.state}_{ts}.png')
                    img.save(path)
                    count += 1
                    print(f"[{existing + count}] {path}")
                time.sleep(args.auto)
        except KeyboardInterrupt:
            print(f"\nDone — {count} images captured")
    else:
        print(f"Manual mode — navigate to '{args.state}' screen")
        print("Press ENTER to capture, Q+ENTER to quit")
        count = 0
        while True:
            key = input(f"[{existing + count} saved] ENTER=capture, Q=quit > ").strip().lower()
            if key == 'q':
                break
            img = cap.grab()
            if img:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
                path = os.path.join(out_dir, f'{args.state}_{ts}.png')
                img.save(path)
                count += 1
                print(f"  Saved: {path} ({img.size[0]}x{img.size[1]})")
            else:
                print("  WARNING: capture failed")

        print(f"\nDone — {count} images captured ({existing + count} total)")
        print(f"Retrain: uv run python tools/train_screen_cnn.py")


if __name__ == '__main__':
    main()
