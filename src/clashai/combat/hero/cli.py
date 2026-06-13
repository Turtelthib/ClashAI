# clashai/combat/hero/cli.py
# Debug CLI: run the troop bar CNN and show detected hero abilities (*_capa).
#
# V4.4: template matching removed — availability now comes from the YOLO
# troop bar detector. This CLI just exercises that path on a live screen
# capture or a static image file.
#
# Usage:
#   uv run python -m clashai.combat.hero.cli            # live ADB capture
#   uv run python -m clashai.combat.hero.cli --file img.png

import os
import sys

from PIL import Image

from clashai.config import HERO_NAMES
from clashai.paths import WEIGHTS_DIR
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot
from clashai.combat.hero.constants import CAPA_SUFFIX, HERO_ABILITY_NAMES
from clashai.combat.hero.manager import HeroAbilityManager


def _load_detector():
    """Loads the troop bar YOLO detector, or None if the weights are missing."""
    path = os.path.join(WEIGHTS_DIR, 'yolo_troupes_barre', 'troop_bar.pt')
    if not os.path.exists(path):
        print(f"ERROR: troop bar model not found: {path}")
        return None
    from clashai.perception.troop_bar_detector import TroopBarDetector
    return TroopBarDetector(path)


def test_scan(image_path=None):
    """Runs the troop bar CNN and reports which hero abilities are available."""
    print("Testing hero ability detection via troop bar CNN...\n")

    detector = _load_detector()
    if detector is None:
        return

    if image_path:
        if not os.path.exists(image_path):
            print(f"ERROR: image not found: {image_path}")
            return
        img = Image.open(image_path).convert('RGB')
    else:
        img = _adb_screenshot()
        if img is None:
            print("ERROR: unable to capture the screen")
            return

    detections = detector.detect(img)
    capa = [d for d in detections if d['name'].endswith(CAPA_SUFFIX)]

    print(f" {len(detections)} troop-bar detections, {len(capa)} ability (*_capa):")
    for d in capa:
        hero = d['name'][:-len(CAPA_SUFFIX)]
        state = 'GRAYED (used/cooldown)' if d['is_grayed'] else 'AVAILABLE'
        in_space = '' if hero in HERO_NAMES else '  [not in action space]'
        print(f"   {d['name']:24s} {state:22s} center={d['center']} "
              f"conf={d['conf']:.2f}{in_space}")

    manager = HeroAbilityManager(verbose=True)
    manager.reset()
    found = manager.update_from_troop_bar(detections)

    print(f"\nAvailable abilities: {found}")
    print(f" Status: {manager.get_status_vector()}")
    print(f" Mask  : {manager.get_ability_mask()}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ClashAI Hero Ability — troop bar CNN test")
    parser.add_argument('--file', type=str, default=None,
                        help="Run on a static image instead of a live capture")
    parser.add_argument('--test', action='store_true',
                        help="(default) Detect abilities on the current screen")
    args = parser.parse_args()

    if args.file or args.test or len(sys.argv) == 1:
        test_scan(args.file)
    else:
        # Offline sanity check (no ADB / no model)
        print("Offline HeroAbilityManager check\n")
        manager = HeroAbilityManager(verbose=True)
        manager.reset()
        fake = [
            {'name': 'roi_capa',   'center': (300, 980), 'conf': 0.9, 'is_grayed': False},
            {'name': 'reine_capa', 'center': (380, 980), 'conf': 0.9, 'is_grayed': True},
        ]
        found = manager.update_from_troop_bar(fake)
        print(f"\n Available: {found}  (reine grayed → excluded)")
        print(f" Mask: {manager.get_ability_mask()}")
        manager.activate('roi', lambda x, y: print(f" TAP ({x}, {y})"))
        print(f" Status: {manager.get_status_vector()}")
        print(f" Ability names: {HERO_ABILITY_NAMES}")
