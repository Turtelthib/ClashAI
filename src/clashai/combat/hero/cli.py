# clashai/combat/hero/cli.py
# Setup / debug CLI: capture the ability zone + test scanning.

import os
import sys
import time

import cv2
import numpy as np

from clashai.config import HERO_NAMES
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot
from clashai.combat.hero.constants import (
    TEMPLATES_DIR,
    ABILITY_ZONE_TOP, ABILITY_ZONE_BOTTOM, ABILITY_ZONE_LEFT, ABILITY_ZONE_RIGHT,
)
from clashai.combat.hero.manager import HeroAbilityManager


def extract_ability_zone():
    """
    Captures a mid-combat screenshot and saves the ability icon zone
    for manual cropping.

    WARNING: Heroes must be DEPLOYED on the battlefield!
    Ability icons only appear when the hero is in combat,
    not when still in the troop bar.
    """
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    print("Capturing hero ability zone...")
    print()
    print(" WARNING: Heroes must be DEPLOYED!")
    print(" WARNING: Not in the troop bar, but ON the battlefield.")
    print(" WARNING: Ability icons only appear after deployment.")
    print()

    img = _adb_screenshot()
    if img is None:
        print("ERROR: Unable to capture the screen")
        return

    screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = screen.shape[:2]

    full_path = os.path.join(TEMPLATES_DIR, '_combat_full.png')
    cv2.imwrite(full_path, screen)
    print(f" Full screen → {full_path}")

    # Troop bar zone (bottom of screen)
    zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                   ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]
    zone_path = os.path.join(TEMPLATES_DIR, '_ability_zone.png')
    cv2.imwrite(zone_path, zone)
    print(f" Bar zone (y={ABILITY_ZONE_TOP}-{ABILITY_ZONE_BOTTOM}) → {zone_path}")

    # Expanded zone: entire bottom half of the screen
    bottom_half = screen[h // 2:, :]
    bottom_path = os.path.join(TEMPLATES_DIR, '_bottom_half.png')
    cv2.imwrite(bottom_path, bottom_half)
    print(f" Bottom half (y={h//2}-{h}) → {bottom_path}")

    print("\nNext steps:")
    print(f" 1. Open {full_path} or {zone_path}")
    print(" 2. Locate the ABILITY icons (portraits of deployed heroes)")
    print(" WARNING: These are NOT the troop bar cards!")
    print(" WARNING: Abilities are small portraits that appear")
    print(" AFTER deploying the hero on the battlefield.")
    print(f" 3. Crop each icon and save in {TEMPLATES_DIR}/:")
    for hero in HERO_NAMES:
        print(f" ability_{hero}.png")
    print("\n If you don't see ability icons, the heroes were not yet deployed")
    print(" at the time of capture!")


def test_scan():
    """Tests ability scanning on a live screenshot."""
    print("Testing ability scan...\n")

    manager = HeroAbilityManager()
    if not manager.has_templates():
        print("ERROR: No templates found. Run --extract and crop the icons.")
        return

    img = _adb_screenshot()
    if img is None:
        print("ERROR: Unable to capture the screen")
        return

    # Save screenshot for debugging
    debug_path = os.path.join(TEMPLATES_DIR, '_test_screenshot.png')
    screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(debug_path, screen)

    # Save scanned zone for verification
    zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                   ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]
    zone_path = os.path.join(TEMPLATES_DIR, '_test_zone.png')
    cv2.imwrite(zone_path, zone)

    print(f" Scanned zone: y={ABILITY_ZONE_TOP}-{ABILITY_ZONE_BOTTOM}, "
          f"x={ABILITY_ZONE_LEFT}-{ABILITY_ZONE_RIGHT}")
    print(f" Screenshot → {debug_path}")
    print(f" Zone → {zone_path}")

    # Simulate all heroes deployed (for testing)
    for hero in HERO_NAMES:
        manager._deployed[hero] = True
        manager._deploy_time[hero] = time.time() - 30

    found = manager.scan(img)

    if found:
        print(f"\nDetected abilities: {found}")
        print(f" Status: {manager.get_status_vector()}")
        print(f" Mask: {manager.get_ability_mask()}")
    else:
        print("\nWARNING: No abilities detected.")
        print(" Possible causes:")
        print(" 1. Heroes are not deployed (still in the troop bar)")
        print(" 2. Templates do not match the ability icons")
        print(" → templates must be icons captured AFTER deployment")
        print(" → not the troop bar cards")
        print(f" 3. Check {zone_path} to see what the scan sees")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClashAI Hero Ability Manager")
    parser.add_argument('--extract', action='store_true',
                        help="Capture the ability zone for templates")
    parser.add_argument('--test', action='store_true',
                        help="Test the scan on the current screen")
    args = parser.parse_args()

    if args.extract:
        extract_ability_zone()
    elif args.test:
        test_scan()
    else:
        print("Test HeroAbilityManager (without ADB)\n")
        manager = HeroAbilityManager(verbose=True)
        manager.reset()

        manager.mark_deployed('roi')
        manager.mark_deployed('reine')
        manager.mark_deployed('grand_gardien')
        manager.mark_deployed('prince_gargouille')

        print(f"\n Deployed: {manager.num_deployed()}/4")
        print(f" Status : {manager.get_status_vector()}")

        for name in ['roi', 'reine']:
            manager._deploy_time[name] = time.time() - 20
            manager._icon_positions[name] = (100, 600, 0.85)

        print("\n After simulated scan:")
        print(f" Available: {manager.get_available_abilities()}")
        print(f" Mask : {manager.get_ability_mask()}")

        manager.activate('roi', lambda x, y: print(f" TAP ({x}, {y})"))
        print(f" Status : {manager.get_status_vector()}")
        print("\nTest complete!")
