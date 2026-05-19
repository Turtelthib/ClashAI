# scripts/rl/hero_ability.py
# Management of hero special abilities for ClashAI V3.
#
# DYNAMIC detection of ability icons during combat via
# template matching (same approach as TroopFinder).
#
# Each hero has an activatable ability during combat:
# - Barbarian King: Royal Rage (damage boost + summons)
# - Archer Queen: Royal Cloak (invisibility + targeted shots)
# - Grand Warden: Eternal Tome (nearby troops invincibility)
# - Royal Champion: Seeking Shield (shield + target search)
#
# Note: Max 4 heroes in combat in CoC.
# The Minion Prince is a pet, not a hero with an ability.
#
# Setup (once):
# 1. python scripts/rl/hero_ability.py --extract
# → Captures a mid-combat screenshot and saves the hero zone
# 2. Crop each ability icon and save:
# ability_roi.png, ability_reine.png, etc.
#
# Usage in code:
# manager = HeroAbilityManager()
# manager.scan(screenshot_pil) # Detects present icons
# manager.activate('roi', adb_tap_fn) # Taps on the king's icon

import os
import subprocess
import time

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

from clashai.paths import HERO_TEMPLATES_DIR

TEMPLATES_DIR = HERO_TEMPLATES_DIR

# Screen zone where ability icons appear during combat.
# In combat, AFTER deploying heroes, their ability icons appear
# in the troop bar at the bottom of the screen.
# We scan the entire bottom UI zone to be sure to find them.
# The troop bar is at approximately y=930-1080, but ability icons
# can appear slightly above → we scan wide.
ABILITY_ZONE_TOP = 850
ABILITY_ZONE_BOTTOM = 1080
ABILITY_ZONE_LEFT = 0
ABILITY_ZONE_RIGHT = 1920

# Template matching — MATCH_SCALES from clashai/config/ (Phase A), threshold local.
from clashai.config import MATCH_SCALES  # noqa: E402

MATCH_THRESHOLD = 0.50  # local — hero_ability tuned at 0.50

# The 5 heroes that can have an ability in combat.
# At the start of each attack, TroopFinder detects which heroes are
# in the bar → only deployed heroes will have their ability activatable.
# If the Royal Champion is being upgraded, she won't be in the bar,
# won't be deployed, and her ability will be masked automatically.
# Re-imported from clashai/config/rl.py (Phase A).
from clashai.config import HERO_NAMES, NUM_HEROES  # noqa: E402

HERO_ABILITY_NAMES = {
    'roi': 'Rage Royale',
    'reine': 'Cloak Royal',
    'grand_gardien': 'Tome Éternel',
    'championne': 'Seeking Shield',
    'prince_gargouille': 'Visage Noir',
}

# Minimum delay after deployment before scanning abilities
# (icons do not appear instantly)
DEPLOY_TO_SCAN_DELAY = 5.0

# Cooldown between scans (avoid spamming screenshots)
SCAN_COOLDOWN = 2.0


# =============================================================================
# ADB FUNCTIONS
# =============================================================================

# Re-exported from the canonical implementation in game_loop (Phase B.1).
# That version routes through WGC (fast, occlusion-proof) with ADB fallback.
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot  # noqa: E402


def _adb_tap(x, y, delay=0.1):
    """Tap ADB."""
    subprocess.run(["adb", "shell", f"input tap {x} {y}"],
                   capture_output=True, timeout=5)
    time.sleep(delay)


# =============================================================================
# TEMPLATE MATCHING
# =============================================================================

def _match_template_multiscale(region, template):
    """
    Multi-scale template matching.

    Returns:
        (best_val, best_loc, best_tw, best_th) or (0, None, 0, 0)
    """
    best_val = 0
    best_loc = None
    best_tw = 0
    best_th = 0

    for scale in MATCH_SCALES:
        th, tw = template.shape[:2]
        new_h = int(th * scale)
        new_w = int(tw * scale)

        if new_h > region.shape[0] or new_w > region.shape[1]:
            continue
        if new_h < 10 or new_w < 10:
            continue

        resized = cv2.resize(template, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_tw = new_w
            best_th = new_h

    return best_val, best_loc, best_tw, best_th


# =============================================================================
# HERO ABILITY MANAGER
# =============================================================================

class HeroAbilityManager:
    """
    Manages detection and activation of hero abilities during combat.

    Uses template matching to dynamically find ability icons on screen
    — no hardcoded positions.

    Workflow:
        1. During deploy, call mark_deployed() for each placed hero.
        2. During combat, call scan(screenshot) to detect icons.
        3. Call activate(hero_name) to tap the detected icon.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

        # Ability icon templates
        self._templates = {}
        self._load_templates()

        # Per-episode state
        self._deployed = {}
        self._deploy_time = {}
        self._activated = {}
        self._icon_positions = {}
        self._last_scan_time = 0

    def _load_templates(self):
        """Loads ability icon templates from hero_ability_templates/."""
        if not os.path.exists(TEMPLATES_DIR):
            if self.verbose:
                print("WARNING: hero_ability_templates/ directory not found")
                print(" Run: python scripts/rl/hero_ability.py --extract")
            return

        count = 0
        for hero in HERO_NAMES:
            filename = f"ability_{hero}.png"
            path = os.path.join(TEMPLATES_DIR, filename)
            if os.path.exists(path):
                tmpl = cv2.imread(path)
                if tmpl is not None:
                    self._templates[hero] = tmpl
                    count += 1

        if self.verbose:
            if count > 0:
                print(f"{count} ability templates loaded: "
                      f"{sorted(self._templates.keys())}")
            else:
                print(f"WARNING: No ability templates found in {TEMPLATES_DIR}")

    def reset(self):
        """Reset at the start of a new episode."""
        self._deployed = {name: False for name in HERO_NAMES}
        self._deploy_time = {}
        self._activated = {name: False for name in HERO_NAMES}
        self._icon_positions = {}
        self._last_scan_time = 0

    def mark_deployed(self, hero_name):
        """
        Marks a hero as deployed on the battlefield.
        Note: prince_gargouille is ignored (no ability).
        """
        if hero_name not in HERO_NAMES:
            return

        if not self._deployed.get(hero_name, False):
            self._deployed[hero_name] = True
            self._deploy_time[hero_name] = time.time()

            if self.verbose:
                ability = HERO_ABILITY_NAMES.get(hero_name, '?')
                print(f" {hero_name} deployed (ability: {ability})")

    def scan(self, screenshot_pil):
        """
        Scans a mid-combat screenshot to detect ability icons.
        Updates internal icon positions.

        Args:
            screenshot_pil: PIL Image of the ongoing combat

        Returns:
            found: list of hero names whose icon is visible
        """
        now = time.time()

        # Cooldown between scans
        if now - self._last_scan_time < SCAN_COOLDOWN:
            return list(self._icon_positions.keys())

        self._last_scan_time = now

        if not self._templates:
            return []

        # Convert and crop the ability zone
        screen = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                       ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]

        found = []

        for hero_name, template in self._templates.items():
            # Only search for deployed heroes that haven't been activated yet
            if not self._deployed.get(hero_name, False):
                continue
            if self._activated.get(hero_name, False):
                continue

            # Check post-deploy delay
            deploy_t = self._deploy_time.get(hero_name, now)
            if now - deploy_t < DEPLOY_TO_SCAN_DELAY:
                continue

            best_val, best_loc, best_tw, best_th = _match_template_multiscale(
                zone, template
            )

            if best_val >= MATCH_THRESHOLD and best_loc is not None:
                # Convert zone position → ADB coordinates
                x_adb = ABILITY_ZONE_LEFT + best_loc[0] + best_tw // 2
                y_adb = ABILITY_ZONE_TOP + best_loc[1] + best_th // 2
                self._icon_positions[hero_name] = (x_adb, y_adb, best_val)
                found.append(hero_name)

        if self.verbose and found:
            for name in found:
                x, y, conf = self._icon_positions[name]
                print(f" {name} ability detected "
                      f"at ({x}, {y}) conf={conf:.2f}")

        return found

    def get_available_abilities(self):
        """
        Returns heroes whose ability is available.

        Conditions:
            - Hero deployed
            - Ability not yet used
            - Icon detected by the last scan
        """
        available = []
        for name in HERO_NAMES:
            if not self._deployed.get(name, False):
                continue
            if self._activated.get(name, False):
                continue
            if name in self._icon_positions:
                available.append(name)
        return available

    def get_ability_mask(self):
        """
        Binary mask (NUM_HEROES,) for PPO action masking.
        1.0 = ability available and detected, 0.0 otherwise.
        """
        available = set(self.get_available_abilities())
        mask = np.zeros(NUM_HEROES, dtype=np.float32)
        for i, name in enumerate(HERO_NAMES):
            if name in available:
                mask[i] = 1.0
        return mask

    def activate(self, hero_name, adb_tap_fn=None):
        """
        Activates a hero's ability by tapping on its detected icon.

        Args:
            hero_name: str ('roi', 'reine', etc.)
            adb_tap_fn: callable(x, y) — if None, uses internal _adb_tap

        Returns:
            success: bool
        """
        tap_fn = adb_tap_fn or _adb_tap

        if hero_name not in HERO_NAMES:
            if self.verbose:
                print(f" WARNING: {hero_name} is not a hero with an ability")
            return False

        if not self._deployed.get(hero_name, False):
            if self.verbose:
                print(f" WARNING: {hero_name} not deployed")
            return False

        if self._activated.get(hero_name, False):
            if self.verbose:
                print(f" WARNING: {hero_name}'s ability already used")
            return False

        if hero_name not in self._icon_positions:
            if self.verbose:
                print(f" WARNING: {hero_name}'s icon not detected — "
                      f"re-run scan()")
            return False

        x, y, conf = self._icon_positions[hero_name]

        if self.verbose:
            ability = HERO_ABILITY_NAMES.get(hero_name, '?')
            print(f" {ability} ({hero_name}) "
                  f"→ tap ({x}, {y}) conf={conf:.2f}")

        tap_fn(x, y)
        time.sleep(0.3)

        self._activated[hero_name] = True
        # Remove from icon_positions to avoid tapping again
        del self._icon_positions[hero_name]

        return True

    def get_status_vector(self):
        """
        Status vector (NUM_HEROES,) for PPO observation.

        0.0  = not deployed
        0.25 = deployed, icon not yet searched
        0.5  = deployed, icon not found by scan
        0.75 = deployed, ability detected and available
        1.0  = ability activated
        """
        status = np.zeros(NUM_HEROES, dtype=np.float32)
        now = time.time()

        for i, name in enumerate(HERO_NAMES):
            if not self._deployed.get(name, False):
                status[i] = 0.0
            elif self._activated.get(name, False):
                status[i] = 1.0
            elif name in self._icon_positions:
                status[i] = 0.75
            else:
                deploy_t = self._deploy_time.get(name, now)
                if now - deploy_t < DEPLOY_TO_SCAN_DELAY:
                    status[i] = 0.25
                else:
                    status[i] = 0.5

        return status

    def num_deployed(self):
        return sum(1 for v in self._deployed.values() if v)

    def is_deployed(self, hero_name):
        return self._deployed.get(hero_name, False)

    def num_activated(self):
        return sum(1 for v in self._activated.values() if v)

    def has_templates(self):
        return len(self._templates) > 0

    # -----------------------------------------------------------------
    # V4: YOLO positions of heroes on the battlefield
    # -----------------------------------------------------------------
    def update_battlefield_positions(self, hero_positions_named: dict):
        """
        Updates hero positions on the battlefield via YOLO.

        Args:
            hero_positions_named: dict {hero_name: (x, y)} from CombatObserver
        """
        self._battlefield_positions = hero_positions_named
        # Automatically mark as deployed if YOLO detects the hero
        for name in hero_positions_named:
            if name in HERO_NAMES and not self._deployed.get(name, False):
                self.mark_deployed(name)
                if self.verbose:
                    print(f" {name} detected by YOLO → marked as deployed")

    def get_hero_position(self, hero_name):
        """Returns the YOLO position (x, y) of a hero, or None."""
        return getattr(self, '_battlefield_positions', {}).get(hero_name)

    def heroes_near_center(self, village_center, radius=250):
        """Returns heroes close to the village center (good time to activate ability)."""
        import math
        result = []
        positions = getattr(self, '_battlefield_positions', {})
        for name, (hx, hy) in positions.items():
            if name not in HERO_NAMES:
                continue
            dist = math.sqrt((hx - village_center[0])**2 + (hy - village_center[1])**2)
            if dist < radius:
                result.append((name, dist))
        return result


# =============================================================================
# TOOLS: TEMPLATE EXTRACTION
# =============================================================================

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