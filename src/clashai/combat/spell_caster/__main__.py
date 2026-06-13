# clashai/combat/spell_caster/__main__.py
# Test CLI: `uv run python -m clashai.combat.spell_caster [image.png]`

import sys

import cv2
import numpy as np
from PIL import Image

from clashai.combat.spell_caster.constants import ADB_WIDTH, ADB_HEIGHT
from clashai.combat.spell_caster.caster import SpellCaster


def test_spell_caster(image_path=None):
    """Test SpellCaster V2 on a screenshot."""
    print("Test SpellCaster V2\n")

    if image_path:
        img_pil = Image.open(image_path).convert("RGB")
    else:
        # Phase B.1: route through the canonical adb_screenshot (WGC → ADB).
        from clashai.navigation.game_loop import adb_screenshot
        img_pil = adb_screenshot()

    caster = SpellCaster(verbose=True)

    # Simulate YOLO-detected defenses
    fake_buildings = [
        {'class': 'tour_enfer_mono', 'confidence': 0.98,
         'bbox': (800, 300, 850, 350), 'center': (825, 325)},
        {'class': 'tour_enfer_multiple', 'confidence': 0.95,
         'bbox': (1000, 300, 1050, 350), 'center': (1025, 325)},
        {'class': 'aigle_artilleur', 'confidence': 0.97,
         'bbox': (900, 200, 950, 250), 'center': (925, 225)},
        {'class': 'canon', 'confidence': 0.99,
         'bbox': (600, 400, 650, 450), 'center': (625, 425)},
    ]
    caster.set_defense_positions(fake_buildings)

    targets = caster.analyze_battlefield(img_pil, village_center_adb=(960, 500))

    print("\nV2 results:")
    print(f" Troops detected: {targets['num_troops']} "
          f"(including {targets['num_hurt']} injured)")
    print(f" Clusters: {targets['num_clusters']}")
    print(f" Main cluster: {targets['troop_cluster']}")
    print(f" Heal -> {targets['heal']}")
    print(f" Rage -> {targets['rage']}")
    print(f" Freeze -> {targets['freeze']} "
          f"({targets['freeze_target_name'] or 'fallback'})")

    # Debug image
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    sx, sy = w / ADB_WIDTH, h / ADB_HEIGHT

    for label, key, color in [('HEAL', 'heal', (0, 255, 0)),
                              ('RAGE', 'rage', (0, 128, 255)),
                              ('FREEZE', 'freeze', (255, 200, 0))]:
        ax, ay = targets[key]
        ix, iy = int(ax * sx), int(ay * sy)
        cv2.circle(img_cv, (ix, iy), 25, color, 3)
        cv2.putText(img_cv, label, (ix + 30, iy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw freeze targets
    for dx, dy, name, prio in caster._defense_targets:
        ix, iy = int(dx * sx), int(dy * sy)
        cv2.circle(img_cv, (ix, iy), 15, (0, 0, 255), 2)
        cv2.putText(img_cv, f"{name[:10]}", (ix + 20, iy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite('debug_spells_v2.png', img_cv)
    print("\nDebug image saved: debug_spells_v2.png")


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else None
    test_spell_caster(img)


if __name__ == "__main__":
    main()
