# clashai/combat/combat_observer/__main__.py
# Smoke test: `uv run python -m clashai.combat.combat_observer`

import time

import cv2
import numpy as np
from PIL import Image

from clashai.combat.combat_observer.observer import CombatObserver


def main():
    print("Test CombatObserver\n")

    observer = CombatObserver()
    observer.start_combat()

    # Test with synthetic image
    time.sleep(0.1)

    # Create a test image (black with a few green bars)
    test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Simulate green health bars
    for y in [300, 320, 350, 400, 410]:
        cv2.rectangle(test_img, (800, y), (840, y+4), (0, 200, 0), -1)
    # Simulate red bars
    cv2.rectangle(test_img, (900, 380), (935, 384), (0, 0, 200), -1)

    pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

    features, raw = observer.observe(
        pil_img,
        village_center_adb=(960, 500),
        spells_remaining={'soin': 2, 'rage': 1, 'gel': 1},
        phase='combat'
    )

    print(f"\nFeatures ({len(features)} dims):")
    labels = [
        'phase', 'progress', 'troops_alive', 'troops_hurt',
        'heroes_alive', 'cluster_x', 'cluster_y', 'num_clusters',
        'cluster_spread', 'near_center', 'hurt_ratio',
        'spell_heal', 'spell_rage', 'spell_freeze', 'pad'
    ]
    for i, (label, val) in enumerate(zip(labels, features)):
        print(f" [{i:2d}] {label:18s} = {val:.3f}")

    print(f"\nRaw: {raw['num_troops']} troops, {raw['num_heroes']} heroes")
    print("Test done!")


if __name__ == "__main__":
    main()
