# tools/run_test.py
# Diagnostic test — runs 1 attack episode and saves annotated captures
# at key moments to logs/test_run/ so you can visually verify every CNN.
#
# Usage:
#   uv run python tools/run_test.py

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

OUT_DIR = os.path.join(project_root, 'logs', 'test_run')
os.makedirs(OUT_DIR, exist_ok=True)

import cv2
import numpy as np
from PIL import Image, ImageDraw

from clashai.navigation.game_loop import load_models, adb_screenshot, classify_screen, analyze_village
from clashai.paths import ADB_DEVICE


# ── helpers ──────────────────────────────────────────────────────────────────

def save_annotated(img_pil, filename, models, extra_text=None):
    """Save a screenshot with all CNN results overlaid."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    # Screen state
    state, conf = classify_screen(img_pil, models)
    cv2.putText(img_cv, f"Screen: {state} ({conf:.0%})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if conf > 0.7 else (0, 165, 255), 2)

    # YOLO buildings
    buildings = analyze_village(img_pil, models)
    for b in buildings:
        x1, y1, x2, y2 = b['bbox']
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(img_cv, b['class'][:12], (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1)

    # YOLO walls
    yolo_walls = models.get('yolo_walls')
    if yolo_walls is not None:
        results = yolo_walls.predict(np.array(img_pil), conf=0.25, verbose=False)
        r = results[0]
        if r.masks is not None:
            overlay = img_cv.copy()
            for mask_t in r.masks.data:
                mask = mask_t.cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay[mask > 0.5] = (0, 180, 255)
            img_cv = cv2.addWeighted(img_cv, 0.75, overlay, 0.25, 0)

    # Troop bar detector
    bar_det = models.get('troop_bar_detector')
    if bar_det is not None:
        detections = bar_det.detect(img_pil)
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            color = (0, 0, 200) if d['is_grayed'] else (0, 200, 0)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            label = f"{d['name']} x{d['count']}"
            cv2.putText(img_cv, label, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Stats overlay
    lines = [
        f"Buildings: {len(buildings)}",
        f"ADB: {ADB_DEVICE}",
    ]
    if extra_text:
        lines.append(extra_text)
    for i, line in enumerate(lines):
        y = h - 15 - i * 20
        cv2.putText(img_cv, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(img_cv, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(path, img_cv)
    print(f"  Saved: {path}  ({state} {conf:.0%}, {len(buildings)} buildings)")
    return state, conf, buildings


def wait_for_state(target, models, timeout=60):
    """Wait until screen reaches target state, return screenshot."""
    print(f"  Waiting for '{target}'...")
    for _ in range(timeout):
        img = adb_screenshot()
        if img:
            state, conf = classify_screen(img, models)
            print(f"    → {state} ({conf:.0%})")
            if state == target and conf > 0.6:
                return img
        time.sleep(1)
    return adb_screenshot()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  ClashAI V4.3 — Diagnostic Test Run")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    print("Loading models...")
    models = load_models()
    print()

    # 1. village_home
    print("[1/5] village_home — screenshot on home screen")
    img = wait_for_state('village_home', models)
    if img:
        save_annotated(img, '1_village_home.jpg', models)

    # 2. Navigate to attack — prep_attaque
    print("\n[2/5] prep_attaque — navigating to attack screen...")
    from clashai.navigation import game_loop as gl
    gl.adb_tap(*gl.BUTTONS.get('attaquer', (960, 900)))
    time.sleep(2)
    img = wait_for_state('prep_attaque', models, timeout=30)
    if img:
        save_annotated(img, '2_prep_attaque.jpg', models, 'screen=prep (counter top-left)')

    # 3. Start attack — debut_attaque
    print("\n[3/5] debut_attaque — starting attack...")
    gl.adb_tap(*gl.BUTTONS.get('trouver_partie', (960, 600)))
    time.sleep(3)
    img = wait_for_state('phase_attaque', models, timeout=60)
    if img:
        save_annotated(img, '3_debut_attaque.jpg', models, 't=0s')

    # 4. After 30s
    print("\n[4/5] attaque_30s — waiting 30 seconds...")
    time.sleep(30)
    img = adb_screenshot()
    if img:
        save_annotated(img, '4_attaque_30s.jpg', models, 't=30s')

    # 5. After 60s
    print("\n[5/5] attaque_60s — waiting 30 more seconds...")
    time.sleep(30)
    img = adb_screenshot()
    if img:
        save_annotated(img, '5_attaque_60s.jpg', models, 't=60s')

    print(f"\n{'='*60}")
    print(f"  Done! Check: {OUT_DIR}")
    print(f"  5 annotated images show what each CNN sees at each moment.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
