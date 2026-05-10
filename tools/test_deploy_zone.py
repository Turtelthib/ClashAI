# tools/test_deploy_zone.py
# Quick test for the new wall-segmentation-based deploy zone.
# Takes a live ADB screenshot, runs yolo_walls_seg, draws the result.
#
# Usage (on an enemy village screen, before attacking):
#   uv run python tools/test_deploy_zone.py
#   uv run python tools/test_deploy_zone.py --image path/to/screenshot.png

import os
import sys
import argparse
import subprocess
import io

import cv2
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from clashai.paths import WEIGHTS_DIR, ADB_DEVICE
from clashai.combat.action_space import NUM_POSITIONS
from clashai.perception.deploy_zone import (
    get_perimeter_from_walls,
    get_perimeter_from_buildings,
)

WALLS_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'yolo_walls_seg', 'walls_detection.pt')
ADB_W, ADB_H = 1920, 1080


def grab_screenshot(image_path=None):
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    print(f"Taking ADB screenshot from {ADB_DEVICE}...")
    r = subprocess.run(
        ['adb', '-s', ADB_DEVICE, 'exec-out', 'screencap', '-p'],
        capture_output=True, timeout=8
    )
    if r.returncode != 0 or len(r.stdout) < 100:
        print("ERROR: ADB screenshot failed")
        sys.exit(1)
    return Image.open(io.BytesIO(r.stdout)).convert('RGB')


def draw_positions(img_cv, positions, color, label_prefix):
    for i, (x, y) in enumerate(positions):
        sx = int(x * img_cv.shape[1] / ADB_W)
        sy = int(y * img_cv.shape[0] / ADB_H)
        cv2.circle(img_cv, (sx, sy), 8, color, -1)
        cv2.putText(img_cv, str(i), (sx + 10, sy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    # Load models
    if not os.path.exists(WALLS_MODEL_PATH):
        print(f"ERROR: walls model not found: {WALLS_MODEL_PATH}")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"Loading walls model: {WALLS_MODEL_PATH}")
    yolo_walls = YOLO(WALLS_MODEL_PATH)

    # Load YOLO buildings for fallback comparison
    yolo_buildings_path = os.path.join(WEIGHTS_DIR, 'best.pt')
    yolo_buildings = YOLO(yolo_buildings_path) if os.path.exists(yolo_buildings_path) else None

    # Screenshot
    screenshot = grab_screenshot(args.image)
    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print(f"Screenshot: {screenshot.width}×{screenshot.height}")

    #  Detect buildings first (used by both tests) 
    buildings = []
    if yolo_buildings is not None:
        res_b = yolo_buildings.predict(np.array(screenshot), conf=0.25, verbose=False)
        for box in res_b[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            buildings.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1+x2)/2, (y1+y2)/2),
                'class': 'building', 'confidence': float(box.conf[0])
            })
        print(f"\n[0] Buildings detected: {len(buildings)}")

    #  Test 1: walls + buildings combined 
    print("\n[1] get_perimeter_from_walls() with buildings...")
    wall_positions, wall_center, wall_ok = get_perimeter_from_walls(
        screenshot, yolo_walls,
        buildings=buildings if buildings else None,
        num_points=NUM_POSITIONS
    )

    if wall_ok:
        print(f"    OK — {len(wall_positions)} positions, center={wall_center}")
        draw_positions(img_cv, wall_positions, (0, 255, 0), 'W')
        # Draw center
        sx = int(wall_center[0] * img_cv.shape[1] / ADB_W)
        sy = int(wall_center[1] * img_cv.shape[0] / ADB_H)
        cv2.drawMarker(img_cv, (sx, sy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)
        # Draw wall masks overlay
        results = yolo_walls.predict(np.array(screenshot), conf=args.conf, verbose=False)
        r = results[0]
        if r.masks is not None:
            overlay = img_cv.copy()
            for mask_t in r.masks.data:
                mask = mask_t.cpu().numpy()
                mask = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                overlay[mask > 0.5] = (0, 180, 255)
            img_cv = cv2.addWeighted(img_cv, 0.7, overlay, 0.3, 0)
            print(f"    {len(r.masks)} wall segments detected")
    else:
        print("    FAILED — no valid positions from walls")

    #  Test 2: building hull fallback (comparison) 
    if buildings:
        print("\n[2] get_perimeter_from_buildings() (fallback comparison)...")
        b_positions, b_center, b_ok, _ = get_perimeter_from_buildings(
            buildings, num_points=NUM_POSITIONS, return_debug=True
        )
        if b_ok:
            print(f"    OK — {len(b_positions)} positions, center={b_center}")
            draw_positions(img_cv, b_positions, (0, 100, 255), 'B')
        else:
            print("    FAILED")

    #  Save result 
    out_path = os.path.join(project_root, '_test_deploy_zone.png')
    cv2.imwrite(out_path, img_cv)
    print(f"\nResult saved: {out_path}")
    print("GREEN circles = walls model positions")
    print("ORANGE circles = building hull positions (fallback)")


if __name__ == '__main__':
    main()
