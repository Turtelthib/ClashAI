# clashai/perception/deploy/debug.py
# Debug visualisations + the legacy get_smart_deploy_positions facade.

import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

from clashai.config import ADB_WIDTH, ADB_HEIGHT
from clashai.perception.deploy.constants import DEPLOY_OFFSET, DIRECTION_LABELS
from clashai.perception.deploy.boundary import detect_village_boundary
from clashai.perception.deploy.positions import (
    compute_deploy_positions, get_village_center_adb, _fallback_positions,
)


DEBUG_DEPLOY_SAVE = True


def save_deploy_debug_image(screenshot_pil, buildings, positions, center,
                             output_dir='logs/deploy_zone',
                             episode=None, extra_info=None,
                             rejected_rays=None):
    """
    Saves an annotated image of the deployment zone for debugging.

    Useful to diagnose episodes where the agent taps incorrectly:
      - Position on a building → bbox poorly detected or MIN_BUILDING_DIST too low
      - Position in water → MAX_RADIAL_PUSH too large
      - Too few positions → MAX_RADIAL_PUSH too small, or off-center village

    Image contents:
      - YOLO bboxes in green
      - Deployment positions in red (numbered 0 to N-1)
      - Village center in blue
      - Text overlay: episode, building count, position count

    Args:
        screenshot_pil: PIL Image of the village (ADB coords 1920×1080)
        buildings: list of dicts with 'bbox' (x1,y1,x2,y2)
        positions: list of (x, y) ADB
        center: (cx, cy) ADB
        output_dir: output folder, created if it does not exist
        episode: episode number (optional, used in filename)
        extra_info: optional str to display in overlay

    Returns:
        path: path of the saved file, or None if disabled
    """
    if not DEBUG_DEPLOY_SAVE:
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)

        img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

        # YOLO bboxes in green
        for b in buildings:
            x1, y1, x2, y2 = b['bbox']
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if rejected_rays:
            for angle_deg, reason, last_pos in rejected_rays:
                # Dashed line from center to last tested point
                cv2.line(img_cv, center, last_pos, (128, 128, 128), 1, cv2.LINE_AA)
                # Small grey X at the position
                px, py = last_pos
                cv2.line(img_cv, (px-6, py-6), (px+6, py+6), (100, 100, 100), 2)
                cv2.line(img_cv, (px-6, py+6), (px+6, py-6), (100, 100, 100), 2)
                # Label with angle
                cv2.putText(img_cv, f'{angle_deg}° {reason[:3]}', (px+8, py+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Village center in blue
        cv2.circle(img_cv, center, 12, (255, 0, 0), -1)
        cv2.circle(img_cv, center, 14, (255, 255, 255), 2)

        # Positions in red, numbered
        for i, (x, y) in enumerate(positions):
            cv2.circle(img_cv, (x, y), 16, (0, 0, 255), -1)
            cv2.circle(img_cv, (x, y), 16, (255, 255, 255), 2)
            # Number in white at the center
            txt = str(i)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(img_cv, txt, (x - tw // 2, y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Overlay info at top (below player UI zone)
        info_lines = []
        if episode is not None:
            info_lines.append(f'Episode {episode}')
        info_lines.append(f'Buildings: {len(buildings)}')
        info_lines.append(f'Positions: {len(positions)}')
        info_lines.append(f'Center: {center}')
        if extra_info:
            info_lines.append(extra_info)

        y_off = 260
        for line in info_lines:
            # Black outline then white text for readability
            cv2.putText(img_cv, line, (12, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
            cv2.putText(img_cv, line, (12, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_off += 32

        # Name: ep_XXX_YYYYMMDD_HHMMSS.png (sorts chronologically in explorer)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if episode is not None:
            filename = f'ep{episode:04d}_{ts}.png'
        else:
            filename = f'{ts}.png'
        path = os.path.join(output_dir, filename)

        cv2.imwrite(path, img_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    except Exception as e:
        print(f" WARNING: Log deploy zone échoué : {e}")
        return None

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_smart_deploy_positions(screenshot_pil, direction_idx, spread=0.5,
                               num_points=12, offset_px=None):
    """
    Main entry point: detects the deployment zone and returns
    optimal positions.

    Args:
        screenshot_pil: PIL image of the screenshot (attack phase)
        direction_idx: 0-7 (N, NE, E, SE, S, SO, O, NO)
        spread: 0.0 (grouped) to 1.0 (spread out)
        num_points: number of positions
        offset_px: distance from the border (default: DEPLOY_OFFSET)

    Returns:
        positions: list of (x, y) in ADB coordinates (1920×1080)
        center_adb: (x, y) village center in ADB coordinates
        success: True if detection succeeded
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    hull, center = detect_village_boundary(img_cv)

    if hull is None or center is None:
        print(" ERROR: Détection de la zone de déploiement échouée")
        return _fallback_positions(direction_idx, spread, num_points), \
               (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50), False

    hull_area = cv2.contourArea(hull)
    n_hull_pts = len(hull.reshape(-1, 2))
    game_area = img_cv.shape[0] * img_cv.shape[1] * 0.55
    zoom_ratio = hull_area / game_area
    zoom_label = "dézoomé" if zoom_ratio < 0.40 else "moyen" if zoom_ratio < 0.55 else "zoomé"
    print(f" Zone détectée : hull={n_hull_pts} pts, "
          f"zoom={zoom_ratio:.0%} ({zoom_label})")

    positions = compute_deploy_positions(
        hull, center, img_cv.shape,
        direction_idx, spread, num_points, offset_px
    )

    center_adb = get_village_center_adb(center, img_cv.shape)

    if len(positions) < 3:
        print(f" WARNING: Seulement {len(positions)} positions, fallback")
        return _fallback_positions(direction_idx, spread, num_points), \
               center_adb, False

    direction_label = DIRECTION_LABELS[direction_idx]
    print(f" {len(positions)} positions de déploiement ({direction_label}, "
          f"spread={spread:.1f})")

    return positions, center_adb, True



def debug_deploy_zone(screenshot_pil, direction_idx=0, spread=0.5,
                      save_path=None):
    """
    Generates a debug image showing the detection and positions.

    Args:
        screenshot_pil: PIL image
        direction_idx: direction to visualize
        spread: spread
        save_path: save path (optional)

    Returns:
        debug_img: BGR image with annotations
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
    debug = img_cv.copy()

    hull, center = detect_village_boundary(img_cv)

    if hull is None:
        cv2.putText(debug, "DETECTION ECHOUEE", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if save_path:
            cv2.imwrite(save_path, debug)
        return debug

    # Draw the hull
    cv2.drawContours(debug, [hull], -1, (0, 255, 0), 3)

    # Draw the center
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(debug, (cx, cy), 10, (0, 255, 255), -1)

    # Compute and draw positions
    positions = compute_deploy_positions(
        hull, center, img_cv.shape,
        direction_idx, spread, num_points=16
    )

    # Convert ADB positions to image coordinates for drawing
    img_h, img_w = img_cv.shape[:2]
    for adb_x, adb_y in positions:
        ix = int(adb_x * img_w / ADB_WIDTH)
        iy = int(adb_y * img_h / ADB_HEIGHT)
        cv2.circle(debug, (ix, iy), 8, (255, 0, 255), -1)

    # Info text
    direction_label = DIRECTION_LABELS[direction_idx]
    cv2.putText(debug, f"Dir: {direction_label} Spread: {spread:.1f} "
                f"Pts: {len(positions)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(save_path, debug)

    return debug


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test on a specific image
        img_path = sys.argv[1]
        img_pil = Image.open(img_path).convert("RGB")

        print(f"Test deploy_zone sur {img_path}")
        print(f" Image: {img_pil.size}")

        for direction in range(8):
            positions, center_adb, success = get_smart_deploy_positions(
                img_pil, direction, spread=0.5, num_points=12
            )
            label = DIRECTION_LABELS[direction]
            status = "" if success else "ERROR:"
            print(f" {status} {label}: {len(positions)} positions, "
                  f"center=({center_adb[0]},{center_adb[1]})")

        # Generate debug images
        for d in range(8):
            out = f"debug_deploy_{DIRECTION_LABELS[d]}.png"
            debug_deploy_zone(img_pil, d, 0.5, save_path=out)
            print(f" {out}")

    else:
        print("deploy_zone.py — Détection de la zone de déploiement")
        print()
        print("Usage :")
        print(" python deploy_zone.py <screenshot.png>")
        print()
        print("Dans le code :")
        print(" from clashai.perception.deploy_zone import get_smart_deploy_positions")
        print(" positions, center, ok = get_smart_deploy_positions(img, dir, spread)")
