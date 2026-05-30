# scripts/rl/deploy_zone.py
# Dynamic detection of the deployment zone on an enemy village.
#
# In Clash of Clans, the deployment zone is delimited by a semi-transparent
# red line around the village. Troops can only be placed OUTSIDE this line
# (on the outer green grass).
#
# Method :
# 1. The red overlay shifts the grass hue (HSV) from H≈33 to H≈20
# 2. We detect this "warm" grass (H=14-28) = inner village zone
# 3. We compute the convex hull = approximate boundary
# 4. Deployment positions are placed JUST OUTSIDE the hull
#
# Usage :
# from clashai.perception.deploy_zone import get_smart_deploy_positions
# positions = get_smart_deploy_positions(screenshot_pil, direction_idx, spread)

import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# ADB resolution — re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# UI exclusion zones (in ADB coordinates 1920×1080)
# Taps in these zones trigger buttons instead of deploying troops
# YOLO walls segmentation trained at imgsz=640 (see
# tools/train_yolo_walls_seg.py DEFAULT_IMG_SIZE). Set explicitly so a
# future retrain at 1280/1600 only requires bumping this constant.
YOLO_WALLS_IMGSZ = 640

UI_EXCLUSION_ZONES = [
    # Top: player info + resources
    (0, 0, 280, 230),
    (1450, 0, 1920, 160),
    # Bottom: troop bar only (real UI, not the village)
    (0, 735, 1920, 1080),
    # Side buttons (smaller than the old large rectangle)
    (0, 590, 210, 730),
    (1220, 560, 1510, 730),
]

# Minimum margin from screen edges
SCREEN_MARGIN = 60

# Distance (in ADB pixels) between the hull and the deployment positions
DEPLOY_OFFSET = 35

# V4.2 — Parameters for get_perimeter_from_buildings (YOLO-only, no HSV)
# Artificial expansion of each bbox to simulate the CoC collision zone
# (~1.5 tile at medium zoom). The hull of expanded bboxes covers the real red zone.
BUILDING_PADDING = 40

# Minimum distance between a final position and any building center.
# Must be > half max building size + margin. Most buildings are
# 60-80px wide → 70px guarantees we never tap on a sprite.
MIN_BUILDING_DIST = 40

# Final offset from the hull, adaptive to zoom level.
# Smaller than DEPLOY_OFFSET because padding already does most of the work.
OFFSET_BY_ZOOM = {'dezoome': 30, 'moyen': 20, 'zoome': 10}

# Radial push cap AFTER exiting the hull to avoid landing in
# water/rocks when an off-center building forces a longer push.
# 40px ≈ 1 CoC tile : if no valid spot at ≤ 40px from hull → discard ray.
MAX_RADIAL_PUSH = 40

# Directions (index → label)
DIRECTION_LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']

# Corresponding angles (in radians, 0 = right, counter-clockwise)
# N=up, E=right, S=down, O=left
DIRECTION_ANGLES = {
    0: np.pi / 2,
    1: np.pi / 4,
    2: 0,
    3: -np.pi / 4,
    4: -np.pi / 2,
    5: -3 * np.pi / 4,
    6: np.pi,
    7: 3 * np.pi / 4,
}

# V4.2 — Local HSV validation as a complement to YOLO:
# when a building is not detected by YOLO (worker huts,
# isolated walls…), we check the color of the candidate pixel.
# The CoC red overlay shifts the grass Hue (~33 green) toward ~15 (orange).
HSV_CHECK_RADIUS = 4
HSV_RED_H_MAX = 28
HSV_RED_SAT_MIN = 50
HSV_RED_RATIO_THRESHOLD = 0.5

# =============================================================================
# ZONE DETECTION
# =============================================================================

def _is_in_red_overlay(img_bgr, x, y, radius=HSV_CHECK_RADIUS):
    """
    True if the small patch around (x, y) is within the CoC red overlay.
    Fails silently (False) on dark villages → no regression.
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x) - radius)
    y1 = max(0, int(y) - radius)
    x2 = min(w, int(x) + radius + 1)
    y2 = min(h, int(y) + radius + 1)

    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red_mask = (hsv[:, :, 0] < HSV_RED_H_MAX) & (hsv[:, :, 1] > HSV_RED_SAT_MIN)
    red_ratio = float(red_mask.sum()) / float(red_mask.size)
    return red_ratio >= HSV_RED_RATIO_THRESHOLD

def detect_village_boundary(img_cv):
    """
    Detects the village boundary from a BGR screenshot.

    CoC's red overlay shifts the HSV hue of the grass: H≈33 (green)
    becomes H≈20 (warm yellow-green). We detect this warm zone to
    find the interior of the village.

    Args:
        img_cv: BGR image (numpy array) of the screenshot

    Returns:
        hull: convex hull of the village (numpy array Nx1x2) or None
        center: village center (x, y) in image pixels or None
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    # --- UI Mask ---
    # Exclude UI zones (in image coordinates)
    # Compute the image→ADB ratio for conversion
    scale_x = w / ADB_WIDTH
    scale_y = h / ADB_HEIGHT

    ui_mask = np.ones((h, w), dtype=np.uint8) * 255
    for ax1, ay1, ax2, ay2 in UI_EXCLUSION_ZONES:
        ix1 = int(ax1 * scale_x)
        iy1 = int(ay1 * scale_y)
        ix2 = int(ax2 * scale_x)
        iy2 = int(ay2 * scale_y)
        ui_mask[iy1:iy2, ix1:ix2] = 0

    # Also exclude extreme edges
    margin = int(SCREEN_MARGIN * min(scale_x, scale_y))
    ui_mask[:margin, :] = 0
    ui_mask[h - margin:, :] = 0
    ui_mask[:, :margin] = 0
    ui_mask[:, w - margin:] = 0

    # --- Warm grass detection (village red zone) ---
    # H=14-28 : grass tinted by the red overlay
    # S>80 : saturated (not grey)
    # V>100 : bright (not forest shadow)
    mask_warm = cv2.inRange(hsv, (14, 80, 100), (28, 255, 255))
    mask_warm = cv2.bitwise_and(mask_warm, ui_mask)

    # --- Morphological cleanup ---
    # Close : fill holes (buildings, walls, decorations)
    kernel_close = np.ones((30, 30), np.uint8)
    # Open : remove noise (small warm pixels in the forest)
    kernel_open = np.ones((15, 15), np.uint8)

    min_contour_area = h * w * 0.05

    # --- Cascade strategy: try from most precise to broadest ---
    strategies = [
        ("warm", mask_warm),
    ]

    # Prepare fallbacks
    # Fallback 1 : light grass (works when the red overlay is faint)
    mask_grass = cv2.inRange(hsv, (18, 70, 130), (38, 255, 255))
    mask_grass = cv2.bitwise_and(mask_grass, ui_mask)
    strategies.append(("herbe claire", mask_grass))

    # Fallback 2 : wide grass (very permissive)
    mask_wide = cv2.inRange(hsv, (15, 60, 110), (40, 255, 255))
    mask_wide = cv2.bitwise_and(mask_wide, ui_mask)
    strategies.append(("herbe large", mask_wide))

    # Fallback 3 : DARK VILLAGES (underwater theme, night, etc.)
    # The ground is blue-green (H=75-120) instead of green (H=33).
    # The village interior is brighter (V>90) than the dark exterior (V<70).
    # We use brightness + saturation to isolate the game zone.
    mask_bright = cv2.inRange(hsv, (0, 20, 90), (180, 255, 255))
    # Exclude UI zones and pure white (text, clouds)
    mask_not_white = cv2.inRange(hsv, (0, 15, 0), (180, 255, 240))
    mask_dark_village = cv2.bitwise_and(mask_bright, mask_not_white)
    mask_dark_village = cv2.bitwise_and(mask_dark_village, ui_mask)
    strategies.append(("village sombre (luminosité)", mask_dark_village))

    # Fallback 4 : Direct red border (some dark villages have a visible red line)
    # H<10 or H>170 (red), S>80, V>60
    mask_red_low = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_red_high = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask_red_border = cv2.bitwise_or(mask_red_low, mask_red_high)
    mask_red_border = cv2.bitwise_and(mask_red_border, ui_mask)
    # Dilate the border to fill the interior
    kernel_dilate = np.ones((50, 50), np.uint8)
    mask_red_filled = cv2.dilate(mask_red_border, kernel_dilate, iterations=3)
    mask_red_filled = cv2.morphologyEx(mask_red_filled, cv2.MORPH_CLOSE,
                                        np.ones((60, 60), np.uint8))
    strategies.append(("bordure rouge", mask_red_filled))

    hull = None
    center = None

    for strategy_name, mask_raw in strategies:
        mask_filled = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close)
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_contour_area:
            continue

        # Success — use this contour
        if strategy_name != "warm":
            from clashai.config.logging import pp
            pp(f" WARNING: Overlay rouge faible, fallback {strategy_name}", tag='warning')

        hull = cv2.convexHull(largest)

        M = cv2.moments(hull)
        if M["m00"] > 0:
            center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        else:
            center = np.mean(hull.reshape(-1, 2).astype(float), axis=0)

        break

    if hull is None:
        return None, None

    return hull, center


# =============================================================================
# DEPLOYMENT POSITION COMPUTATION
# =============================================================================

def _sample_hull_point(hull_pts, frac):
    """Interpolates a point along the hull perimeter at the given fraction."""
    n = len(hull_pts)
    idx_float = frac * n
    i1 = int(idx_float) % n
    i2 = (i1 + 1) % n
    t = idx_float - int(idx_float)
    return hull_pts[i1] * (1 - t) + hull_pts[i2] * t


def _angle_from_center(pt, center):
    """Computes the angle of a point relative to the center (0=right, counter-clockwise)."""
    dx = pt[0] - center[0]
    dy = -(pt[1] - center[1])
    return np.arctan2(dy, dx)


def _angle_diff(a, b):
    """Signed angular difference between a and b, normalized to [-π, π]."""
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def _is_in_exclusion_zone(x, y, img_h, img_w):
    """Checks whether a point (in ADB coordinates) is inside a UI zone."""
    for ax1, ay1, ax2, ay2 in UI_EXCLUSION_ZONES:
        if ax1 <= x <= ax2 and ay1 <= y <= ay2:
            return True
    return False


def compute_deploy_positions(hull, center, img_shape, direction_idx,
                             spread=0.5, num_points=12, offset_px=None):
    """
    Computes deployment positions along the edge of the village.

    Automatically adapts to zoom level:
    - Zoomed out : positions offset outside the hull
    - Zoomed in : positions on the hull edge or screen edge

    Args:
        hull: convex hull (Nx1x2 array in image coordinates)
        center: village center (x, y) in image coordinates
        img_shape: (height, width) of the source image
        direction_idx: 0-7 (N, NE, E, SE, S, SO, O, NO)
        spread: 0.0 (grouped at direction center) to 1.0 (spread over the full side)
        num_points: number of positions to generate
        offset_px: distance in ADB pixels from the hull (default: auto)

    Returns:
        positions: list of (x, y) in ADB coordinates
    """
    img_h, img_w = img_shape[:2]
    scale_x = ADB_WIDTH / img_w
    scale_y = ADB_HEIGHT / img_h

    hull_pts = hull.reshape(-1, 2).astype(float)
    target_angle = DIRECTION_ANGLES[direction_idx]

    # --- Zoom detection ---
    hull_area = cv2.contourArea(hull)
    game_area = img_h * img_w * 0.55
    zoom_ratio = hull_area / game_area

    # Adapt parameters to zoom level
    if offset_px is None:
        if zoom_ratio < 0.40:
            offset_px = DEPLOY_OFFSET
        elif zoom_ratio < 0.55:
            offset_px = 20
        else:
            offset_px = 8

    margin = SCREEN_MARGIN
    if zoom_ratio > 0.50:
        margin = 30

    dedup_dist_sq = 400 if zoom_ratio < 0.50 else 200

    # --- Sample many points along the hull ---
    n_samples = 200
    hull_samples = []
    for i in range(n_samples):
        frac = i / n_samples
        pt = _sample_hull_point(hull_pts, frac)
        angle = _angle_from_center(pt, center)
        hull_samples.append((pt, angle, frac))

    # --- Sort by angular proximity to target direction ---
    hull_samples.sort(key=lambda x: abs(_angle_diff(x[1], target_angle)))

    # --- Select points according to spread ---
    # When zoomed in, widen the spread to compensate for lost positions
    effective_spread = spread
    if zoom_ratio > 0.50:
        effective_spread = min(1.0, spread + 0.3)

    max_angle_range = np.pi * (0.15 + 0.85 * effective_spread)

    # Filter points within the angular arc
    candidates = []
    for pt, angle, frac in hull_samples:
        if abs(_angle_diff(angle, target_angle)) <= max_angle_range:
            candidates.append((pt, angle))

    if not candidates:
        candidates = [(pt, angle) for pt, angle, frac in hull_samples[:num_points * 3]]

    # --- Sort by angle for ordered deployment ---
    candidates.sort(key=lambda x: x[1])

    # --- Subsample to obtain num_points positions ---
    # Request more than needed since some will be filtered out
    target_count = int(num_points * 1.5)
    if len(candidates) > target_count:
        step = len(candidates) / target_count
        selected = [candidates[int(i * step)] for i in range(target_count)]
    else:
        selected = candidates

    # --- Convert to ADB coordinates with outward offset ---
    positions = []
    offset_img = offset_px / max(scale_x, scale_y)

    for pt, angle in selected:
        # Outward direction (from center)
        direction = pt - center
        norm = np.linalg.norm(direction)
        if norm < 1:
            continue
        direction = direction / norm

        # Point shifted outward
        deploy_pt = pt + direction * offset_img

        # Convert to ADB coordinates
        adb_x = int(deploy_pt[0] * scale_x)
        adb_y = int(deploy_pt[1] * scale_y)

        # Clamp to screen bounds
        adb_x = max(margin, min(ADB_WIDTH - margin, adb_x))
        adb_y = max(margin, min(ADB_HEIGHT - margin, adb_y))

        # Check UI exclusion zones
        if _is_in_exclusion_zone(adb_x, adb_y, ADB_HEIGHT, ADB_WIDTH):
            continue

        positions.append((adb_x, adb_y))

    # Deduplicate positions that are too close
    if positions:
        unique = [positions[0]]
        for px, py in positions[1:]:
            too_close = False
            for ux, uy in unique:
                if (px - ux) ** 2 + (py - uy) ** 2 < dedup_dist_sq:
                    too_close = True
                    break
            if not too_close:
                unique.append((px, py))
        positions = unique

    # --- Guarantee a minimum number of positions ---
    # If not enough positions (extreme zoom), add points
    # along the screen edge in the requested direction
    if len(positions) < 6:
        positions = _add_screen_edge_positions(
            positions, direction_idx, margin, num_points
        )

    return positions


def _add_screen_edge_positions(existing, direction_idx, margin, target_count):
    """
    Adds positions along the screen edge when the hull
    extends beyond the screen (strong zoom). These positions are valid because
    in CoC, the visible edge of the map is always deployable.
    """
    # Which screen edge is in the requested direction?
    edge_positions = {
        0: [(x, margin) for x in range(200, 1720, 120)],
        1: [(x, margin + (1920 - x) // 3)
            for x in range(800, 1860, 100)],
        2: [(1920 - margin, y) for y in range(100, 880, 80)],
        3: [(x, 880 - (1920 - x) // 3)
            for x in range(800, 1860, 100)],
        4: [(x, 880) for x in range(200, 1720, 120)],
        5: [(x, 880 - x // 3)
            for x in range(100, 1100, 100)],
        6: [(margin, y) for y in range(100, 880, 80)],
        7: [(x, margin + x // 3)
            for x in range(100, 1100, 100)],
    }

    edge_pts = edge_positions.get(direction_idx, [])

    # Filter UI points
    edge_pts = [(x, y) for x, y in edge_pts
                if not _is_in_exclusion_zone(x, y, ADB_HEIGHT, ADB_WIDTH)]

    # Merge with existing positions (avoid duplicates)
    combined = list(existing)
    for px, py in edge_pts:
        if len(combined) >= target_count:
            break
        too_close = False
        for ux, uy in combined:
            if (px - ux) ** 2 + (py - uy) ** 2 < 400:
                too_close = True
                break
        if not too_close:
            combined.append((px, py))

    return combined


def get_village_center_adb(center, img_shape):
    """
    Converts the village center to ADB coordinates.
    """
    img_h, img_w = img_shape[:2]
    adb_x = int(center[0] * ADB_WIDTH / img_w)
    adb_y = int(center[1] * ADB_HEIGHT / img_h)
    return (adb_x, adb_y)


def get_full_perimeter_positions(screenshot_pil, num_points=20, offset_px=None):
    """
    Generates deployment positions spread over the FULL perimeter
    of the village (360°), not just one side.

    This is the function to use for V2 where the agent freely chooses
    from all positions around the village.

    Args:
        screenshot_pil: PIL Image of the screenshot
        num_points: number of positions to generate (evenly distributed)
        offset_px: distance from the border (auto-adapted to zoom)

    Returns:
        positions: list of (x, y) in ADB coordinates, spread over 360°
        center_adb: (x, y) village center
        success: True if detection succeeded
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv.shape[:2]

    hull, center = detect_village_boundary(img_cv)

    if hull is None or center is None:
        return None, (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50), False

    scale_x = ADB_WIDTH / img_w
    scale_y = ADB_HEIGHT / img_h

    # Adapt offset to zoom level
    hull_area = cv2.contourArea(hull)
    game_area = img_h * img_w * 0.55
    zoom_ratio = hull_area / game_area

    if offset_px is None:
        if zoom_ratio < 0.40:
            offset_px = DEPLOY_OFFSET
        elif zoom_ratio < 0.55:
            offset_px = 20
        else:
            offset_px = 8

    margin = 30 if zoom_ratio > 0.50 else SCREEN_MARGIN

    zoom_label = "dézoomé" if zoom_ratio < 0.40 else \
                 "moyen" if zoom_ratio < 0.55 else "zoomé"
    print(f" Zone détectée : hull={len(hull)} pts, "
          f"zoom={zoom_ratio:.0%} ({zoom_label})")

    # Sample uniformly over the full hull perimeter
    hull_pts = hull.reshape(-1, 2).astype(float)
    offset_img = offset_px / max(scale_x, scale_y)

    positions = []
    for i in range(num_points * 3):
        frac = i / (num_points * 3)
        pt = _sample_hull_point(hull_pts, frac)

        # Outward direction
        direction = pt - center
        norm = np.linalg.norm(direction)
        if norm < 1:
            continue
        direction = direction / norm

        # Outward offset
        deploy_pt = pt + direction * offset_img

        # Convert to ADB coordinates
        adb_x = int(deploy_pt[0] * scale_x)
        adb_y = int(deploy_pt[1] * scale_y)

        # Clamp
        adb_x = max(margin, min(ADB_WIDTH - margin, adb_x))
        adb_y = max(margin, min(ADB_HEIGHT - margin, adb_y))

        # Exclude UI zones
        if _is_in_exclusion_zone(adb_x, adb_y, ADB_HEIGHT, ADB_WIDTH):
            continue

        positions.append((adb_x, adb_y))

    # Deduplicate
    dedup_dist = 200 if zoom_ratio < 0.50 else 100
    if positions:
        unique = [positions[0]]
        for px, py in positions[1:]:
            too_close = any(
                (px - ux) ** 2 + (py - uy) ** 2 < dedup_dist
                for ux, uy in unique
            )
            if not too_close:
                unique.append((px, py))
        positions = unique

    # Sort by angle (so pos 0 = North, pos 5 = East, etc.)
    def angle_from_center(p):
        dx = p[0] - ADB_WIDTH / 2
        dy = -(p[1] - ADB_HEIGHT / 2)
        return np.arctan2(dy, dx)

    positions.sort(key=angle_from_center, reverse=True)

    # Subsample to the requested count
    if len(positions) > num_points:
        step = len(positions) / num_points
        positions = [positions[int(i * step)] for i in range(num_points)]

    center_adb = get_village_center_adb(center, img_cv.shape)

    print(f" {len(positions)} positions (360° périmètre)")

    return positions, center_adb, True


# =============================================================================
# DEPLOYMENT ZONE FROM WALL SEGMENTATION (V4.3) — primary method
# =============================================================================

def get_perimeter_from_walls(screenshot_pil, yolo_walls_model,
                              buildings=None, num_points=20):
    """
    V4.3 — Deployment positions from wall segmentation + building bboxes.

    Combines BOTH models to define the forbidden zone:
      - Wall masks (yolo_walls_seg) → exact village boundary
      - Building bboxes (from buildings YOLO) → filled forbidden zones

    Raycasting is done entirely in IMAGE pixel space to avoid scaling
    artifacts, then converted to ADB coordinates at the end.

    Args:
        screenshot_pil: PIL Image
        yolo_walls_model: loaded YOLO segmentation model
        buildings: list of {'bbox': (x1,y1,x2,y2), 'center': (cx,cy), ...}
                   from the buildings YOLO (optional, improves accuracy)
        num_points: number of deploy positions to generate

    Returns:
        (positions_adb, center_adb, success)
    """
    import numpy as np
    import cv2

    fallback_center = (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50)

    if yolo_walls_model is None:
        return None, fallback_center, False

    img_arr = np.array(screenshot_pil)
    img_h, img_w = img_arr.shape[:2]
    scale_x = ADB_WIDTH / img_w
    scale_y = ADB_HEIGHT / img_h

    # UI exclusion zones in IMAGE pixel space
    def _ui_zone_img(x1, y1, x2, y2):
        return (int(x1 / scale_x), int(y1 / scale_y),
                int(x2 / scale_x), int(y2 / scale_y))

    ui_zones_img = [_ui_zone_img(*z) for z in UI_EXCLUSION_ZONES]
    margin_x = int(SCREEN_MARGIN / scale_x)
    margin_y = int(SCREEN_MARGIN / scale_y)

    #  1. Wall segmentation mask 
    try:
        results = yolo_walls_model.predict(
            img_arr, conf=0.25, imgsz=YOLO_WALLS_IMGSZ, verbose=False,
        )
    except Exception as e:
        print(f" WARNING: yolo_walls inference failed: {e}")
        return None, fallback_center, False

    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        print(" WARNING: yolo_walls — no walls detected, falling back")
        return None, fallback_center, False

    # Combine all wall masks
    forbidden = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask_tensor in r.masks.data:
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.shape != (img_h, img_w):
            mask_np = cv2.resize(mask_np, (img_w, img_h),
                                 interpolation=cv2.INTER_NEAREST)
        forbidden = np.maximum(forbidden, (mask_np > 0.5).astype(np.uint8))

    #  2. Add building bboxes to the forbidden zone 
    if buildings:
        for b in buildings:
            x1, y1, x2, y2 = b['bbox']
            ix1 = max(0, int(x1 / scale_x) - 4)
            iy1 = max(0, int(y1 / scale_y) - 4)
            ix2 = min(img_w, int(x2 / scale_x) + 4)
            iy2 = min(img_h, int(y2 / scale_y) + 4)
            forbidden[iy1:iy2, ix1:ix2] = 1

    #  3. Dilate to close gaps and add safety margin 
    # Close gaps between wall segments
    k_close = np.ones((11, 11), np.uint8)
    forbidden = cv2.morphologyEx(forbidden, cv2.MORPH_CLOSE, k_close)
    # Small dilation for safety margin
    k_expand = np.ones((7, 7), np.uint8)
    forbidden = cv2.dilate(forbidden, k_expand, iterations=2)

    #  4. Compute centroid from the forbidden zone 
    contours, _ = cv2.findContours(forbidden, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, fallback_center, False

    main = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main) < 3000:
        print(" WARNING: yolo_walls — forbidden zone too small, falling back")
        return None, fallback_center, False

    M = cv2.moments(main)
    if M['m00'] > 0:
        center_img = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
    else:
        center_img = np.array([img_w / 2, img_h / 2])

    center_adb = (int(center_img[0] * scale_x), int(center_img[1] * scale_y))

    #  5. Raycasting in IMAGE pixel space 
    # For each angle: march ALL the way outward from center, track the LAST
    # forbidden pixel encountered. Deploy position = just after that last pixel.
    # This handles multi-ring bases: the point is placed outside the outermost
    # wall, not the first inner ring.
    offset_px_img = int(DEPLOY_OFFSET / scale_x)  # offset in image pixels
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    positions_img = []

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        last_forbidden_r = 0  # track the LAST forbidden radius on this ray

        # Full sweep to find the outermost forbidden pixel
        for radius in range(0, int(img_w * 0.8), 2):
            px = int(center_img[0] + dx * radius)
            py = int(center_img[1] + dy * radius)

            if px < 0 or px >= img_w or py < 0 or py >= img_h:
                break

            if forbidden[py, px] > 0:
                last_forbidden_r = radius  # keep updating → last forbidden wins

        if last_forbidden_r == 0:
            continue  # ray never hit a forbidden zone (shouldn't happen)

        # Place the position just outside the outermost forbidden pixel
        deploy_r = last_forbidden_r + offset_px_img
        ex = int(center_img[0] + dx * deploy_r)
        ey = int(center_img[1] + dy * deploy_r)

        # Clamp to image bounds
        ex = max(margin_x, min(img_w - margin_x, ex))
        ey = max(margin_y, min(img_h - margin_y, ey))

        # Skip UI zones
        in_ui = any(
            x1 <= ex <= x2 and y1 <= ey <= y2
            for x1, y1, x2, y2 in ui_zones_img
        )
        if not in_ui:
            positions_img.append((ex, ey))

    if len(positions_img) < 4:
        print(f" WARNING: yolo_walls — only {len(positions_img)} positions, falling back")
        return None, fallback_center, False

    #  6. Convert image pixel → ADB coordinates 
    positions_adb = [
        (int(px * scale_x), int(py * scale_y))
        for px, py in positions_img
    ]

    # Deduplicate (min distance 20px in ADB space)
    unique = [positions_adb[0]]
    for px, py in positions_adb[1:]:
        if not any((px - ux)**2 + (py - uy)**2 < 400 for ux, uy in unique):
            unique.append((px, py))

    n_walls = len(r.masks) if r.masks is not None else 0
    n_buildings = len(buildings) if buildings else 0
    print(f" Deploy zone (walls+buildings): {n_walls} wall segments, "
          f"{n_buildings} buildings, {len(unique)}/{num_points} positions")

    return unique, center_adb, True


# =============================================================================
# DEPLOYMENT ZONE FROM YOLO BOUNDING BOXES (V4.2)
# =============================================================================

def get_perimeter_from_buildings(buildings, num_points=20, offset_px=None,
                                  return_debug=False, screenshot_pil=None):
    """
    V4.2 — Deployment positions via angular raycasting, 100% YOLO.

    Algorithm:
      1. Expand bboxes by BUILDING_PADDING (simulates CoC collision zone).
      2. Convex hull of expanded bboxes → village contour + red zone.
      3. Cast num_points rays at equal angles from the centroid.
      4. Phase A: advance until exiting the hull (abort if UI/screen edge).
      5. Phase B: place position with offset + bounded push (MAX_RADIAL_PUSH).
      6. Dedup + sort by angle + subsample.

    Args:
        buildings: list of {'bbox': (x1,y1,x2,y2), 'center': (cx,cy), ...}
        num_points: target number of positions (default 20)
        offset_px: override offset (default: adaptive to zoom)
        return_debug: if True, also returns a debug dict

    Returns:
        Without debug: (positions, center_adb, success)
        With debug: (positions, center_adb, success, debug_dict)
    """
    def _empty_result(reason='no_buildings'):
        fallback_center = (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50)
        if return_debug:
            return None, fallback_center, False, {
                'rejected_rays': [], 'reason': reason,
            }
        return None, fallback_center, False

    if not buildings or len(buildings) < 3:
        return _empty_result('no_buildings')

    img_bgr = None
    if screenshot_pil is not None:
        img_bgr = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    # --- 1. Expanded bboxes + centers ---
    pts = []
    bboxes_list = []
    for b in buildings:
        x1, y1, x2, y2 = b['bbox']
        pts.extend([
            (x1 - BUILDING_PADDING, y1 - BUILDING_PADDING),
            (x2 + BUILDING_PADDING, y1 - BUILDING_PADDING),
            (x1 - BUILDING_PADDING, y2 + BUILDING_PADDING),
            (x2 + BUILDING_PADDING, y2 + BUILDING_PADDING),
        ])
        bboxes_list.append((x1, y1, x2, y2))

    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_arr)
    # raw bboxes (non-expanded) for point-to-rectangle distance
    bboxes_np = np.array(bboxes_list, dtype=np.float32)

    # --- 2. Centroid ---
    M = cv2.moments(hull)
    if M['m00'] > 0:
        center = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
    else:
        center = np.mean(hull.reshape(-1, 2).astype(float), axis=0)
    center_adb = (int(center[0]), int(center[1]))

    # --- 3. Mean radius + distance cap ---
    hull_pts = hull.reshape(-1, 2).astype(float)
    radii = np.linalg.norm(hull_pts - center, axis=1)
    mean_radius = float(np.mean(radii))
    max_radius = mean_radius * 1.3 

    # --- Zoom-adaptive offset ---
    if offset_px is None:
        hull_area = cv2.contourArea(hull)
        zoom_ratio = hull_area / (ADB_WIDTH * ADB_HEIGHT * 0.55)
        if zoom_ratio < 0.40:
            offset_px = OFFSET_BY_ZOOM['dezoome']
            zoom_label = 'dezoome'
        elif zoom_ratio < 0.55:
            offset_px = OFFSET_BY_ZOOM['moyen']
            zoom_label = 'moyen'
        else:
            offset_px = OFFSET_BY_ZOOM['zoome']
            zoom_label = 'zoome'
    else:
        zoom_label = 'custom'

    # --- 4. Angular raycasting ---
    positions = []
    rejected_rays = []
    rejected = 0
    step = 10

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        direction = np.array([np.cos(angle), -np.sin(angle)])

        # Phase A: find the hull exit point
        exit_point = None
        dist = 0.0
        aborted = False
        abort_reason = None
        last_px, last_py = int(center[0]), int(center[1])

        while dist < max_radius:
            dist += step
            pos = center + direction * dist
            px, py = float(pos[0]), float(pos[1])
            last_px, last_py = int(px), int(py)

            if (px < SCREEN_MARGIN or px > ADB_WIDTH - SCREEN_MARGIN or
                py < SCREEN_MARGIN or py > ADB_HEIGHT - SCREEN_MARGIN):
                aborted = True
                abort_reason = 'hors_ecran'
                break

            if _is_in_exclusion_zone(int(px), int(py), ADB_HEIGHT, ADB_WIDTH):
                aborted = True
                abort_reason = 'ui_avant_hull'
                break

            if cv2.pointPolygonTest(hull, (px, py), False) >= 0:
                continue

            exit_point = pos
            break

        if exit_point is None or aborted:
            rejected += 1
            if return_debug:
                reason = abort_reason or 'max_radius_atteint'
                rejected_rays.append((int(np.degrees(angle)), reason,
                                      (last_px, last_py)))
            continue

        # Phase B: place with bounded push
        found = None
        last_candidate = None
        for push in range(0, MAX_RADIAL_PUSH + 1, step):
            candidate = exit_point + direction * (offset_px + push)
            cx_, cy_ = float(candidate[0]), float(candidate[1])
            last_candidate = (int(cx_), int(cy_))

            if (cx_ < SCREEN_MARGIN or cx_ > ADB_WIDTH - SCREEN_MARGIN or
                cy_ < SCREEN_MARGIN or cy_ > ADB_HEIGHT - SCREEN_MARGIN):
                break
            if _is_in_exclusion_zone(int(cx_), int(cy_), ADB_HEIGHT, ADB_WIDTH):
                break

            dx = np.maximum.reduce([
                bboxes_np[:, 0] - cx_,
                cx_ - bboxes_np[:, 2],
                np.zeros(len(bboxes_np), dtype=np.float32),
            ])
            dy = np.maximum.reduce([
                bboxes_np[:, 1] - cy_,
                cy_ - bboxes_np[:, 3],
                np.zeros(len(bboxes_np), dtype=np.float32),
            ])
            dists = np.sqrt(dx * dx + dy * dy)
            if dists.min() >= MIN_BUILDING_DIST:
                if img_bgr is not None and _is_in_red_overlay(img_bgr, cx_, cy_):
                    continue
                found = (int(cx_), int(cy_))
                break

        if found is not None:
            positions.append(found)
        else:
            rejected += 1
            if return_debug:
                rejected_rays.append((int(np.degrees(angle)), 'push_epuise',
                                      last_candidate or (last_px, last_py)))

    # --- 5. Dedup + sort by angle + subsample ---
    if len(positions) < 3:
        print(f" WARNING: Raycasting : seulement {len(positions)} positions "
              f"({rejected} rayons rejetés)")
        if return_debug:
            return None, center_adb, False, {
                'rejected_rays': rejected_rays,
                'mean_radius': mean_radius,
                'max_radius': max_radius,
                'reason': 'insufficient_positions',
            }
        return None, center_adb, False

    # Dedup: remove positions too close to each other (< 20px apart)
    unique = [positions[0]]
    for px, py in positions[1:]:
        if not any((px - ux) ** 2 + (py - uy) ** 2 < 400 for ux, uy in unique):
            unique.append((px, py))

    # Sort by angle from center (0 = East, trigonometric direction)
    unique.sort(
        key=lambda p: np.arctan2(-(p[1] - center[1]), p[0] - center[0]),
        reverse=True,
    )

    # Subsample if too many positions
    if len(unique) > num_points:
        step_s = len(unique) / num_points
        unique = [unique[int(i * step_s)] for i in range(num_points)]

    print(f" Zone YOLO raycast : {len(buildings)} bats, "
          f"r̄={mean_radius:.0f}px, offset={offset_px}px ({zoom_label}), "
          f"{len(unique)}/{num_points} pos, {rejected} rejetés")

    if return_debug:
        return unique, center_adb, True, {
            'rejected_rays': rejected_rays,
            'mean_radius': mean_radius,
            'max_radius': max_radius,
        }
    return unique, center_adb, True

# =============================================================================
# DEBUG: VISUAL LOG OF THE DEPLOYMENT ZONE
# =============================================================================

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


# =============================================================================
# FALLBACK
# =============================================================================

def _fallback_positions(direction_idx, spread=0.5, num_points=12):
    """
    Default deployment positions (fixed coordinates).
    Used when zone detection fails.
    """
    margin = 80
    centers = {
        0: (ADB_WIDTH // 2, margin),
        1: (ADB_WIDTH - margin, margin),
        2: (ADB_WIDTH - margin, ADB_HEIGHT // 2 - 100),
        3: (ADB_WIDTH - margin, ADB_HEIGHT - 250),
        4: (ADB_WIDTH // 2, ADB_HEIGHT - 250),
        5: (margin, ADB_HEIGHT - 250),
        6: (margin, ADB_HEIGHT // 2 - 100),
        7: (margin, margin),
    }

    cx, cy = centers[direction_idx]
    max_spread_px = 400
    spread_px = int(spread * max_spread_px)

    positions = []
    for i in range(num_points):
        offset = int((i - num_points / 2) * (spread_px / max(num_points - 1, 1)))

        if direction_idx in (0, 4):
            x, y = cx + offset, cy
        elif direction_idx in (2, 6):
            x, y = cx, cy + offset
        else:
            x = cx + offset
            y = cy + (offset if direction_idx in (3, 5) else -offset)

        x = max(margin, min(ADB_WIDTH - margin, x))
        y = max(margin, min(ADB_HEIGHT - 250, y))
        positions.append((int(x), int(y)))

    return positions


# =============================================================================
# DEBUG / VISUALIZATION
# =============================================================================

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