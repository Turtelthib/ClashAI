# clashai/perception/deploy/positions.py
# Hull geometry -> deploy positions (angles, exclusion zones, perimeter).

import cv2
import numpy as np
from PIL import Image

from clashai.config import ADB_WIDTH, ADB_HEIGHT
from clashai.perception.deploy.constants import (
    UI_EXCLUSION_ZONES, SCREEN_MARGIN, DEPLOY_OFFSET, DIRECTION_ANGLES,
)
from clashai.perception.deploy.boundary import detect_village_boundary


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

