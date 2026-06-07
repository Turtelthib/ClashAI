# clashai/perception/deploy/yolo_zone.py
# YOLO-based deploy perimeter: walls segmentation + building bbox hull.

import cv2
import numpy as np

from clashai.config import ADB_WIDTH, ADB_HEIGHT
from clashai.perception.deploy.constants import (
    UI_EXCLUSION_ZONES, SCREEN_MARGIN, DEPLOY_OFFSET, BUILDING_PADDING,
    MIN_BUILDING_DIST, OFFSET_BY_ZOOM, MAX_RADIAL_PUSH, YOLO_WALLS_IMGSZ,
)
from clashai.perception.deploy.boundary import _is_in_red_overlay
from clashai.perception.deploy.positions import _is_in_exclusion_zone


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

    # BGR for ultralytics (it reads numpy as BGR). np.array(pil) is RGB,
    # so convert — otherwise R/B channels are swapped at inference.
    img_arr = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
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
