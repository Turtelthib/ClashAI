# clashai/combat/spell_caster.py
# SpellCaster V2 — intelligent spell targeting mid-combat.
#
# V2 changes:
# - FREEZE: targets the inferno tower / eagle artillery closest to troops
#   (uses pre-attack YOLO positions)
# - HEAL: targets the cluster of INJURED troops (red/orange health bars)
#   instead of the main cluster
# - RAGE: unchanged (in front of troops, toward village center)
#
# Method:
# 1. On reset, receive dangerous building positions from YOLO
# 2. Mid-combat, screenshot -> health bar detection -> clustering
# 3. Freeze -> nearest inferno to the main cluster
# 4. Heal -> cluster of red/orange bars (injured troops)
# 5. Rage -> 50px in front of troops toward village center
#
# Usage:
# caster = SpellCaster()
# caster.set_defense_positions(buildings)  # From pre-attack YOLO
# targets = caster.analyze_battlefield(screenshot_pil, village_center)

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

# Re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# --- Green health bars (healthy troops) ---
HP_BAR_H_MIN = 45
HP_BAR_H_MAX = 85
HP_BAR_S_MIN = 100
HP_BAR_V_MIN = 120

# --- Red/orange health bars (injured troops) ---
HP_RED_H_MIN = 0
HP_RED_H_MAX = 10
HP_RED_S_MIN = 120
HP_RED_V_MIN = 120

HP_ORANGE_H_MIN = 10
HP_ORANGE_H_MAX = 25
HP_ORANGE_S_MIN = 100
HP_ORANGE_V_MIN = 120

# --- Health bar size ---
HP_BAR_MIN_AREA = 30
HP_BAR_MAX_AREA = 800
HP_BAR_MIN_RATIO = 1.5

# --- UI exclusion zones ---
UI_EXCLUSION_Y = 0.60
UI_EXCLUSION_TOP = 0.08

# --- Defense classes to target with freeze ---
FREEZE_PRIORITY_CLASSES = [
    'tour_enfer_mono',
    'tour_enfer_multiple',
    'aigle_artilleur',
    'catapulte_erratique',
    'arcX_sol', 'arcX_sol_air',
    'monolithe',
]

# Priority weights (higher = higher priority for freeze)
FREEZE_PRIORITY_WEIGHTS = {
    'tour_enfer_mono': 10.0,
    'tour_enfer_multiple': 10.0,
    'aigle_artilleur': 7.0,
    'catapulte_erratique': 5.0,
    'arcX_sol': 4.0,
    'arcX_sol_air': 4.0,
    'monolithe': 6.0,
}

# Max distance to consider a defense threatening to troops (ADB pixels)
FREEZE_MAX_RANGE = 600


# =============================================================================
# TROOP DETECTION
# =============================================================================

def detect_health_bars(img_cv, color='green'):
    """
    Detects health bars on a combat screenshot.

    Args:
        img_cv: BGR image
        color: 'green' (healthy), 'red' (injured), 'all' (both)

    Returns:
        positions: list of (x, y) — centers of detected bars
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    y_min = int(h * UI_EXCLUSION_TOP)
    y_max = int(h * UI_EXCLUSION_Y)
    roi_hsv = hsv[y_min:y_max, :, :]

    if color == 'green':
        mask = cv2.inRange(roi_hsv,
                           (HP_BAR_H_MIN, HP_BAR_S_MIN, HP_BAR_V_MIN),
                           (HP_BAR_H_MAX, 255, 255))
    elif color == 'red':
        # Red
        mask1 = cv2.inRange(roi_hsv,
                            (HP_RED_H_MIN, HP_RED_S_MIN, HP_RED_V_MIN),
                            (HP_RED_H_MAX, 255, 255))
        mask2 = cv2.inRange(roi_hsv,
                            (170, HP_RED_S_MIN, HP_RED_V_MIN),
                            (180, 255, 255))
        # Orange
        mask3 = cv2.inRange(roi_hsv,
                            (HP_ORANGE_H_MIN, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN),
                            (HP_ORANGE_H_MAX, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
    else:
        mask_g = cv2.inRange(roi_hsv,
                             (HP_BAR_H_MIN, HP_BAR_S_MIN, HP_BAR_V_MIN),
                             (HP_BAR_H_MAX, 255, 255))
        mask_r1 = cv2.inRange(roi_hsv,
                              (HP_RED_H_MIN, HP_RED_S_MIN, HP_RED_V_MIN),
                              (HP_RED_H_MAX, 255, 255))
        mask_r2 = cv2.inRange(roi_hsv,
                              (170, HP_RED_S_MIN, HP_RED_V_MIN),
                              (180, 255, 255))
        mask_o = cv2.inRange(roi_hsv,
                             (HP_ORANGE_H_MIN, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN),
                             (HP_ORANGE_H_MAX, 255, 255))
        mask = cv2.bitwise_or(mask_g, mask_r1)
        mask = cv2.bitwise_or(mask, mask_r2)
        mask = cv2.bitwise_or(mask, mask_o)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < HP_BAR_MIN_AREA or area > HP_BAR_MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        if bw / bh < HP_BAR_MIN_RATIO:
            continue
        cx = x + bw // 2
        cy = (y + bh // 2) + y_min
        positions.append((cx, cy))

    return positions


def cluster_positions(positions, min_cluster_size=2, cluster_radius=150):
    """
    Groups nearby positions into clusters.

    Returns:
        clusters: list of {'center': (x,y), 'size': n, 'points': [...]}
                  sorted by descending size
    """
    if not positions:
        return []

    points = np.array(positions, dtype=float)
    visited = [False] * len(points)
    clusters = []

    for i in range(len(points)):
        if visited[i]:
            continue

        cluster_pts = [i]
        visited[i] = True
        queue = [i]

        while queue:
            current = queue.pop(0)
            for j in range(len(points)):
                if visited[j]:
                    continue
                dist = np.linalg.norm(points[current] - points[j])
                if dist < cluster_radius:
                    visited[j] = True
                    cluster_pts.append(j)
                    queue.append(j)

        if len(cluster_pts) >= min_cluster_size:
            cluster_points = points[cluster_pts]
            center = cluster_points.mean(axis=0)
            clusters.append({
                'center': (int(center[0]), int(center[1])),
                'size': len(cluster_pts),
                'points': cluster_points.tolist(),
            })

    clusters.sort(key=lambda c: c['size'], reverse=True)
    return clusters


# =============================================================================
# SPELL CASTER V2
# =============================================================================

class SpellCaster:
    """
    Intelligent spell targeting V2.

    V2 additions:
    - Freeze targeted at the inferno tower / eagle closest to troops
    - Heal targeted at injured troops (red/orange health bars)
    - Uses pre-attack YOLO positions for targeting
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._defense_targets = []

    def set_defense_positions(self, buildings):
        """
        Registers dangerous defense positions from pre-attack YOLO detection.

        Call once at reset(), before combat starts.

        Args:
            buildings: list of dicts [{class, confidence, bbox, center}, ...]
        """
        self._defense_targets = []

        for b in buildings:
            cls_name = b['class']
            if cls_name in FREEZE_PRIORITY_WEIGHTS:
                cx, cy = b['center']
                priority = FREEZE_PRIORITY_WEIGHTS[cls_name]
                self._defense_targets.append((cx, cy, cls_name, priority))

        # Sort by descending priority
        self._defense_targets.sort(key=lambda t: t[3], reverse=True)

        if self.verbose and self._defense_targets:
            from clashai.config.logging import pp, priority_tag, styled
            pp(f" SpellCaster V2 : {len(self._defense_targets)} freeze targets registered",
               tag='init_done')
            for x, y, name, prio in self._defense_targets[:5]:
                name_str = styled(name, 'def_name')
                prio_str = styled(f"prio={prio:.0f}", priority_tag(int(prio)))
                pp(f" {name_str} at ({x}, {y}) {prio_str}")

    def analyze_battlefield(self, screenshot_pil, village_center_adb=None):
        """
        Analyzes a mid-combat screenshot and returns spell targets.

        Args:
            screenshot_pil: PIL Image of the ongoing combat
            village_center_adb: (x, y) village center in ADB coordinates

        Returns:
            targets: dict {
                'troop_cluster': (x, y),
                'heal': (x, y),
                'rage': (x, y),
                'freeze': (x, y),
                'freeze_target_name': str or None,
                'num_troops': int,
                'num_hurt': int,
                'num_clusters': int,
            }
        """
        img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        img_h, img_w = img_cv.shape[:2]

        scale_x = ADB_WIDTH / img_w
        scale_y = ADB_HEIGHT / img_h

        # 1. Detect green health bars (healthy troops)
        green_bars = detect_health_bars(img_cv, 'green')

        # 2. Detect red/orange bars (injured troops)
        hurt_bars = detect_health_bars(img_cv, 'red')

        if self.verbose:
            print(f" Health bars detected: "
                  f"{len(green_bars)} healthy, {len(hurt_bars)} injured")

        # Fallback if nothing detected
        all_bars = green_bars + hurt_bars
        if not all_bars:
            fallback = village_center_adb or (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50)
            return {
                'troop_cluster': fallback,
                'heal': fallback,
                'rage': fallback,
                'freeze': fallback,
                'freeze_target_name': None,
                'num_troops': 0,
                'num_hurt': 0,
                'num_clusters': 0,
            }

        # 3. Convert to ADB coordinates
        all_adb = [(int(x * scale_x), int(y * scale_y)) for x, y in all_bars]
        hurt_adb = [(int(x * scale_x), int(y * scale_y)) for x, y in hurt_bars]

        # 4. Cluster all troops
        clusters = cluster_positions(all_adb, min_cluster_size=2, cluster_radius=150)

        if self.verbose and clusters:
            print(f" {len(clusters)} cluster(s), "
                  f"main: {clusters[0]['size']} troops "
                  f"at ({clusters[0]['center'][0]}, {clusters[0]['center'][1]})")

        main_cluster = (clusters[0]['center'] if clusters
                        else all_adb[len(all_adb) // 2])

        # ===== HEAL V2: target injured troops =====
        if hurt_adb:
            hurt_clusters = cluster_positions(hurt_adb, min_cluster_size=1,
                                              cluster_radius=200)
            if hurt_clusters:
                # Heal on the largest injured cluster
                heal_target = hurt_clusters[0]['center']
                if self.verbose:
                    print(f" Heal -> injured cluster "
                          f"({hurt_clusters[0]['size']} troops) "
                          f"at {heal_target}")
            else:
                heal_target = main_cluster
        else:
            heal_target = main_cluster

        # ===== RAGE: in front of troops =====
        if village_center_adb:
            dx = village_center_adb[0] - main_cluster[0]
            dy = village_center_adb[1] - main_cluster[1]
            norm = max(1, (dx ** 2 + dy ** 2) ** 0.5)
            rage_x = int(main_cluster[0] + dx / norm * 50)
            rage_y = int(main_cluster[1] + dy / norm * 50)
            rage_target = (
                max(60, min(ADB_WIDTH - 60, rage_x)),
                max(60, min(ADB_HEIGHT - 200, rage_y))
            )
        else:
            rage_target = main_cluster

        # ===== FREEZE V2: target the nearest dangerous defense =====
        freeze_target, freeze_name = self._find_freeze_target(main_cluster)

        return {
            'troop_cluster': main_cluster,
            'heal': heal_target,
            'rage': rage_target,
            'freeze': freeze_target,
            'freeze_target_name': freeze_name,
            'num_troops': len(all_bars),
            'num_hurt': len(hurt_bars),
            'num_clusters': len(clusters),
        }

    def analyze_from_yolo(self, raw_data, village_center_adb=None):
        """
        Analyzes spell targets from CombatObserver YOLO data.

        Same return interface as analyze_battlefield, but uses YOLO detections
        (exact positions by class) instead of HSV health bars.

        Args:
            raw_data: dict returned by CombatObserver._observe_yolo()
            village_center_adb: (x, y)

        Returns:
            targets: same format as analyze_battlefield
        """
        clusters = raw_data.get('clusters', [])
        all_pos = raw_data.get('green_positions', [])
        hurt_adb = raw_data.get('hurt_positions', [])
        hero_named = raw_data.get('hero_positions_named', {})

        fallback = village_center_adb or (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50)

        if not all_pos and not hurt_adb:
            return {
                'troop_cluster': fallback, 'heal': fallback,
                'rage': fallback, 'freeze': fallback,
                'freeze_target_name': None,
                'num_troops': 0, 'num_hurt': 0, 'num_clusters': 0,
                'hero_positions': hero_named,
            }

        main_cluster = clusters[0]['center'] if clusters else fallback

        # Heal: injured cluster
        if hurt_adb:
            from clashai.combat.combat_observer import _cluster_positions
            hurt_clusters = _cluster_positions(hurt_adb, radius=200, min_size=1)
            heal_target = hurt_clusters[0]['center'] if hurt_clusters else main_cluster
        else:
            heal_target = main_cluster

        # Rage: in front of troops (toward village center)
        if village_center_adb:
            dx = village_center_adb[0] - main_cluster[0]
            dy = village_center_adb[1] - main_cluster[1]
            norm = max(1, (dx ** 2 + dy ** 2) ** 0.5)
            rage_x = int(main_cluster[0] + dx / norm * 50)
            rage_y = int(main_cluster[1] + dy / norm * 50)
            rage_target = (max(60, min(ADB_WIDTH - 60, rage_x)),
                           max(60, min(ADB_HEIGHT - 200, rage_y)))
        else:
            rage_target = main_cluster

        # Freeze: nearest dangerous defense
        freeze_target, freeze_name = self._find_freeze_target(main_cluster)

        return {
            'troop_cluster': main_cluster,
            'heal': heal_target,
            'rage': rage_target,
            'freeze': freeze_target,
            'freeze_target_name': freeze_name,
            'num_troops': raw_data.get('num_troops', 0),
            'num_hurt': len(hurt_adb),
            'num_clusters': len(clusters),
            'hero_positions': hero_named,
        }

    def _find_freeze_target(self, troop_center):
        """
        Finds the best target for the freeze spell.

        Logic: among dangerous defenses detected by YOLO, find the one
        closest to troops AND with the highest priority. Uses a combined score:
            score = priority / (distance + 100)

        Higher priority and lower distance gives a better score.

        Args:
            troop_center: (x, y) center of the troop cluster

        Returns:
            (target_pos, target_name) or (troop_center, None) if no target found
        """
        if not self._defense_targets:
            if self.verbose:
                print(" Freeze -> no YOLO targets, fallback to troops")
            return troop_center, None

        best_score = -1
        best_target = None
        best_name = None

        tx, ty = troop_center

        for dx, dy, cls_name, priority in self._defense_targets:
            dist = np.sqrt((dx - tx) ** 2 + (dy - ty) ** 2)

            # Skip defenses too far from troops
            if dist > FREEZE_MAX_RANGE:
                continue

            # Combined score: high priority + low distance = good score
            score = priority / (dist + 100)

            if score > best_score:
                best_score = score
                best_target = (dx, dy)
                best_name = cls_name

        if best_target is not None:
            if self.verbose:
                dist = np.sqrt((best_target[0] - tx) ** 2 +
                               (best_target[1] - ty) ** 2)
                print(f" Freeze -> {best_name} at {best_target} "
                      f"(dist={dist:.0f}px, score={best_score:.3f})")
            return best_target, best_name
        else:
            if self.verbose:
                print(" Freeze -> no defense in range, "
                      "fallback in front of troops")
            return troop_center, None


# =============================================================================
# TEST
# =============================================================================

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


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else None
    test_spell_caster(img)
