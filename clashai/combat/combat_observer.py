# clashai/combat/combat_observer.py
# Mid-combat observation for ClashAI V4.
#
# Two perception modes:
# - YOLO troops (V4): exact position of each troop/hero by class
# - HSV health bars (V3 fallback): clusters of green/red bars
#
# When a YOLO TroopDetector is provided, it takes priority and health bars
# are only used to detect injured troops.
#
# Usage:
# from clashai.perception.troop_detector import TroopDetector
# detector = TroopDetector()
# observer = CombatObserver(troop_detector=detector)
# features, raw = observer.observe(screenshot_pil, village_center)

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

# Re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# --- Green health bars (healthy troops) ---
HP_GREEN_H_RANGE = (45, 85)
HP_GREEN_S_MIN = 100
HP_GREEN_V_MIN = 120

# --- Red/orange health bars (injured troops) ---
HP_RED_H_RANGE = (0, 15)
HP_RED_S_MIN = 120
HP_RED_V_MIN = 120

HP_ORANGE_H_RANGE = (15, 30)
HP_ORANGE_S_MIN = 100
HP_ORANGE_V_MIN = 120

# --- Health bar size ---
HP_BAR_MIN_AREA = 30
HP_BAR_MAX_AREA = 800
HP_BAR_MIN_RATIO = 1.5

# --- Hero health bars (larger than normal troop bars) ---
HERO_BAR_MIN_AREA = 200
HERO_BAR_MAX_AREA = 2000
HERO_BAR_MIN_RATIO = 2.0

# --- UI exclusion zones ---
UI_BOTTOM_Y = 0.60
UI_TOP_Y = 0.08
UI_LEFT_X = 0.02
UI_RIGHT_X = 0.98

# --- Clustering ---
CLUSTER_RADIUS = 150
MIN_CLUSTER_SIZE = 2

# Number of output features
COMBAT_FEATURES_SIZE = 15


# =============================================================================
# HEALTH BAR DETECTION
# =============================================================================

def _detect_bars(img_cv, h_range, s_min, v_min, min_area, max_area, min_ratio):
    """
    Detects horizontal health bars by HSV color.

    Returns:
        positions: list of (x, y) in image coordinates
        areas: list of areas for each bar (to distinguish heroes from troops)
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    # Color mask
    lower = np.array([h_range[0], s_min, v_min])
    upper = np.array([h_range[1], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Exclude UI zones
    mask[:int(h * UI_TOP_Y), :] = 0
    mask[int(h * UI_BOTTOM_Y):, :] = 0
    mask[:, :int(w * UI_LEFT_X)] = 0
    mask[:, int(w * UI_RIGHT_X):] = 0

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
        if h_rect == 0:
            continue
        ratio = w_rect / h_rect
        if ratio < min_ratio:
            continue

        cx = x_rect + w_rect // 2
        cy = y_rect + h_rect // 2
        positions.append((cx, cy))
        areas.append(area)

    return positions, areas


def detect_troop_bars(img_cv):
    """Detects green health bars (healthy troops)."""
    return _detect_bars(
        img_cv, HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )


def detect_hurt_bars(img_cv):
    """Detects red/orange health bars (injured troops)."""
    red_pos, red_areas = _detect_bars(
        img_cv, HP_RED_H_RANGE, HP_RED_S_MIN, HP_RED_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )
    orange_pos, orange_areas = _detect_bars(
        img_cv, HP_ORANGE_H_RANGE, HP_ORANGE_S_MIN, HP_ORANGE_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )
    return red_pos + orange_pos, red_areas + orange_areas


def detect_hero_bars(img_cv):
    """
    Detects hero health bars (larger than normal troop bars).
    Looks for large green AND orange/red bars.
    """
    green_pos, green_areas = _detect_bars(
        img_cv, HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
        HERO_BAR_MIN_AREA, HERO_BAR_MAX_AREA, HERO_BAR_MIN_RATIO
    )
    red_pos, red_areas = _detect_bars(
        img_cv, HP_RED_H_RANGE, HP_RED_S_MIN, HP_RED_V_MIN,
        HERO_BAR_MIN_AREA, HERO_BAR_MAX_AREA, HERO_BAR_MIN_RATIO
    )

    all_pos = green_pos + red_pos
    # Number of heroes detected (typically 0-5).
    # We cannot identify WHICH hero from the bar alone,
    # but we know how many are alive.
    return all_pos


# =============================================================================
# CLUSTERING
# =============================================================================

def _cluster_positions(positions, radius=CLUSTER_RADIUS, min_size=MIN_CLUSTER_SIZE):
    """
    Simple distance-based clustering (BFS).

    Returns:
        clusters: list of {'center': (x,y), 'size': n}
        sorted by descending size.
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
                if dist < radius:
                    visited[j] = True
                    cluster_pts.append(j)
                    queue.append(j)

        if len(cluster_pts) >= min_size:
            center = points[cluster_pts].mean(axis=0)
            clusters.append({
                'center': (int(center[0]), int(center[1])),
                'size': len(cluster_pts),
            })

    clusters.sort(key=lambda c: c['size'], reverse=True)
    return clusters


# =============================================================================
# MAIN OBSERVER
# =============================================================================

class CombatObserver:
    """
    Observes the battlefield during combat and returns a compact feature
    vector for the PPO agent.

    Features (15 dims) — V3 compatible:
        0 : buildings_remaining_ratio (1.0=none destroyed, 0.0=all destroyed)
        1 : combat_progress (0.0-1.0, elapsed / max time)
        2 : num_troops_alive (normalized /50)
        3 : num_troops_hurt (normalized /20)
        4 : num_heroes_alive (normalized /5)
        5 : main_cluster_x (normalized 0-1)
        6 : main_cluster_y (normalized 0-1)
        7 : num_clusters (normalized /5)
        8 : cluster_spread (max distance between clusters, normalized)
        9 : troops_near_center (% of troops close to village center)
        10 : hurt_ratio (injured troops / total)
        11-14 : spells_remaining (heal, rage, freeze, pad) normalized

    When troop_detector is provided (V4), features 2-9 are computed from
    YOLO detections (more reliable than health bars).
    raw_data then contains the full per-class detections.
    """

    def __init__(self, verbose: bool = True, troop_detector=None):
        self.verbose = verbose
        self._combat_start_time = None
        self._max_combat_time = 180.0
        self._troop_detector = troop_detector
        self._initial_building_count = 0
        self._current_building_count = 0

    @property
    def has_yolo(self) -> bool:
        return self._troop_detector is not None

    def start_combat(self, initial_building_count=None):
        """Called at reset — starts the timer and initializes the building counter."""
        import time
        self._combat_start_time = time.time()
        self._initial_building_count = initial_building_count or 0
        self._current_building_count = initial_building_count or 0

    def observe(self, screenshot_pil, village_center_adb=None,
                spells_remaining=None, phase='combat', buildings_count=None):
        """
        Analyzes a mid-combat screenshot.

        Args:
            screenshot_pil: PIL Image
            village_center_adb: (x, y) village center
            spells_remaining: dict {'soin': n, 'rage': n, 'gel': n}
            phase: 'deploy' or 'combat' (kept for V3 compatibility, unused for feature[0])
            buildings_count: number of buildings still standing (optional)

        Returns:
            combat_features: np.array (COMBAT_FEATURES_SIZE,)
            raw_data: dict with raw data for SpellCaster
        """
        import time

        if buildings_count is not None:
            self._current_building_count = buildings_count

        features = np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32)

        img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        img_h, img_w = img_cv.shape[:2]
        scale_x = ADB_WIDTH / img_w
        scale_y = ADB_HEIGHT / img_h

        # Feature 0: remaining buildings ratio (0=all destroyed, 1=none destroyed)
        if self._initial_building_count > 0:
            features[0] = self._current_building_count / self._initial_building_count
        else:
            features[0] = 1.0

        # Feature 1: Combat progress
        if self._combat_start_time:
            elapsed = time.time() - self._combat_start_time
            features[1] = min(elapsed / self._max_combat_time, 1.0)

        # === DETECTION ===
        if self._troop_detector is not None:
            raw_data = self._observe_yolo(screenshot_pil, img_cv, scale_x, scale_y,
                                           features, village_center_adb)
        else:
            raw_data = self._observe_bars(img_cv, scale_x, scale_y,
                                           features, village_center_adb)

        # Features 11-14: remaining spells (common to both modes)
        if spells_remaining:
            features[11] = min(spells_remaining.get('soin', 0) / 2.0, 1.0)
            features[12] = min(spells_remaining.get('rage', 0) / 3.0, 1.0)
            features[13] = min(spells_remaining.get('gel', 0) / 1.0, 1.0)

        return features, raw_data

    # -----------------------------------------------------------------
    # V4: observation via YOLO troops
    # -----------------------------------------------------------------
    def _observe_yolo(self, screenshot_pil, img_cv, scale_x, scale_y,
                      features, village_center_adb):
        """Fills features[2-10] and raw_data via YOLO."""
        grouped = self._troop_detector.detect_grouped(screenshot_pil)

        all_dets = grouped['all']
        troops = grouped['troops']
        heroes = grouped['heroes']

        # Health bars for injured detection (YOLO does not see health)
        hurt_pos, _ = detect_hurt_bars(img_cv)
        hurt_adb = [(int(x * scale_x), int(y * scale_y)) for x, y in hurt_pos]

        # ADB positions of all YOLO troops
        all_positions = [(d.x, d.y) for d in all_dets]

        # Feature 2: troops alive (YOLO count, more reliable)
        features[2] = min(len(troops) / 50.0, 1.0)

        # Feature 3: injured troops (red bars near a YOLO detection)
        num_hurt = self._match_hurt_to_yolo(hurt_adb, all_positions)
        features[3] = min(num_hurt / 20.0, 1.0)

        # Feature 4: heroes alive
        features[4] = min(len(heroes) / 5.0, 1.0)

        # Clustering
        clusters = _cluster_positions(all_positions)
        self._fill_cluster_features(features, clusters, all_positions, village_center_adb)

        # Feature 10: hurt ratio
        total = len(all_dets)
        features[10] = num_hurt / max(total, 1)

        # Raw data for SpellCaster V3 compatibility
        hero_positions = {d.class_name: (d.x, d.y) for d in heroes}

        raw_data = {
            # V3 compat
            'green_positions': [(d.x, d.y) for d in troops],
            'hurt_positions': hurt_adb,
            'hero_positions': [(d.x, d.y) for d in heroes],
            'clusters': clusters,
            'main_cluster': clusters[0]['center'] if clusters else None,
            'num_troops': total,
            'num_heroes': len(heroes),
            # V4
            'yolo_detections': all_dets,
            'yolo_grouped': grouped,
            'hero_positions_named': hero_positions,
            # V4.1: count from existing detections (avoids a second YOLO call)
            'troop_counts': self._count_from_detections(all_dets),
        }

        if self.verbose:
            counts = {}
            for d in all_dets:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
            summary = ', '.join(f"{v}x{k}" for k, v in counts.items())
            print(f" YOLO: {summary} | {num_hurt} injured | {len(clusters)} clusters")

        return raw_data

    def _match_hurt_to_yolo(self, hurt_adb, yolo_positions, radius=80):
        """
        Counts red health bars that are close to a YOLO detection.
        Avoids false positives from damaged enemy buildings.
        """
        count = 0
        for hx, hy in hurt_adb:
            for tx, ty in yolo_positions:
                if abs(hx - tx) < radius and abs(hy - ty) < radius:
                    count += 1
                    break
        return count

    def _count_from_detections(self, detections):
        """
        Counts detections by class without re-running YOLO.
        V4.1: replaces count_by_class() which triggered a second inference.
        """
        counts = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    # -----------------------------------------------------------------
    # V3 fallback: observation via HSV health bars
    # -----------------------------------------------------------------
    def _observe_bars(self, img_cv, scale_x, scale_y, features, village_center_adb):
        """Fills features[2-10] and raw_data via health bars (V3)."""
        green_pos, _ = detect_troop_bars(img_cv)
        hurt_pos, _ = detect_hurt_bars(img_cv)
        hero_pos = detect_hero_bars(img_cv)

        green_adb = [(int(x * scale_x), int(y * scale_y)) for x, y in green_pos]
        hurt_adb = [(int(x * scale_x), int(y * scale_y)) for x, y in hurt_pos]
        all_troops_adb = green_adb + hurt_adb

        features[2] = min(len(green_pos) / 50.0, 1.0)
        features[3] = min(len(hurt_pos) / 20.0, 1.0)
        features[4] = min(len(hero_pos) / 5.0, 1.0)

        clusters = _cluster_positions(all_troops_adb)
        self._fill_cluster_features(features, clusters, all_troops_adb, village_center_adb)

        total_troops = len(green_pos) + len(hurt_pos)
        features[10] = len(hurt_pos) / max(total_troops, 1)

        raw_data = {
            'green_positions': green_adb,
            'hurt_positions': hurt_adb,
            'hero_positions': [(int(x*scale_x), int(y*scale_y)) for x, y in hero_pos],
            'clusters': clusters,
            'main_cluster': clusters[0]['center'] if clusters else None,
            'num_troops': total_troops,
            'num_heroes': len(hero_pos),
        }

        if self.verbose:
            print(f" Bars: {len(green_pos)} healthy, "
                  f"{len(hurt_pos)} injured, "
                  f"{len(hero_pos)} heroes, "
                  f"{len(clusters)} clusters")

        return raw_data

    # -----------------------------------------------------------------
    # Common utility: cluster features
    # -----------------------------------------------------------------
    def _fill_cluster_features(self, features, clusters, all_positions, village_center_adb):
        """Fills features[5-9] from clusters."""
        if not clusters:
            return

        main = clusters[0]['center']
        features[5] = main[0] / ADB_WIDTH
        features[6] = main[1] / ADB_HEIGHT
        features[7] = min(len(clusters) / 5.0, 1.0)

        if len(clusters) >= 2:
            positions = [c['center'] for c in clusters]
            max_dist = max(
                np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                for i, p1 in enumerate(positions)
                for p2 in positions[i+1:]
            )
            features[8] = min(max_dist / 1000.0, 1.0)

        if village_center_adb and all_positions:
            vc = village_center_adb
            near_center = sum(
                1 for x, y in all_positions
                if np.sqrt((x-vc[0])**2 + (y-vc[1])**2) < 300
            )
            features[9] = near_center / max(len(all_positions), 1)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Test CombatObserver\n")

    observer = CombatObserver()
    observer.start_combat()

    # Test with synthetic image
    import time
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
