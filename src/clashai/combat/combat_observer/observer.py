# clashai/combat/combat_observer/observer.py
# CombatObserver — compact battlefield feature vector for the PPO agent.

import cv2
import numpy as np

from clashai.combat.combat_observer.constants import (
    ADB_WIDTH, ADB_HEIGHT, COMBAT_FEATURES_SIZE,
)
from clashai.combat.combat_observer.health_bars import (
    detect_troop_bars, detect_hurt_bars, detect_hero_bars,
)
from clashai.combat.combat_observer.clustering import _cluster_positions


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
            from clashai.config.logging import pp, styled
            counts = {}
            for d in all_dets:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
            summary = ', '.join(f"{v}x{k}" for k, v in counts.items())
            pp(f" YOLO: {styled(summary, 'yolo_alt')} | {num_hurt} injured | {len(clusters)} clusters",
               tag='yolo')

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
