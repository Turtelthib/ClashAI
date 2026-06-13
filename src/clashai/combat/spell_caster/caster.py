# clashai/combat/spell_caster/caster.py
# SpellCaster V2 — intelligent heal / rage / freeze targeting mid-combat.

import cv2
import numpy as np

from clashai.combat.spell_caster.constants import (
    ADB_WIDTH, ADB_HEIGHT, FREEZE_PRIORITY_WEIGHTS, FREEZE_MAX_RANGE,
)
from clashai.combat.spell_caster.health_bars import detect_health_bars
from clashai.combat.spell_caster.clustering import cluster_positions


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
