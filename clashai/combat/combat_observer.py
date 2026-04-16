# clashai/combat/combat_observer.py
# Observation mid-combat pour ClashAI V4.
#
# Deux modes de perception :
#   - YOLO troupes (V4) : position exacte de chaque troupe/héro par classe
#   - Barres de vie HSV (V3 fallback) : clusters de barres vertes/rouges
#
# Quand le TroopDetector YOLO est fourni, on l'utilise en priorité et les
# barres de vie ne servent plus qu'à détecter les troupes blessées.
#
# Usage :
#   from clashai.perception.troop_detector import TroopDetector
#   detector = TroopDetector()
#   observer = CombatObserver(troop_detector=detector)
#   features, raw = observer.observe(screenshot_pil, village_center)

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

ADB_WIDTH = 1920
ADB_HEIGHT = 1080

# --- Barres de vie vertes (troupes en bonne santé) ---
HP_GREEN_H_RANGE = (45, 85)
HP_GREEN_S_MIN = 100
HP_GREEN_V_MIN = 120

# --- Barres de vie orange/rouges (troupes blessées) ---
HP_RED_H_RANGE = (0, 15)
HP_RED_S_MIN = 120
HP_RED_V_MIN = 120

HP_ORANGE_H_RANGE = (15, 30)
HP_ORANGE_S_MIN = 100
HP_ORANGE_V_MIN = 120

# --- Taille des barres de vie ---
HP_BAR_MIN_AREA = 30
HP_BAR_MAX_AREA = 800
HP_BAR_MIN_RATIO = 1.5  # largeur/hauteur (les barres sont horizontales)

# --- Barres de vie héros (plus grandes que les troupes normales) ---
HERO_BAR_MIN_AREA = 200
HERO_BAR_MAX_AREA = 2000
HERO_BAR_MIN_RATIO = 2.0

# --- Zones d'exclusion UI ---
UI_BOTTOM_Y = 0.60   # Barre troupes, boutons
UI_TOP_Y = 0.08      # Timer, ressources
UI_LEFT_X = 0.02
UI_RIGHT_X = 0.98

# --- Clustering ---
CLUSTER_RADIUS = 150    # pixels ADB
MIN_CLUSTER_SIZE = 2

# Nombre de features en sortie
COMBAT_FEATURES_SIZE = 15


# =============================================================================
#                    DÉTECTION DES BARRES DE VIE
# =============================================================================

def _detect_bars(img_cv, h_range, s_min, v_min, min_area, max_area, min_ratio):
    """
    Détecte des barres de vie horizontales par couleur HSV.
    
    Returns:
        positions: liste de (x, y) en coordonnées image
        areas: liste des aires de chaque barre (pour distinguer héros/troupes)
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]
    
    # Masque couleur
    lower = np.array([h_range[0], s_min, v_min])
    upper = np.array([h_range[1], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Exclure les zones UI
    mask[:int(h * UI_TOP_Y), :] = 0
    mask[int(h * UI_BOTTOM_Y):, :] = 0
    mask[:, :int(w * UI_LEFT_X)] = 0
    mask[:, int(w * UI_RIGHT_X):] = 0
    
    # Nettoyer
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Trouver les contours
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
    """Détecte les barres de vie vertes (troupes en bonne santé)."""
    return _detect_bars(
        img_cv, HP_GREEN_H_RANGE, HP_GREEN_S_MIN, HP_GREEN_V_MIN,
        HP_BAR_MIN_AREA, HP_BAR_MAX_AREA, HP_BAR_MIN_RATIO
    )


def detect_hurt_bars(img_cv):
    """Détecte les barres de vie rouges/oranges (troupes blessées)."""
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
    Détecte les barres de vie des héros (plus grandes que les troupes normales).
    Cherche les barres vertes ET oranges/rouges de grande taille.
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
    # Nombre de héros détectés (probablement 0-5)
    # On ne peut pas distinguer QUEL héros c'est juste par la barre,
    # mais on sait combien sont vivants.
    return all_pos


# =============================================================================
#                    CLUSTERING
# =============================================================================

def _cluster_positions(positions, radius=CLUSTER_RADIUS, min_size=MIN_CLUSTER_SIZE):
    """
    Clustering simple par distance (BFS).
    
    Returns:
        clusters: list of {'center': (x,y), 'size': n}
        trié par taille décroissante.
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
#                    OBSERVATEUR PRINCIPAL
# =============================================================================

class CombatObserver:
    """
    Observe le champ de bataille pendant le combat et retourne
    un vecteur de features compactes pour l'agent PPO.
    
    Features (15 dims) — compatibles V3 :
        0  : phase (0.0 = deploy, 1.0 = combat)
        1  : combat_progress (0.0-1.0, temps écoulé / temps max)
        2  : num_troops_alive (normalisé /50)
        3  : num_troops_hurt (normalisé /20)
        4  : num_heroes_alive (normalisé /5)
        5  : main_cluster_x (normalisé 0-1)
        6  : main_cluster_y (normalisé 0-1)
        7  : num_clusters (normalisé /5)
        8  : cluster_spread (distance max entre clusters, normalisé)
        9  : troops_near_center (% de troupes proches du centre village)
        10 : hurt_ratio (troupes blessées / total)
        11-14 : spells_remaining (soin, rage, gel, pad) normalisé
    
    Quand troop_detector est fourni (V4), les features 2-9 sont calculées
    à partir des détections YOLO (plus précises que les barres de vie).
    Le raw_data contient alors les détections complètes par classe.
    """
    
    def __init__(self, verbose: bool = True, troop_detector=None):
        self.verbose = verbose
        self._combat_start_time = None
        self._max_combat_time = 180.0  # 3 minutes
        self._troop_detector = troop_detector  # TroopDetector ou None
    
    @property
    def has_yolo(self) -> bool:
        return self._troop_detector is not None
    
    def start_combat(self):
        """Appelé quand le combat commence (transition deploy → combat)."""
        import time
        self._combat_start_time = time.time()
    
    def observe(self, screenshot_pil, village_center_adb=None,
                spells_remaining=None, phase='combat'):
        """
        Analyse un screenshot mid-combat.
        
        Args:
            screenshot_pil: PIL Image
            village_center_adb: (x, y) centre du village
            spells_remaining: dict {'soin': n, 'rage': n, 'gel': n}
            phase: 'deploy' ou 'combat'
        
        Returns:
            combat_features: np.array (COMBAT_FEATURES_SIZE,)
            raw_data: dict avec les données brutes pour le SpellCaster
        """
        import time
        
        features = np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32)
        
        img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        img_h, img_w = img_cv.shape[:2]
        scale_x = ADB_WIDTH / img_w
        scale_y = ADB_HEIGHT / img_h
        
        # Feature 0: Phase
        features[0] = 1.0 if phase == 'combat' else 0.0
        
        # Feature 1: Combat progress
        if self._combat_start_time and phase == 'combat':
            elapsed = time.time() - self._combat_start_time
            features[1] = min(elapsed / self._max_combat_time, 1.0)
        
        # === DÉTECTION ===
        if self._troop_detector is not None:
            raw_data = self._observe_yolo(screenshot_pil, img_cv, scale_x, scale_y,
                                           features, village_center_adb)
        else:
            raw_data = self._observe_bars(img_cv, scale_x, scale_y,
                                           features, village_center_adb)
        
        # Features 11-14: Sorts restants (commun aux deux modes)
        if spells_remaining:
            features[11] = min(spells_remaining.get('soin', 0) / 2.0, 1.0)
            features[12] = min(spells_remaining.get('rage', 0) / 3.0, 1.0)
            features[13] = min(spells_remaining.get('gel', 0) / 1.0, 1.0)
        
        return features, raw_data

    # -----------------------------------------------------------------
    #  V4 : observation via YOLO troupes
    # -----------------------------------------------------------------
    def _observe_yolo(self, screenshot_pil, img_cv, scale_x, scale_y,
                      features, village_center_adb):
        """Remplit features[2-10] et raw_data via YOLO."""
        grouped = self._troop_detector.detect_grouped(screenshot_pil)
        
        all_dets = grouped['all']
        troops = grouped['troops']
        heroes = grouped['heroes']
        
        # Barres de vie pour les blessés (YOLO ne voit pas la santé)
        hurt_pos, _ = detect_hurt_bars(img_cv)
        hurt_adb = [(int(x * scale_x), int(y * scale_y)) for x, y in hurt_pos]
        
        # Positions ADB de toutes les troupes YOLO
        all_positions = [(d.x, d.y) for d in all_dets]
        
        # Feature 2: Troupes en vie (YOLO count, plus fiable)
        features[2] = min(len(troops) / 50.0, 1.0)
        
        # Feature 3: Troupes blessées (barres rouges proches d'une détection YOLO)
        num_hurt = self._match_hurt_to_yolo(hurt_adb, all_positions)
        features[3] = min(num_hurt / 20.0, 1.0)
        
        # Feature 4: Héros en vie
        features[4] = min(len(heroes) / 5.0, 1.0)
        
        # Clustering
        clusters = _cluster_positions(all_positions)
        self._fill_cluster_features(features, clusters, all_positions, village_center_adb)
        
        # Feature 10: Hurt ratio
        total = len(all_dets)
        features[10] = num_hurt / max(total, 1)
        
        # Raw data enrichi pour SpellCaster V3
        hero_positions = {d.class_name: (d.x, d.y) for d in heroes}
        
        raw_data = {
            # Compat V3
            'green_positions': [(d.x, d.y) for d in troops],
            'hurt_positions': hurt_adb,
            'hero_positions': [(d.x, d.y) for d in heroes],
            'clusters': clusters,
            'main_cluster': clusters[0]['center'] if clusters else None,
            'num_troops': total,
            'num_heroes': len(heroes),
            # Nouveau V4
            'yolo_detections': all_dets,
            'yolo_grouped': grouped,
            'hero_positions_named': hero_positions,
            # V4.1: comptage depuis les détections existantes (évite un 2e appel YOLO)
            'troop_counts': self._count_from_detections(all_dets),
        }
        
        if self.verbose:
            counts = {}
            for d in all_dets:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
            summary = ', '.join(f"{v}×{k}" for k, v in counts.items())
            print(f"      👁️  YOLO: {summary} | {num_hurt} blessés | {len(clusters)} clusters")
        
        return raw_data

    def _match_hurt_to_yolo(self, hurt_adb, yolo_positions, radius=80):
        """
        Compte les barres de vie rouges qui sont proches d'une détection YOLO.
        Évite les faux positifs (bâtiments ennemis endommagés).
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
        Compte les détections par classe sans re-lancer YOLO.
        V4.1: remplace count_by_class() qui faisait une 2e inférence.
        """
        counts = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    # -----------------------------------------------------------------
    #  V3 fallback : observation via barres de vie HSV
    # -----------------------------------------------------------------
    def _observe_bars(self, img_cv, scale_x, scale_y, features, village_center_adb):
        """Remplit features[2-10] et raw_data via barres de vie (V3)."""
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
            print(f"      👁️  Barres: {len(green_pos)} saines, "
                  f"{len(hurt_pos)} blessées, "
                  f"{len(hero_pos)} héros, "
                  f"{len(clusters)} clusters")
        
        return raw_data

    # -----------------------------------------------------------------
    #  Utilitaire commun : features de clusters
    # -----------------------------------------------------------------
    def _fill_cluster_features(self, features, clusters, all_positions, village_center_adb):
        """Remplit features[5-9] à partir des clusters."""
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
#                            TEST
# =============================================================================

if __name__ == "__main__":
    print("🧪 Test CombatObserver\n")
    
    observer = CombatObserver()
    observer.start_combat()
    
    # Test avec image synthétique
    import time
    time.sleep(0.1)
    
    # Créer une image de test (noir avec quelques barres vertes)
    test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Simuler des barres de vie vertes
    for y in [300, 320, 350, 400, 410]:
        cv2.rectangle(test_img, (800, y), (840, y+4), (0, 200, 0), -1)
    # Simuler des barres rouges
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
        print(f"   [{i:2d}] {label:18s} = {val:.3f}")
    
    print(f"\nRaw: {raw['num_troops']} troupes, {raw['num_heroes']} héros")
    print("✅ Test terminé !")