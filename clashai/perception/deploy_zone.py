# scripts/rl/deploy_zone.py
# Détection dynamique de la zone de déploiement sur un village ennemi.
#
# Dans Clash of Clans, la zone de déploiement est délimitée par une ligne
# rouge semi-transparente autour du village. Les troupes ne peuvent être
# posées qu'EN DEHORS de cette ligne (sur l'herbe verte extérieure).
#
# Méthode :
#   1. L'overlay rouge décale la teinte (HSV) de l'herbe de H≈33 vers H≈20
#   2. On détecte cette herbe "chaude" (H=14-28) = zone intérieure du village
#   3. On calcule le convex hull = frontière approximative
#   4. Les positions de déploiement sont placées JUSTE EN DEHORS du hull
#
# Usage :
#   from clashai.perception.deploy_zone import get_smart_deploy_positions
#   positions = get_smart_deploy_positions(screenshot_pil, direction_idx, spread)

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

# Résolution ADB (coordonnées de tap)
ADB_WIDTH = 1920
ADB_HEIGHT = 1080

# Zones d'exclusion UI (en coordonnées ADB 1920×1080)
# Les taps dans ces zones déclenchent des boutons au lieu de poser des troupes
UI_EXCLUSION_ZONES = [
    (0, 640, 1920, 1080),      # Tout le bas : boutons + barre de troupes
    (0, 0, 280, 230),          # Info joueur haut-gauche
    (1450, 0, 1920, 160),      # Ressources haut-droite
]

# Marge minimale depuis les bords de l'écran
SCREEN_MARGIN = 60

# Distance (en pixels ADB) entre le hull et les positions de déploiement
DEPLOY_OFFSET = 35

# Directions (index → label)
DIRECTION_LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']

# Angles correspondants (en radians, 0 = droite, anti-horaire)
# N=haut, E=droite, S=bas, O=gauche
DIRECTION_ANGLES = {
    0: np.pi / 2,       # N  (haut)
    1: np.pi / 4,       # NE
    2: 0,               # E  (droite)
    3: -np.pi / 4,      # SE
    4: -np.pi / 2,      # S  (bas)
    5: -3 * np.pi / 4,  # SO
    6: np.pi,           # O  (gauche)
    7: 3 * np.pi / 4,   # NO
}


# =============================================================================
#                    DÉTECTION DE LA ZONE
# =============================================================================

def detect_village_boundary(img_cv):
    """
    Détecte la frontière du village à partir d'un screenshot BGR.

    L'overlay rouge de CoC décale la teinte HSV de l'herbe : H≈33 (vert)
    devient H≈20 (jaune-vert chaud). On détecte cette zone chaude pour
    trouver l'intérieur du village.

    Args:
        img_cv: image BGR (numpy array) du screenshot

    Returns:
        hull: convex hull du village (numpy array Nx1x2) ou None
        center: centre du village (x, y) en pixels image ou None
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, w = img_cv.shape[:2]

    # --- Masque UI ---
    # Exclure les zones d'interface (en coordonnées image)
    # On calcule le ratio image→ADB pour convertir
    scale_x = w / ADB_WIDTH
    scale_y = h / ADB_HEIGHT

    ui_mask = np.ones((h, w), dtype=np.uint8) * 255
    for ax1, ay1, ax2, ay2 in UI_EXCLUSION_ZONES:
        ix1 = int(ax1 * scale_x)
        iy1 = int(ay1 * scale_y)
        ix2 = int(ax2 * scale_x)
        iy2 = int(ay2 * scale_y)
        ui_mask[iy1:iy2, ix1:ix2] = 0

    # Exclure aussi les bords extrêmes
    margin = int(SCREEN_MARGIN * min(scale_x, scale_y))
    ui_mask[:margin, :] = 0
    ui_mask[h - margin:, :] = 0
    ui_mask[:, :margin] = 0
    ui_mask[:, w - margin:] = 0

    # --- Détection herbe chaude (zone rouge du village) ---
    # H=14-28 : herbe teintée par l'overlay rouge
    # S>80 : saturée (pas gris)
    # V>100 : lumineuse (pas ombre de forêt)
    mask_warm = cv2.inRange(hsv, (14, 80, 100), (28, 255, 255))
    mask_warm = cv2.bitwise_and(mask_warm, ui_mask)

    # --- Nettoyage morphologique ---
    # Close : combler les trous (bâtiments, murs, décorations)
    kernel_close = np.ones((30, 30), np.uint8)
    # Open : retirer le bruit (petits pixels chauds dans la forêt)
    kernel_open = np.ones((15, 15), np.uint8)

    min_contour_area = h * w * 0.05

    # --- Stratégie en cascade : essayer du plus précis au plus large ---
    strategies = [
        ("warm", mask_warm),
    ]

    # Préparer les fallbacks
    # Fallback 1 : herbe claire (fonctionne quand l'overlay rouge est faible)
    mask_grass = cv2.inRange(hsv, (18, 70, 130), (38, 255, 255))
    mask_grass = cv2.bitwise_and(mask_grass, ui_mask)
    strategies.append(("herbe claire", mask_grass))

    # Fallback 2 : herbe large (très permissif)
    mask_wide = cv2.inRange(hsv, (15, 60, 110), (40, 255, 255))
    mask_wide = cv2.bitwise_and(mask_wide, ui_mask)
    strategies.append(("herbe large", mask_wide))

    # Fallback 3 : VILLAGES SOMBRES (thème sous-marin, nuit, etc.)
    # Le sol est bleu-vert (H=75-120) au lieu de vert (H=33).
    # L'intérieur du village est plus lumineux (V>90) que l'extérieur sombre (V<70).
    # On utilise la luminosité + saturation pour isoler la zone de jeu.
    mask_bright = cv2.inRange(hsv, (0, 20, 90), (180, 255, 255))  # Tout ce qui est lumineux
    # Exclure les zones UI et le blanc pur (texte, nuages)
    mask_not_white = cv2.inRange(hsv, (0, 15, 0), (180, 255, 240))  # Pas surexposé
    mask_dark_village = cv2.bitwise_and(mask_bright, mask_not_white)
    mask_dark_village = cv2.bitwise_and(mask_dark_village, ui_mask)
    strategies.append(("village sombre (luminosité)", mask_dark_village))

    # Fallback 4 : Bordure rouge directe (certains villages dark ont une ligne rouge visible)
    # H<10 ou H>170 (rouge), S>80, V>60
    mask_red_low = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_red_high = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask_red_border = cv2.bitwise_or(mask_red_low, mask_red_high)
    mask_red_border = cv2.bitwise_and(mask_red_border, ui_mask)
    # Dilater la bordure pour remplir l'intérieur
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

        # Succès — on utilise ce contour
        if strategy_name != "warm":
            print(f"   ⚠️  Overlay rouge faible, fallback {strategy_name}")

        hull = cv2.convexHull(largest)

        M = cv2.moments(hull)
        if M["m00"] > 0:
            center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        else:
            center = np.mean(hull.reshape(-1, 2).astype(float), axis=0)

        break  # On a trouvé, on arrête

    if hull is None:
        return None, None

    return hull, center


# =============================================================================
#                CALCUL DES POSITIONS DE DÉPLOIEMENT
# =============================================================================

def _sample_hull_point(hull_pts, frac):
    """Interpole un point le long du périmètre du hull à la fraction donnée."""
    n = len(hull_pts)
    idx_float = frac * n
    i1 = int(idx_float) % n
    i2 = (i1 + 1) % n
    t = idx_float - int(idx_float)
    return hull_pts[i1] * (1 - t) + hull_pts[i2] * t


def _angle_from_center(pt, center):
    """Calcule l'angle d'un point par rapport au centre (0=droite, anti-horaire)."""
    dx = pt[0] - center[0]
    dy = -(pt[1] - center[1])  # Y inversé (écran)
    return np.arctan2(dy, dx)


def _angle_diff(a, b):
    """Différence angulaire signée entre a et b, normalisée dans [-π, π]."""
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def _is_in_exclusion_zone(x, y, img_h, img_w):
    """Vérifie si un point (en coordonnées ADB) est dans une zone UI."""
    for ax1, ay1, ax2, ay2 in UI_EXCLUSION_ZONES:
        if ax1 <= x <= ax2 and ay1 <= y <= ay2:
            return True
    return False


def compute_deploy_positions(hull, center, img_shape, direction_idx,
                             spread=0.5, num_points=12, offset_px=None):
    """
    Calcule les positions de déploiement le long du bord du village.

    S'adapte automatiquement au niveau de zoom :
    - Dézoomé : positions offset en dehors du hull
    - Zoomé : positions sur le bord du hull ou le bord de l'écran

    Args:
        hull: convex hull (Nx1x2 array en coordonnées image)
        center: centre du village (x, y) en coordonnées image
        img_shape: (height, width) de l'image source
        direction_idx: 0-7 (N, NE, E, SE, S, SO, O, NO)
        spread: 0.0 (groupé au centre de la direction) à 1.0 (étalé sur tout le côté)
        num_points: nombre de positions à générer
        offset_px: distance en pixels ADB depuis le hull (défaut: auto)

    Returns:
        positions: liste de (x, y) en coordonnées ADB
    """
    img_h, img_w = img_shape[:2]
    scale_x = ADB_WIDTH / img_w
    scale_y = ADB_HEIGHT / img_h

    hull_pts = hull.reshape(-1, 2).astype(float)
    target_angle = DIRECTION_ANGLES[direction_idx]

    # --- Détection du zoom ---
    hull_area = cv2.contourArea(hull)
    game_area = img_h * img_w * 0.55
    zoom_ratio = hull_area / game_area  # 0.3 = dézoomé, 0.6+ = zoomé

    # Adapter les paramètres au zoom
    if offset_px is None:
        if zoom_ratio < 0.40:
            offset_px = DEPLOY_OFFSET        # 35px — normal
        elif zoom_ratio < 0.55:
            offset_px = 20                    # Moyen
        else:
            offset_px = 8                     # Très zoomé — quasi sur le hull

    margin = SCREEN_MARGIN
    if zoom_ratio > 0.50:
        margin = 30  # Réduire la marge écran quand zoomé

    dedup_dist_sq = 400 if zoom_ratio < 0.50 else 200  # 20px ou 14px

    # --- Échantillonner beaucoup de points le long du hull ---
    n_samples = 200
    hull_samples = []
    for i in range(n_samples):
        frac = i / n_samples
        pt = _sample_hull_point(hull_pts, frac)
        angle = _angle_from_center(pt, center)
        hull_samples.append((pt, angle, frac))

    # --- Trier par proximité angulaire à la direction cible ---
    hull_samples.sort(key=lambda x: abs(_angle_diff(x[1], target_angle)))

    # --- Sélectionner les points selon le spread ---
    # Quand zoomé, élargir le spread pour compenser les positions perdues
    effective_spread = spread
    if zoom_ratio > 0.50:
        effective_spread = min(1.0, spread + 0.3)

    max_angle_range = np.pi * (0.15 + 0.85 * effective_spread)

    # Filtrer les points dans l'arc angulaire
    candidates = []
    for pt, angle, frac in hull_samples:
        if abs(_angle_diff(angle, target_angle)) <= max_angle_range:
            candidates.append((pt, angle))

    if not candidates:
        candidates = [(pt, angle) for pt, angle, frac in hull_samples[:num_points * 3]]

    # --- Trier par angle pour un déploiement ordonné ---
    candidates.sort(key=lambda x: x[1])

    # --- Sous-échantillonner pour obtenir num_points positions ---
    # Demander plus que nécessaire car certains seront filtrés
    target_count = int(num_points * 1.5)
    if len(candidates) > target_count:
        step = len(candidates) / target_count
        selected = [candidates[int(i * step)] for i in range(target_count)]
    else:
        selected = candidates

    # --- Convertir en coordonnées ADB avec offset vers l'extérieur ---
    positions = []
    offset_img = offset_px / max(scale_x, scale_y)

    for pt, angle in selected:
        # Direction vers l'extérieur (depuis le centre)
        direction = pt - center
        norm = np.linalg.norm(direction)
        if norm < 1:
            continue
        direction = direction / norm

        # Point décalé vers l'extérieur
        deploy_pt = pt + direction * offset_img

        # Convertir en coordonnées ADB
        adb_x = int(deploy_pt[0] * scale_x)
        adb_y = int(deploy_pt[1] * scale_y)

        # Clamp aux limites de l'écran
        adb_x = max(margin, min(ADB_WIDTH - margin, adb_x))
        adb_y = max(margin, min(ADB_HEIGHT - margin, adb_y))

        # Vérifier les zones d'exclusion UI
        if _is_in_exclusion_zone(adb_x, adb_y, ADB_HEIGHT, ADB_WIDTH):
            continue

        positions.append((adb_x, adb_y))

    # Dédupliquer les positions trop proches
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

    # --- Garantir un minimum de positions ---
    # Si pas assez de positions (zoom extrême), ajouter des points
    # le long du bord de l'écran dans la direction demandée
    if len(positions) < 6:
        positions = _add_screen_edge_positions(
            positions, direction_idx, margin, num_points
        )

    return positions


def _add_screen_edge_positions(existing, direction_idx, margin, target_count):
    """
    Ajoute des positions le long du bord de l'écran quand le hull
    sort de l'écran (zoom fort). Ces positions sont valides car
    dans CoC, le bord visible de la carte est toujours déployable.
    """
    # Quel bord de l'écran est dans la direction demandée ?
    edge_positions = {
        0: [(x, margin) for x in range(200, 1720, 120)],           # N → haut
        1: [(x, margin + (1920 - x) // 3)                          # NE → coin haut-droit
            for x in range(800, 1860, 100)],
        2: [(1920 - margin, y) for y in range(100, 880, 80)],      # E → droite
        3: [(x, 880 - (1920 - x) // 3)                             # SE → coin bas-droit
            for x in range(800, 1860, 100)],
        4: [(x, 880) for x in range(200, 1720, 120)],              # S → bas
        5: [(x, 880 - x // 3)                                      # SO → coin bas-gauche
            for x in range(100, 1100, 100)],
        6: [(margin, y) for y in range(100, 880, 80)],             # O → gauche
        7: [(x, margin + x // 3)                                    # NO → coin haut-gauche
            for x in range(100, 1100, 100)],
    }

    edge_pts = edge_positions.get(direction_idx, [])

    # Filtrer les points UI
    edge_pts = [(x, y) for x, y in edge_pts
                if not _is_in_exclusion_zone(x, y, ADB_HEIGHT, ADB_WIDTH)]

    # Fusionner avec les positions existantes (éviter les doublons)
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
    Convertit le centre du village en coordonnées ADB.
    """
    img_h, img_w = img_shape[:2]
    adb_x = int(center[0] * ADB_WIDTH / img_w)
    adb_y = int(center[1] * ADB_HEIGHT / img_h)
    return (adb_x, adb_y)


def get_full_perimeter_positions(screenshot_pil, num_points=20, offset_px=None):
    """
    Génère des positions de déploiement réparties sur TOUT le périmètre
    du village (360°), pas juste un côté.

    C'est la fonction à utiliser pour la V2 où l'agent choisit librement
    parmi toutes les positions autour du village.

    Args:
        screenshot_pil: image PIL du screenshot
        num_points: nombre de positions à générer (réparties uniformément)
        offset_px: distance depuis la bordure (auto-adapté au zoom)

    Returns:
        positions: liste de (x, y) en coordonnées ADB, réparties sur 360°
        center_adb: (x, y) centre du village
        success: True si la détection a fonctionné
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv.shape[:2]

    hull, center = detect_village_boundary(img_cv)

    if hull is None or center is None:
        return None, (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50), False

    scale_x = ADB_WIDTH / img_w
    scale_y = ADB_HEIGHT / img_h

    # Adapter l'offset au zoom
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
    print(f"   ✅ Zone détectée : hull={len(hull)} pts, "
          f"zoom={zoom_ratio:.0%} ({zoom_label})")

    # Échantillonner uniformément sur tout le périmètre du hull
    hull_pts = hull.reshape(-1, 2).astype(float)
    offset_img = offset_px / max(scale_x, scale_y)

    positions = []
    for i in range(num_points * 3):  # Sur-échantillonner puis filtrer
        frac = i / (num_points * 3)
        pt = _sample_hull_point(hull_pts, frac)

        # Direction vers l'extérieur
        direction = pt - center
        norm = np.linalg.norm(direction)
        if norm < 1:
            continue
        direction = direction / norm

        # Offset vers l'extérieur
        deploy_pt = pt + direction * offset_img

        # Convertir en coordonnées ADB
        adb_x = int(deploy_pt[0] * scale_x)
        adb_y = int(deploy_pt[1] * scale_y)

        # Clamp
        adb_x = max(margin, min(ADB_WIDTH - margin, adb_x))
        adb_y = max(margin, min(ADB_HEIGHT - margin, adb_y))

        # Exclure les zones UI
        if _is_in_exclusion_zone(adb_x, adb_y, ADB_HEIGHT, ADB_WIDTH):
            continue

        positions.append((adb_x, adb_y))

    # Dédupliquer
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

    # Trier par angle (pour que pos 0 = Nord, pos 5 = Est, etc.)
    def angle_from_center(p):
        dx = p[0] - ADB_WIDTH / 2
        dy = -(p[1] - ADB_HEIGHT / 2)
        return np.arctan2(dy, dx)

    positions.sort(key=angle_from_center, reverse=True)

    # Sous-échantillonner au nombre demandé
    if len(positions) > num_points:
        step = len(positions) / num_points
        positions = [positions[int(i * step)] for i in range(num_points)]

    center_adb = get_village_center_adb(center, img_cv.shape)

    print(f"   📍 {len(positions)} positions (360° périmètre)")

    return positions, center_adb, True


# =============================================================================
#                    FONCTION PRINCIPALE
# =============================================================================

def get_smart_deploy_positions(screenshot_pil, direction_idx, spread=0.5,
                               num_points=12, offset_px=None):
    """
    Point d'entrée principal : détecte la zone de déploiement et retourne
    les positions optimales.

    Args:
        screenshot_pil: image PIL du screenshot (phase d'attaque)
        direction_idx: 0-7 (N, NE, E, SE, S, SO, O, NO)
        spread: 0.0 (groupé) à 1.0 (étalé)
        num_points: nombre de positions
        offset_px: distance depuis la bordure (défaut: DEPLOY_OFFSET)

    Returns:
        positions: liste de (x, y) en coordonnées ADB (1920×1080)
        center_adb: (x, y) centre du village en coordonnées ADB
        success: True si la détection a réussi
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    hull, center = detect_village_boundary(img_cv)

    if hull is None or center is None:
        print("   ❌ Détection de la zone de déploiement échouée")
        return _fallback_positions(direction_idx, spread, num_points), \
               (ADB_WIDTH // 2, ADB_HEIGHT // 2 - 50), False

    hull_area = cv2.contourArea(hull)
    n_hull_pts = len(hull.reshape(-1, 2))
    game_area = img_cv.shape[0] * img_cv.shape[1] * 0.55
    zoom_ratio = hull_area / game_area
    zoom_label = "dézoomé" if zoom_ratio < 0.40 else "moyen" if zoom_ratio < 0.55 else "zoomé"
    print(f"   ✅ Zone détectée : hull={n_hull_pts} pts, "
          f"zoom={zoom_ratio:.0%} ({zoom_label})")

    positions = compute_deploy_positions(
        hull, center, img_cv.shape,
        direction_idx, spread, num_points, offset_px
    )

    center_adb = get_village_center_adb(center, img_cv.shape)

    if len(positions) < 3:
        print(f"   ⚠️  Seulement {len(positions)} positions, fallback")
        return _fallback_positions(direction_idx, spread, num_points), \
               center_adb, False

    direction_label = DIRECTION_LABELS[direction_idx]
    print(f"   📍 {len(positions)} positions de déploiement ({direction_label}, "
          f"spread={spread:.1f})")

    return positions, center_adb, True


# =============================================================================
#                       FALLBACK
# =============================================================================

def _fallback_positions(direction_idx, spread=0.5, num_points=12):
    """
    Positions de déploiement par défaut (coordonnées fixes).
    Utilisé quand la détection de zone échoue.
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
#                     DEBUG / VISUALISATION
# =============================================================================

def debug_deploy_zone(screenshot_pil, direction_idx=0, spread=0.5,
                      save_path=None):
    """
    Génère une image de debug montrant la détection et les positions.

    Args:
        screenshot_pil: image PIL
        direction_idx: direction à visualiser
        spread: spread
        save_path: chemin de sauvegarde (optionnel)

    Returns:
        debug_img: image BGR avec les annotations
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

    # Dessiner le hull
    cv2.drawContours(debug, [hull], -1, (0, 255, 0), 3)

    # Dessiner le centre
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(debug, (cx, cy), 10, (0, 255, 255), -1)

    # Calculer et dessiner les positions
    positions = compute_deploy_positions(
        hull, center, img_cv.shape,
        direction_idx, spread, num_points=16
    )

    # Convertir les positions ADB en coordonnées image pour le dessin
    img_h, img_w = img_cv.shape[:2]
    for adb_x, adb_y in positions:
        ix = int(adb_x * img_w / ADB_WIDTH)
        iy = int(adb_y * img_h / ADB_HEIGHT)
        cv2.circle(debug, (ix, iy), 8, (255, 0, 255), -1)

    # Texte d'info
    direction_label = DIRECTION_LABELS[direction_idx]
    cv2.putText(debug, f"Dir: {direction_label}  Spread: {spread:.1f}  "
                f"Pts: {len(positions)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(save_path, debug)

    return debug


# =============================================================================
#                            TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test sur une image spécifique
        img_path = sys.argv[1]
        img_pil = Image.open(img_path).convert("RGB")

        print(f"🧪 Test deploy_zone sur {img_path}")
        print(f"   Image: {img_pil.size}")

        for direction in range(8):
            positions, center_adb, success = get_smart_deploy_positions(
                img_pil, direction, spread=0.5, num_points=12
            )
            label = DIRECTION_LABELS[direction]
            status = "✅" if success else "❌"
            print(f"   {status} {label}: {len(positions)} positions, "
                  f"center=({center_adb[0]},{center_adb[1]})")

        # Générer les images de debug
        for d in range(8):
            out = f"debug_deploy_{DIRECTION_LABELS[d]}.png"
            debug_deploy_zone(img_pil, d, 0.5, save_path=out)
            print(f"   💾 {out}")

    else:
        print("deploy_zone.py — Détection de la zone de déploiement")
        print()
        print("Usage :")
        print("  python deploy_zone.py <screenshot.png>")
        print()
        print("Dans le code :")
        print("  from clashai.perception.deploy_zone import get_smart_deploy_positions")
        print("  positions, center, ok = get_smart_deploy_positions(img, dir, spread)")