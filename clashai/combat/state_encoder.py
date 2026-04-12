# scripts/rl/state_encoder.py
# Transforme la sortie de YOLO+CNN en état exploitable par l'agent RL.
#
# V3 : 2 nouveaux canaux de grille (danger sol, danger air) + features enrichies
#
# Entrée : liste de bâtiments [{class, confidence, bbox, center}, ...]
# Sortie : grille (12, 40, 40) + vecteur features (20,)
#
# Canaux de grille (12) :
#   0  : défenses dangereuses (infernos, eagle, arcX, etc.)
#   1  : défenses moyennes (canons, tours archères, mortiers, etc.)
#   2  : défenses anti-aériennes
#   3  : défenses spéciales (teslas, tours runiques)
#   4  : ressources
#   5  : HDV
#   6  : château de clan
#   7  : hall des héros
#   8  : bâtiments non-défensifs
#   9  : densité totale
#   10 : danger sol (heatmap DPS sol approximatif)     ← NOUVEAU V3
#   11 : danger air (heatmap DPS air approximatif)     ← NOUVEAU V3
#
# Features village (20) :
#   0  : nombre de défenses dangereuses (normalisé)
#   1  : nombre de défenses moyennes (normalisé)
#   2  : nombre de défenses anti-aériennes (normalisé)
#   3  : nombre de ressources (normalisé)
#   4  : total de bâtiments (normalisé)
#   5-6: position HDV (x, y normalisées)
#   7  : HDV centré ou en bord
#   8  : nombre de tours d'enfer (normalisé)           ← NOUVEAU V3
#   9-10: position moyenne des infernos (x, y)          ← NOUVEAU V3
#   11-12: position aigle artilleur (x, y)              ← NOUVEAU V3
#   13-14: position château de clan (x, y)              ← NOUVEAU V3
#   15-18: score de faiblesse par quadrant (N, E, S, O) ← NOUVEAU V3
#   19 : nombre de scattershots (normalisé)             ← NOUVEAU V3

import math
import numpy as np


# =============================================================================
#                    CATÉGORIES DE BÂTIMENTS
# =============================================================================

CATEGORIES = {
    'defenses_dangereuses': [
        'tour_enfer_mono', 'tour_enfer_multiple', 'aigle_artilleur',
        'arcX_sol', 'arcX_sol_air', 'catapulte_erratique', 'monolithe',
        'tour_vengeuse', 'gigabombe', 'tour_runique_seisme'
    ],
    'defenses_moyennes': [
        'canon', 'tour_archere', 'mortier', 'multi_mortier', 'tour_sorcier',
        'tour_bombe', 'canon_double', 'canon_ricochet', 'cracheur_feu',
        'super_tour_sorcier', 'tour_archere_multiple', 'tour_archere_rapide',
        'tour_multi_equipe_rapide', 'tour_multi_equipe_lente',
        'cabane_ouvrier_arme'
    ],
    'defenses_anti_aeriennes': [
        'defense_antiaerienne', 'prop_air'
    ],
    'defenses_speciales': [
        'tesla', 'tour_runique_rage', 'tour_runique_poison',
        'tour_runique_invisible'
    ],
    'ressources': [
        'reserve_or', 'reserve_elixir', 'reserve_noire', 'ressources'
    ],
    'hdv': [
        'hdv'
    ],
    'chateau_clan': [
        'chateau_clan'
    ],
    'hall_heros': [
        'hall_heros'
    ],
    'batiments_non_defensifs': [
        'caserne', 'camps_militaires', 'laboratoire', 'atelier',
        'animalerie', 'forgeron', 'cabane_assistants', 'cabane_bob',
        'sort'
    ],
    # Canaux 10-11 (danger) calculés séparément
}

CHANNEL_NAMES = list(CATEGORIES.keys()) + [
    'densite_totale',
    'danger_sol',       # NOUVEAU V3
    'danger_air',       # NOUVEAU V3
]

# Mapping inversé : nom de classe → index du canal
CLASS_TO_CHANNEL = {}
for channel_idx, (category, classes) in enumerate(CATEGORIES.items()):
    for cls_name in classes:
        CLASS_TO_CHANNEL[cls_name] = channel_idx


# =============================================================================
#                    CONFIGURATION DE LA GRILLE
# =============================================================================

GRID_SIZE = 40
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
NUM_CHANNELS = len(CHANNEL_NAMES)  # 12

CELL_WIDTH = SCREEN_WIDTH / GRID_SIZE    # 48 px
CELL_HEIGHT = SCREEN_HEIGHT / GRID_SIZE  # 27 px

# Nombre de features village
NUM_VILLAGE_FEATURES = 20


# =============================================================================
#                    DPS & PORTÉE DES DÉFENSES (V3)
# =============================================================================

# Portée approximative en pixels ADB (HDV18 max level)
# et DPS normalisé (1.0 = défense la plus forte)
# 'ground' = cible les troupes au sol, 'air' = cible les troupes aériennes
# 'both' = cible les deux
DEFENSE_STATS = {
    # --- Défenses dangereuses ---
    'tour_enfer_mono':       {'range': 400, 'dps': 1.0, 'targets': 'both'},
    'tour_enfer_multiple':   {'range': 350, 'dps': 0.9, 'targets': 'both'},
    'aigle_artilleur':       {'range': 600, 'dps': 0.8, 'targets': 'both'},
    'arcX_sol':              {'range': 450, 'dps': 0.85, 'targets': 'ground'},
    'arcX_sol_air':          {'range': 450, 'dps': 0.85, 'targets': 'both'},
    'catapulte_erratique':   {'range': 500, 'dps': 0.7, 'targets': 'ground'},
    'monolithe':             {'range': 400, 'dps': 0.9, 'targets': 'ground'},
    'tour_vengeuse':         {'range': 350, 'dps': 0.6, 'targets': 'both'},
    'gigabombe':             {'range': 300, 'dps': 0.5, 'targets': 'ground'},
    'tour_runique_seisme':   {'range': 350, 'dps': 0.5, 'targets': 'ground'},
    # --- Défenses moyennes ---
    'canon':                 {'range': 300, 'dps': 0.3, 'targets': 'ground'},
    'tour_archere':          {'range': 350, 'dps': 0.35, 'targets': 'both'},
    'mortier':               {'range': 400, 'dps': 0.2, 'targets': 'ground'},
    'multi_mortier':         {'range': 350, 'dps': 0.4, 'targets': 'ground'},
    'tour_sorcier':          {'range': 350, 'dps': 0.4, 'targets': 'both'},
    'tour_bombe':            {'range': 300, 'dps': 0.35, 'targets': 'ground'},
    'canon_double':          {'range': 300, 'dps': 0.4, 'targets': 'ground'},
    'canon_ricochet':        {'range': 350, 'dps': 0.35, 'targets': 'ground'},
    'cracheur_feu':          {'range': 250, 'dps': 0.5, 'targets': 'ground'},
    'super_tour_sorcier':    {'range': 350, 'dps': 0.5, 'targets': 'both'},
    'tour_archere_multiple': {'range': 350, 'dps': 0.4, 'targets': 'both'},
    'tour_archere_rapide':   {'range': 350, 'dps': 0.45, 'targets': 'both'},
    'tour_multi_equipe_rapide': {'range': 350, 'dps': 0.4, 'targets': 'ground'},
    'tour_multi_equipe_lente':  {'range': 350, 'dps': 0.35, 'targets': 'ground'},
    'cabane_ouvrier_arme':   {'range': 250, 'dps': 0.2, 'targets': 'ground'},
    # --- Anti-aérien ---
    'defense_antiaerienne':  {'range': 350, 'dps': 0.6, 'targets': 'air'},
    'prop_air':              {'range': 300, 'dps': 0.3, 'targets': 'air'},
    # --- Défenses spéciales ---
    'tesla':                 {'range': 250, 'dps': 0.5, 'targets': 'both'},
    'tour_runique_rage':     {'range': 300, 'dps': 0.3, 'targets': 'both'},
    'tour_runique_poison':   {'range': 300, 'dps': 0.3, 'targets': 'both'},
    'tour_runique_invisible': {'range': 300, 'dps': 0.2, 'targets': 'both'},
}

# Classes spécifiques pour les features enrichies
INFERNO_CLASSES = ['tour_enfer_mono', 'tour_enfer_multiple']
EAGLE_CLASSES = ['aigle_artilleur']
SCATTER_CLASSES = ['catapulte_erratique']
CC_CLASSES = ['chateau_clan']


# =============================================================================
#                    FONCTIONS D'ENCODAGE
# =============================================================================

def buildings_to_grid(buildings):
    """
    Convertit une liste de bâtiments en grille 2D multi-canaux.

    Returns:
        grid: np.array (12, 40, 40)
    """
    grid = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Canaux 0-8 : catégories de bâtiments (identique V2)
    for b in buildings:
        cls_name = b['class']
        cx, cy = b['center']
        confidence = b['confidence']

        grid_x = max(0, min(GRID_SIZE - 1, int(cx / CELL_WIDTH)))
        grid_y = max(0, min(GRID_SIZE - 1, int(cy / CELL_HEIGHT)))

        if cls_name in CLASS_TO_CHANNEL:
            channel = CLASS_TO_CHANNEL[cls_name]
            grid[channel, grid_y, grid_x] += confidence

        # Canal 9 : densité totale
        grid[9, grid_y, grid_x] += confidence

    # Canaux 10-11 : danger heatmaps (NOUVEAU V3)
    _compute_danger_heatmap(grid, buildings)

    # Normaliser chaque canal entre 0 et 1
    for c in range(NUM_CHANNELS):
        max_val = grid[c].max()
        if max_val > 0:
            grid[c] /= max_val

    return grid


def _compute_danger_heatmap(grid, buildings):
    """
    Calcule les canaux de danger (sol et air).
    
    Pour chaque cellule de la grille, on additionne le DPS de toutes
    les défenses dont la portée couvre cette cellule.
    C'est une approximation — la portée réelle dépend du niveau de la
    défense et du zoom, mais ça donne une carte de chaleur utile.
    """
    ch_ground = 10  # Canal danger sol
    ch_air = 11     # Canal danger air

    for b in buildings:
        cls_name = b['class']
        if cls_name not in DEFENSE_STATS:
            continue

        stats = DEFENSE_STATS[cls_name]
        cx, cy = b['center']
        dps = stats['dps'] * b['confidence']
        targets = stats['targets']

        # Convertir la portée en cellules de grille
        range_cells_x = stats['range'] / CELL_WIDTH
        range_cells_y = stats['range'] / CELL_HEIGHT

        # Position de la défense sur la grille
        def_gx = cx / CELL_WIDTH
        def_gy = cy / CELL_HEIGHT

        # Scanner les cellules dans le carré englobant
        min_gx = max(0, int(def_gx - range_cells_x))
        max_gx = min(GRID_SIZE - 1, int(def_gx + range_cells_x))
        min_gy = max(0, int(def_gy - range_cells_y))
        max_gy = min(GRID_SIZE - 1, int(def_gy + range_cells_y))

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                # Distance en pixels (approximée)
                dist_px = math.sqrt(
                    ((gx - def_gx) * CELL_WIDTH) ** 2 +
                    ((gy - def_gy) * CELL_HEIGHT) ** 2
                )
                if dist_px <= stats['range']:
                    # DPS décroissant avec la distance (linéaire)
                    falloff = 1.0 - (dist_px / stats['range']) * 0.5
                    contribution = dps * falloff

                    if targets in ('ground', 'both'):
                        grid[ch_ground, gy, gx] += contribution
                    if targets in ('air', 'both'):
                        grid[ch_air, gy, gx] += contribution


def extract_features(buildings):
    """
    Extrait un vecteur de features enrichies du village.

    Returns:
        features: np.array (20,)
    """
    features = np.zeros(NUM_VILLAGE_FEATURES, dtype=np.float32)

    # Compteurs
    n_defenses_danger = 0
    n_defenses_medium = 0
    n_anti_aerien = 0
    n_ressources = 0
    total_buildings = len(buildings)

    # Positions spécifiques
    hdv_position = None
    inferno_positions = []
    eagle_position = None
    cc_position = None
    scatter_count = 0

    # Toutes les défenses (pour les quadrants)
    defense_classes = set()
    for cat in ['defenses_dangereuses', 'defenses_moyennes',
                'defenses_anti_aeriennes', 'defenses_speciales']:
        defense_classes.update(CATEGORIES[cat])

    defense_positions = []  # (x_norm, y_norm, weight)

    for b in buildings:
        cls_name = b['class']
        cx, cy = b['center']

        if cls_name in CATEGORIES['defenses_dangereuses']:
            n_defenses_danger += 1
        elif cls_name in CATEGORIES['defenses_moyennes']:
            n_defenses_medium += 1
        elif cls_name in CATEGORIES['defenses_anti_aeriennes']:
            n_anti_aerien += 1
        elif cls_name in CATEGORIES['ressources']:
            n_ressources += 1

        if cls_name == 'hdv':
            hdv_position = (cx, cy)
        if cls_name in INFERNO_CLASSES:
            inferno_positions.append((cx, cy))
        if cls_name in EAGLE_CLASSES:
            eagle_position = (cx, cy)
        if cls_name in CC_CLASSES:
            cc_position = (cx, cy)
        if cls_name in SCATTER_CLASSES:
            scatter_count += 1

        if cls_name in defense_classes:
            weight = 2.0 if cls_name in CATEGORIES['defenses_dangereuses'] else 1.0
            defense_positions.append((cx / SCREEN_WIDTH, cy / SCREEN_HEIGHT, weight))

    # --- Features 0-7 : identiques V2 ---

    features[0] = min(n_defenses_danger / 10.0, 1.0)
    features[1] = min(n_defenses_medium / 20.0, 1.0)
    features[2] = min(n_anti_aerien / 8.0, 1.0)
    features[3] = min(n_ressources / 15.0, 1.0)
    features[4] = min(total_buildings / 100.0, 1.0)

    if hdv_position is not None:
        features[5] = hdv_position[0] / SCREEN_WIDTH
        features[6] = hdv_position[1] / SCREEN_HEIGHT
    else:
        features[5] = 0.5
        features[6] = 0.5

    if hdv_position is not None:
        dx = abs(hdv_position[0] / SCREEN_WIDTH - 0.5)
        dy = abs(hdv_position[1] / SCREEN_HEIGHT - 0.5)
        features[7] = min((dx + dy) * 2, 1.0)
    else:
        features[7] = 0.0

    # --- Features 8-19 : NOUVEAU V3 ---

    # 8 : Nombre de tours d'enfer
    features[8] = min(len(inferno_positions) / 4.0, 1.0)

    # 9-10 : Position moyenne des infernos
    if inferno_positions:
        mean_x = np.mean([p[0] for p in inferno_positions])
        mean_y = np.mean([p[1] for p in inferno_positions])
        features[9] = mean_x / SCREEN_WIDTH
        features[10] = mean_y / SCREEN_HEIGHT
    else:
        features[9] = 0.5
        features[10] = 0.5

    # 11-12 : Position aigle artilleur
    if eagle_position is not None:
        features[11] = eagle_position[0] / SCREEN_WIDTH
        features[12] = eagle_position[1] / SCREEN_HEIGHT
    else:
        features[11] = 0.5
        features[12] = 0.5

    # 13-14 : Position château de clan
    if cc_position is not None:
        features[13] = cc_position[0] / SCREEN_WIDTH
        features[14] = cc_position[1] / SCREEN_HEIGHT
    else:
        features[13] = 0.5
        features[14] = 0.5

    # 15-18 : Score de faiblesse par quadrant (N, E, S, O)
    # Plus le score est BAS, plus le quadrant est faible → meilleur côté d'attaque
    # On calcule le DPS total pondéré dans chaque quadrant
    quadrant_scores = [0.0, 0.0, 0.0, 0.0]  # N, E, S, O
    for dx_norm, dy_norm, weight in defense_positions:
        # Centrer sur le milieu de l'écran
        rx = dx_norm - 0.5  # -0.5 à 0.5
        ry = dy_norm - 0.5

        # Contribution à chaque quadrant (inversée par distance au bord)
        # Nord = haut (ry négatif), Sud = bas (ry positif)
        # Est = droite (rx positif), Ouest = gauche (rx négatif)
        quadrant_scores[0] += weight * max(0, 0.5 - ry)  # N : plus fort si défense en haut
        quadrant_scores[1] += weight * max(0, 0.5 + rx)  # E : plus fort si défense à droite
        quadrant_scores[2] += weight * max(0, 0.5 + ry)  # S : plus fort si défense en bas
        quadrant_scores[3] += weight * max(0, 0.5 - rx)  # O : plus fort si défense à gauche

    # Normaliser par le max
    max_q = max(quadrant_scores) if max(quadrant_scores) > 0 else 1.0
    features[15] = quadrant_scores[0] / max_q
    features[16] = quadrant_scores[1] / max_q
    features[17] = quadrant_scores[2] / max_q
    features[18] = quadrant_scores[3] / max_q

    # 19 : Nombre de scattershots
    features[19] = min(scatter_count / 3.0, 1.0)

    return features


def encode_state(buildings):
    """
    Encode complet de l'état : grille + features.

    Returns:
        state: dict avec 'grid' (12, 40, 40) et 'features' (20,)
    """
    grid = buildings_to_grid(buildings)
    features = extract_features(buildings)

    return {
        'grid': grid,
        'features': features,
    }


def get_attack_direction_coords(direction_idx, spread=0.5):
    """
    Convertit un index de direction (0-7) en coordonnées de déploiement.
    """
    directions = {
        0: (SCREEN_WIDTH // 2, 50),
        1: (SCREEN_WIDTH - 100, 50),
        2: (SCREEN_WIDTH - 100, SCREEN_HEIGHT // 2),
        3: (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 150),
        4: (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150),
        5: (100, SCREEN_HEIGHT - 150),
        6: (100, SCREEN_HEIGHT // 2),
        7: (100, 50),
    }

    center_x, center_y = directions[direction_idx]
    positions = []
    num_points = 10
    max_spread_px = 300
    spread_px = int(spread * max_spread_px)

    for i in range(num_points):
        offset = (i - num_points // 2) * (spread_px // num_points)
        if direction_idx in (0, 4):
            x, y = center_x + offset, center_y
        elif direction_idx in (2, 6):
            x, y = center_x, center_y + offset
        else:
            x = center_x + offset
            y = center_y + (offset if direction_idx in (3, 5) else -offset)
        x = max(50, min(SCREEN_WIDTH - 50, x))
        y = max(50, min(SCREEN_HEIGHT - 50, y))
        positions.append((int(x), int(y)))

    return positions


# =============================================================================
#                         UTILITAIRES
# =============================================================================

def find_best_attack_side(buildings, verbose=False):
    """
    Heuristique V3.1 : choix intelligent du côté d'attaque.
    
    Combine 3 critères au lieu d'un seul :
    
    1. Faiblesse défensive (DPS total dans ce secteur) — poids 1.0
       → Attaquer là où il y a moins de défenses
    
    2. Proximité du TH — poids 0.7
       → Les troupes doivent atteindre le TH pour les étoiles
    
    3. Accessibilité des infernos — poids 0.5
       → Arriver tôt sur les infernos permet de les geler/détruire
       → Avec GoWitch, les golems absorbent pendant qu'on freeze
    
    4. Distance à l'eagle — poids 0.3
       → L'eagle a une zone morte (ne tire pas au contact)
       → Arriver vite sur l'eagle réduit ses dégâts
    
    Returns:
        best_direction: int 0-7 (index du meilleur secteur d'attaque)
    """
    # Points de déploiement normalisés (0-1) pour chaque direction
    DEPLOY_POINTS = {
        0: (0.50, 0.05),   # N  (haut centre)
        1: (0.95, 0.05),   # NE (haut droite)
        2: (0.95, 0.50),   # E  (droite centre)
        3: (0.95, 0.90),   # SE (bas droite)
        4: (0.50, 0.90),   # S  (bas centre)
        5: (0.05, 0.90),   # SW (bas gauche)
        6: (0.05, 0.50),   # W  (gauche centre)
        7: (0.05, 0.05),   # NW (haut gauche)
    }
    
    DIRECTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    MAX_DIST = math.sqrt(2.0)  # Distance max normalisée (coin à coin)
    
    # Poids des critères
    W_DEFENSE = 1.0    # Faiblesse défensive
    W_TH = 0.7         # Proximité du TH
    W_INFERNO = 0.5    # Accessibilité des infernos
    W_EAGLE = 0.3      # Accessibilité de l'eagle
    
    # --- 1. Score DPS par secteur (logique existante) ---
    sector_dps = [0.0] * 8
    for b in buildings:
        cls_name = b['class']
        if cls_name not in DEFENSE_STATS:
            continue
        cx, cy = b['center']
        dps = DEFENSE_STATS[cls_name]['dps']
        nx = (cx / SCREEN_WIDTH) - 0.5
        ny = (cy / SCREEN_HEIGHT) - 0.5
        angle = math.atan2(ny, nx)
        angle_deg = math.degrees(angle) % 360
        sector_angle = (angle_deg + 90) % 360
        sector_idx = int(sector_angle / 45) % 8
        sector_dps[sector_idx] += dps
    
    max_dps = max(sector_dps) if max(sector_dps) > 0 else 1.0
    
    # --- 2. Positions clés ---
    hdv_pos = None
    inferno_positions = []
    eagle_pos = None
    
    for b in buildings:
        cls = b['class']
        cx_norm = b['center'][0] / SCREEN_WIDTH
        cy_norm = b['center'][1] / SCREEN_HEIGHT
        
        if cls == 'hdv':
            hdv_pos = (cx_norm, cy_norm)
        elif cls in INFERNO_CLASSES:
            inferno_positions.append((cx_norm, cy_norm))
        elif cls in EAGLE_CLASSES:
            eagle_pos = (cx_norm, cy_norm)
    
    # --- 3. Scorer chaque direction ---
    scores = []
    for d in range(8):
        deploy_x, deploy_y = DEPLOY_POINTS[d]
        
        # Critère 1 : faiblesse défensive (inversé : faible DPS = bon)
        defense_score = 1.0 - (sector_dps[d] / max_dps)
        
        # Critère 2 : proximité TH
        if hdv_pos is not None:
            th_dist = math.sqrt(
                (deploy_x - hdv_pos[0])**2 + (deploy_y - hdv_pos[1])**2
            )
            th_score = 1.0 - (th_dist / MAX_DIST)
        else:
            th_score = 0.5  # Pas de TH détecté → neutre
        
        # Critère 3 : accessibilité des infernos
        #   On prend la distance au plus PROCHE inferno
        #   (l'idée : les golems arrivent dessus → freeze)
        if inferno_positions:
            min_inf_dist = min(
                math.sqrt((deploy_x - ix)**2 + (deploy_y - iy)**2)
                for ix, iy in inferno_positions
            )
            inferno_score = 1.0 - (min_inf_dist / MAX_DIST)
        else:
            inferno_score = 0.5  # Pas d'infernos → neutre
        
        # Critère 4 : accessibilité eagle
        if eagle_pos is not None:
            eagle_dist = math.sqrt(
                (deploy_x - eagle_pos[0])**2 + (deploy_y - eagle_pos[1])**2
            )
            eagle_score = 1.0 - (eagle_dist / MAX_DIST)
        else:
            eagle_score = 0.5
        
        # Score composite
        total = (W_DEFENSE * defense_score 
                 + W_TH * th_score
                 + W_INFERNO * inferno_score
                 + W_EAGLE * eagle_score)
        
        scores.append({
            'direction': d,
            'total': total,
            'defense': defense_score,
            'th': th_score,
            'inferno': inferno_score,
            'eagle': eagle_score,
        })
    
    # Trier par score total décroissant
    scores.sort(key=lambda s: s['total'], reverse=True)
    best = scores[0]
    
    if verbose:
        print("\n   🎯 Analyse des côtés d'attaque :")
        for s in scores:
            marker = " ← BEST" if s['direction'] == best['direction'] else ""
            print(f"      {DIRECTION_NAMES[s['direction']]:2s}: "
                  f"total={s['total']:.2f} "
                  f"(déf={s['defense']:.2f} th={s['th']:.2f} "
                  f"inf={s['inferno']:.2f} eag={s['eagle']:.2f}){marker}")
    
    return best['direction']


def print_state_summary(state):
    """Affiche un résumé de l'état encodé."""
    grid = state['grid']
    features = state['features']

    print("\n📊 État encodé V3 :")
    print(f"   Grille : {grid.shape}")

    for i, name in enumerate(CHANNEL_NAMES):
        active_cells = (grid[i] > 0).sum()
        if active_cells > 0:
            print(f"   Canal {i:2d} ({name}): {active_cells} cellules actives")

    labels = [
        'Déf. dangereuses', 'Déf. moyennes', 'Anti-aérien', 'Ressources',
        'Total bâtiments', 'HDV x', 'HDV y', 'HDV bord',
        'Infernos', 'Inferno moy. x', 'Inferno moy. y',
        'Aigle x', 'Aigle y', 'CC x', 'CC y',
        'Quadrant N', 'Quadrant E', 'Quadrant S', 'Quadrant O',
        'Scattershots'
    ]
    print(f"\n   Features ({len(features)} dims) :")
    for i, (label, val) in enumerate(zip(labels, features)):
        marker = " ←NEW" if i >= 8 else ""
        print(f"   [{i:2d}] {label:18s} = {val:.3f}{marker}")


# =============================================================================
#                            TEST
# =============================================================================

if __name__ == "__main__":
    fake_buildings = [
        {'class': 'hdv', 'confidence': 1.0, 'bbox': (900, 500, 980, 580), 'center': (940, 540)},
        {'class': 'tour_enfer_mono', 'confidence': 0.98, 'bbox': (800, 400, 850, 450), 'center': (825, 425)},
        {'class': 'tour_enfer_multiple', 'confidence': 0.95, 'bbox': (1000, 400, 1050, 450), 'center': (1025, 425)},
        {'class': 'aigle_artilleur', 'confidence': 0.97, 'bbox': (900, 300, 950, 350), 'center': (925, 325)},
        {'class': 'catapulte_erratique', 'confidence': 0.94, 'bbox': (750, 350, 800, 400), 'center': (775, 375)},
        {'class': 'canon', 'confidence': 0.99, 'bbox': (600, 300, 650, 350), 'center': (625, 325)},
        {'class': 'canon', 'confidence': 0.97, 'bbox': (1200, 300, 1250, 350), 'center': (1225, 325)},
        {'class': 'tour_archere', 'confidence': 0.98, 'bbox': (400, 500, 450, 550), 'center': (425, 525)},
        {'class': 'tour_archere', 'confidence': 0.96, 'bbox': (1400, 500, 1450, 550), 'center': (1425, 525)},
        {'class': 'defense_antiaerienne', 'confidence': 0.94, 'bbox': (700, 600, 750, 650), 'center': (725, 625)},
        {'class': 'defense_antiaerienne', 'confidence': 0.93, 'bbox': (1100, 600, 1150, 650), 'center': (1125, 625)},
        {'class': 'reserve_or', 'confidence': 0.99, 'bbox': (300, 200, 370, 270), 'center': (335, 235)},
        {'class': 'reserve_elixir', 'confidence': 0.98, 'bbox': (1500, 200, 1570, 270), 'center': (1535, 235)},
        {'class': 'chateau_clan', 'confidence': 0.96, 'bbox': (850, 350, 920, 420), 'center': (885, 385)},
        {'class': 'tesla', 'confidence': 0.90, 'bbox': (950, 600, 990, 640), 'center': (970, 620)},
    ]

    state = encode_state(fake_buildings)
    print_state_summary(state)

    best_side = find_best_attack_side(fake_buildings, verbose=True)
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    print(f"\n🎯 Meilleur côté d'attaque : {directions[best_side]}")

    # Vérifier les dimensions
    print("\n📐 Dimensions :")
    print(f"   Grid    : {state['grid'].shape} (attendu: ({NUM_CHANNELS}, {GRID_SIZE}, {GRID_SIZE}))")
    print(f"   Features: {state['features'].shape} (attendu: ({NUM_VILLAGE_FEATURES},))")

    # Vérifier la danger map
    ground = state['grid'][10]
    air = state['grid'][11]
    print("\n🔥 Danger map :")
    print(f"   Sol : {(ground > 0).sum()} cellules couvertes, max={ground.max():.3f}")
    print(f"   Air : {(air > 0).sum()} cellules couvertes, max={air.max():.3f}")