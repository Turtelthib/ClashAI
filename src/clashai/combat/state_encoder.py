# scripts/rl/state_encoder.py
# Transforms YOLO+CNN output into a state usable by the RL agent.
#
# V3: 2 new grid channels (ground danger, air danger) + enriched features
#
# Input: list of buildings [{class, confidence, bbox, center}, ...]
# Output: grid (12, 40, 40) + feature vector (20,)
#
# Grid channels (12):
# 0: dangerous defenses (infernos, eagle, arcX, etc.)
# 1: medium defenses (cannons, archer towers, mortars, etc.)
# 2: anti-air defenses
# 3: special defenses (teslas, runic towers)
# 4: resources
# 5: TH (Town Hall)
# 6: clan castle
# 7: hero hall
# 8: non-defensive buildings
# 9: total density
# 10: ground danger (approximate ground DPS heatmap) ← NEW V3
# 11: air danger (approximate air DPS heatmap) ← NEW V3
#
# Village features (20):
# 0: number of dangerous defenses (normalized)
# 1: number of medium defenses (normalized)
# 2: number of anti-air defenses (normalized)
# 3: number of resources (normalized)
# 4: total buildings (normalized)
# 5-6: TH position (x, y normalized)
# 7: TH centered or on edge
# 8: number of inferno towers (normalized) ← NEW V3
# 9-10: average inferno position (x, y) ← NEW V3
# 11-12: eagle artillery position (x, y) ← NEW V3
# 13-14: clan castle position (x, y) ← NEW V3
# 15-18: weakness score per quadrant (N, E, S, W) ← NEW V3
# 19: number of scattershots (normalized) ← NEW V3

import math
import numpy as np


# =============================================================================
# CATÉGORIES DE BÂTIMENTS
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
    # Channels 10-11 (danger) computed separately
}

CHANNEL_NAMES = list(CATEGORIES.keys()) + [
    'densite_totale',
    'danger_sol',
    'danger_air',
]

# Reverse mapping: class name → channel index
CLASS_TO_CHANNEL = {}
for channel_idx, (category, classes) in enumerate(CATEGORIES.items()):
    for cls_name in classes:
        CLASS_TO_CHANNEL[cls_name] = channel_idx


# =============================================================================
# GRID CONFIGURATION
# =============================================================================

# Re-imported from clashai/config/ (Phase A).
from clashai.config import GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT  # noqa: E402

NUM_CHANNELS = len(CHANNEL_NAMES)

CELL_WIDTH = SCREEN_WIDTH / GRID_SIZE
CELL_HEIGHT = SCREEN_HEIGHT / GRID_SIZE

# Number of village features
NUM_VILLAGE_FEATURES = 20


# =============================================================================
# DPS & RANGE OF DEFENSES (V3)
# =============================================================================

# Approximate range in ADB pixels (TH18 max level)
# and normalised DPS (1.0 = strongest defense)
# 'ground' = targets ground troops, 'air' = targets air troops
# 'both' = targets both
DEFENSE_STATS = {
    # --- Dangerous defenses ---
    'tour_enfer_mono': {'range': 400, 'dps': 1.0, 'targets': 'both'},
    'tour_enfer_multiple': {'range': 350, 'dps': 0.9, 'targets': 'both'},
    'aigle_artilleur': {'range': 600, 'dps': 0.8, 'targets': 'both'},
    'arcX_sol': {'range': 450, 'dps': 0.85, 'targets': 'ground'},
    'arcX_sol_air': {'range': 450, 'dps': 0.85, 'targets': 'both'},
    'catapulte_erratique': {'range': 500, 'dps': 0.7, 'targets': 'ground'},
    'monolithe': {'range': 400, 'dps': 0.9, 'targets': 'ground'},
    'tour_vengeuse': {'range': 350, 'dps': 0.6, 'targets': 'both'},
    'gigabombe': {'range': 300, 'dps': 0.5, 'targets': 'ground'},
    'tour_runique_seisme': {'range': 350, 'dps': 0.5, 'targets': 'ground'},
    # --- Medium defenses ---
    'canon': {'range': 300, 'dps': 0.3, 'targets': 'ground'},
    'tour_archere': {'range': 350, 'dps': 0.35, 'targets': 'both'},
    'mortier': {'range': 400, 'dps': 0.2, 'targets': 'ground'},
    'multi_mortier': {'range': 350, 'dps': 0.4, 'targets': 'ground'},
    'tour_sorcier': {'range': 350, 'dps': 0.4, 'targets': 'both'},
    'tour_bombe': {'range': 300, 'dps': 0.35, 'targets': 'ground'},
    'canon_double': {'range': 300, 'dps': 0.4, 'targets': 'ground'},
    'canon_ricochet': {'range': 350, 'dps': 0.35, 'targets': 'ground'},
    'cracheur_feu': {'range': 250, 'dps': 0.5, 'targets': 'ground'},
    'super_tour_sorcier': {'range': 350, 'dps': 0.5, 'targets': 'both'},
    'tour_archere_multiple': {'range': 350, 'dps': 0.4, 'targets': 'both'},
    'tour_archere_rapide': {'range': 350, 'dps': 0.45, 'targets': 'both'},
    'tour_multi_equipe_rapide': {'range': 350, 'dps': 0.4, 'targets': 'ground'},
    'tour_multi_equipe_lente': {'range': 350, 'dps': 0.35, 'targets': 'ground'},
    'cabane_ouvrier_arme': {'range': 250, 'dps': 0.2, 'targets': 'ground'},
    # --- Anti-air ---
    'defense_antiaerienne': {'range': 350, 'dps': 0.6, 'targets': 'air'},
    'prop_air': {'range': 300, 'dps': 0.3, 'targets': 'air'},
    # --- Special defenses ---
    'tesla': {'range': 250, 'dps': 0.5, 'targets': 'both'},
    'tour_runique_rage': {'range': 300, 'dps': 0.3, 'targets': 'both'},
    'tour_runique_poison': {'range': 300, 'dps': 0.3, 'targets': 'both'},
    'tour_runique_invisible': {'range': 300, 'dps': 0.2, 'targets': 'both'},
}

# Specific classes for enriched features
INFERNO_CLASSES = ['tour_enfer_mono', 'tour_enfer_multiple']
EAGLE_CLASSES = ['aigle_artilleur']
SCATTER_CLASSES = ['catapulte_erratique']
CC_CLASSES = ['chateau_clan']


# =============================================================================
# FONCTIONS D'ENCODAGE
# =============================================================================

def buildings_to_grid(buildings):
    """
    Convertit une liste de bâtiments en grille 2D multi-canaux.

    Returns:
        grid: np.array (12, 40, 40)
    """
    grid = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Channels 0-8: building categories (same as V2)
    for b in buildings:
        cls_name = b['class']
        cx, cy = b['center']
        confidence = b['confidence']

        grid_x = max(0, min(GRID_SIZE - 1, int(cx / CELL_WIDTH)))
        grid_y = max(0, min(GRID_SIZE - 1, int(cy / CELL_HEIGHT)))

        if cls_name in CLASS_TO_CHANNEL:
            channel = CLASS_TO_CHANNEL[cls_name]
            grid[channel, grid_y, grid_x] += confidence

        # Channel 9: total density
        grid[9, grid_y, grid_x] += confidence

    # Channels 10-11: danger heatmaps (NEW V3)
    _compute_danger_heatmap(grid, buildings)

    # Normalise each channel between 0 and 1
    for c in range(NUM_CHANNELS):
        max_val = grid[c].max()
        if max_val > 0:
            grid[c] /= max_val

    return grid


def _compute_danger_heatmap(grid, buildings):
    """
    Computes the danger channels (ground and air).

    For each grid cell, the DPS of all defenses whose range covers
    that cell is accumulated.
    This is an approximation — actual range depends on defense level
    and zoom, but it produces a useful heatmap.
    """
    ch_ground = 10
    ch_air = 11

    for b in buildings:
        cls_name = b['class']
        if cls_name not in DEFENSE_STATS:
            continue

        stats = DEFENSE_STATS[cls_name]
        cx, cy = b['center']
        dps = stats['dps'] * b['confidence']
        targets = stats['targets']

        # Convert range to grid cells
        range_cells_x = stats['range'] / CELL_WIDTH
        range_cells_y = stats['range'] / CELL_HEIGHT

        # Defense position on the grid
        def_gx = cx / CELL_WIDTH
        def_gy = cy / CELL_HEIGHT

        # Scan cells within the bounding square
        min_gx = max(0, int(def_gx - range_cells_x))
        max_gx = min(GRID_SIZE - 1, int(def_gx + range_cells_x))
        min_gy = max(0, int(def_gy - range_cells_y))
        max_gy = min(GRID_SIZE - 1, int(def_gy + range_cells_y))

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                # Distance in pixels (approximated)
                dist_px = math.sqrt(
                    ((gx - def_gx) * CELL_WIDTH) ** 2 +
                    ((gy - def_gy) * CELL_HEIGHT) ** 2
                )
                if dist_px <= stats['range']:
                    # DPS decaying with distance (linear)
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

    # Counters
    n_defenses_danger = 0
    n_defenses_medium = 0
    n_anti_aerien = 0
    n_ressources = 0
    total_buildings = len(buildings)

    # Specific positions
    hdv_position = None
    inferno_positions = []
    eagle_position = None
    cc_position = None
    scatter_count = 0

    # All defenses (for quadrant scoring)
    defense_classes = set()
    for cat in ['defenses_dangereuses', 'defenses_moyennes',
                'defenses_anti_aeriennes', 'defenses_speciales']:
        defense_classes.update(CATEGORIES[cat])

    defense_positions = []

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

    # --- Features 0-7: identical to V2 ---

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

    # --- Features 8-19: NEW V3 ---

    # 8: Number of inferno towers
    features[8] = min(len(inferno_positions) / 4.0, 1.0)

    # 9-10: Average inferno position
    if inferno_positions:
        mean_x = np.mean([p[0] for p in inferno_positions])
        mean_y = np.mean([p[1] for p in inferno_positions])
        features[9] = mean_x / SCREEN_WIDTH
        features[10] = mean_y / SCREEN_HEIGHT
    else:
        features[9] = 0.5
        features[10] = 0.5

    # 11-12: Eagle artillery position
    if eagle_position is not None:
        features[11] = eagle_position[0] / SCREEN_WIDTH
        features[12] = eagle_position[1] / SCREEN_HEIGHT
    else:
        features[11] = 0.5
        features[12] = 0.5

    # 13-14: Clan castle position
    if cc_position is not None:
        features[13] = cc_position[0] / SCREEN_WIDTH
        features[14] = cc_position[1] / SCREEN_HEIGHT
    else:
        features[13] = 0.5
        features[14] = 0.5

    # 15-18: Weakness score per quadrant (N, E, S, W)
    # Lower score = weaker quadrant → better attack side
    # Compute total weighted DPS per quadrant
    quadrant_scores = [0.0, 0.0, 0.0, 0.0]
    for dx_norm, dy_norm, weight in defense_positions:
        # Centre on the screen midpoint
        rx = dx_norm - 0.5
        ry = dy_norm - 0.5

        # Contribution to each quadrant (inversely weighted by edge distance)
        # North = top (ry negative), South = bottom (ry positive)
        # East = right (rx positive), West = left (rx negative)
        quadrant_scores[0] += weight * max(0, 0.5 - ry)
        quadrant_scores[1] += weight * max(0, 0.5 + rx)
        quadrant_scores[2] += weight * max(0, 0.5 + ry)
        quadrant_scores[3] += weight * max(0, 0.5 - rx)

    # Normalise by the max
    max_q = max(quadrant_scores) if max(quadrant_scores) > 0 else 1.0
    features[15] = quadrant_scores[0] / max_q
    features[16] = quadrant_scores[1] / max_q
    features[17] = quadrant_scores[2] / max_q
    features[18] = quadrant_scores[3] / max_q

    # 19: Number of scattershots
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
# UTILITAIRES
# =============================================================================

def find_best_attack_side(buildings, verbose=False):
    """
    Heuristic V3.1: smart attack-side selection.

    Combines 3 criteria instead of one:

    1. Defensive weakness (total DPS in that sector) — weight 1.0
       → Attack where there are fewer defenses

    2. TH proximity — weight 0.7
       → Troops must reach the TH for stars

    3. Inferno accessibility — weight 0.5
       → Reaching infernos early allows freezing/destroying them
       → With GoWitch, golems absorb while we freeze

    4. Eagle distance — weight 0.3
       → The eagle has a dead zone (doesn't fire at close range)
       → Reaching the eagle quickly reduces its damage

    Returns:
        best_direction: int 0-7 (index of the best attack sector)
    """
    # Normalised deploy points (0-1) for each direction
    DEPLOY_POINTS = {
        0: (0.50, 0.05),
        1: (0.95, 0.05),
        2: (0.95, 0.50),
        3: (0.95, 0.90),
        4: (0.50, 0.90),
        5: (0.05, 0.90),
        6: (0.05, 0.50),
        7: (0.05, 0.05),
    }
    
    DIRECTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    MAX_DIST = math.sqrt(2.0)
    
    # Criteria weights
    W_DEFENSE = 1.0
    W_TH = 0.7
    W_INFERNO = 0.5
    W_EAGLE = 0.3
    
    # --- 1. DPS score per sector (existing logic) ---
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
    
    # --- 2. Key positions ---
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
    
    # --- 3. Score each direction ---
    scores = []
    for d in range(8):
        deploy_x, deploy_y = DEPLOY_POINTS[d]
        
        # Criterion 1: defensive weakness (inverted: low DPS = good)
        defense_score = 1.0 - (sector_dps[d] / max_dps)
        
        # Criterion 2: TH proximity
        if hdv_pos is not None:
            th_dist = math.sqrt(
                (deploy_x - hdv_pos[0])**2 + (deploy_y - hdv_pos[1])**2
            )
            th_score = 1.0 - (th_dist / MAX_DIST)
        else:
            th_score = 0.5
        
        # Criterion 3: inferno accessibility
        # Take the distance to the CLOSEST inferno
        # (idea: golems reach it → freeze)
        if inferno_positions:
            min_inf_dist = min(
                math.sqrt((deploy_x - ix)**2 + (deploy_y - iy)**2)
                for ix, iy in inferno_positions
            )
            inferno_score = 1.0 - (min_inf_dist / MAX_DIST)
        else:
            inferno_score = 0.5
        
        # Criterion 4: eagle accessibility
        if eagle_pos is not None:
            eagle_dist = math.sqrt(
                (deploy_x - eagle_pos[0])**2 + (deploy_y - eagle_pos[1])**2
            )
            eagle_score = 1.0 - (eagle_dist / MAX_DIST)
        else:
            eagle_score = 0.5
        
        # Composite score
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
    
    # Sort by descending total score
    scores.sort(key=lambda s: s['total'], reverse=True)
    best = scores[0]
    
    if verbose:
        print("\n Attack side analysis:")
        for s in scores:
            marker = " ← BEST" if s['direction'] == best['direction'] else ""
            print(f" {DIRECTION_NAMES[s['direction']]:2s}: "
                  f"total={s['total']:.2f} "
                  f"(def={s['defense']:.2f} th={s['th']:.2f} "
                  f"inf={s['inferno']:.2f} eag={s['eagle']:.2f}){marker}")
    
    return best['direction']


def print_state_summary(state):
    """Prints a summary of the encoded state."""
    grid = state['grid']
    features = state['features']

    print("\nEncoded state V3:")
    print(f" Grille : {grid.shape}")

    for i, name in enumerate(CHANNEL_NAMES):
        active_cells = (grid[i] > 0).sum()
        if active_cells > 0:
            print(f" Canal {i:2d} ({name}): {active_cells} cellules actives")

    labels = [
        'Dangerous def.', 'Medium def.', 'Anti-air', 'Resources',
        'Total buildings', 'TH x', 'TH y', 'TH edge',
        'Infernos', 'Avg inferno x', 'Avg inferno y',
        'Eagle x', 'Eagle y', 'CC x', 'CC y',
        'Quadrant N', 'Quadrant E', 'Quadrant S', 'Quadrant W',
        'Scattershots'
    ]
    print(f"\n Features ({len(features)} dims):")
    for i, (label, val) in enumerate(zip(labels, features)):
        marker = " ←NEW" if i >= 8 else ""
        print(f" [{i:2d}] {label:18s} = {val:.3f}{marker}")


# =============================================================================
# TEST
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
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    print(f"\nBest attack side: {directions[best_side]}")

    # Check dimensions
    print("\n Dimensions:")
    print(f" Grid: {state['grid'].shape} (expected: ({NUM_CHANNELS}, {GRID_SIZE}, {GRID_SIZE}))")
    print(f" Features: {state['features'].shape} (expected: ({NUM_VILLAGE_FEATURES},))")

    # Check danger map
    ground = state['grid'][10]
    air = state['grid'][11]
    print("\n Danger map:")
    print(f" Ground: {(ground > 0).sum()} cells covered, max={ground.max():.3f}")
    print(f" Air: {(air > 0).sum()} cells covered, max={air.max():.3f}")