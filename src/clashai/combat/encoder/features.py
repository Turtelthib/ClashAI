# clashai/combat/encoder/features.py
# Village feature vector + encode_state (grid + features bundle).

import math
import numpy as np

from clashai.config import SCREEN_WIDTH, SCREEN_HEIGHT
from clashai.combat.encoder.constants import (
    CATEGORIES, NUM_VILLAGE_FEATURES,
    INFERNO_CLASSES, EAGLE_CLASSES, SCATTER_CLASSES, CC_CLASSES,
)
from clashai.combat.encoder.grid import buildings_to_grid


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

