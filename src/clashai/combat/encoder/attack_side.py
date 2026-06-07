# clashai/combat/encoder/attack_side.py
# Pick the weakest side to attack + human-readable state summary.

import math
import numpy as np

from clashai.config import SCREEN_WIDTH, SCREEN_HEIGHT
from clashai.combat.encoder.constants import (
    CHANNEL_NAMES, NUM_CHANNELS, NUM_VILLAGE_FEATURES,
    DEFENSE_STATS, INFERNO_CLASSES, EAGLE_CLASSES,
)
from clashai.combat.encoder.features import encode_state
from clashai.config import GRID_SIZE


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
