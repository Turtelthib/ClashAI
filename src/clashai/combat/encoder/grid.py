# clashai/combat/encoder/grid.py
# YOLO buildings -> (NUM_CHANNELS, GRID, GRID) tensor + danger heatmaps.

import math
import numpy as np

from clashai.config import GRID_SIZE
from clashai.combat.encoder.constants import (
    CLASS_TO_CHANNEL, NUM_CHANNELS, CELL_WIDTH, CELL_HEIGHT, DEFENSE_STATS,
)


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

