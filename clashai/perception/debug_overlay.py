# clashai/perception/debug_overlay.py
# Debug overlay — annotated image of what the agent perceives at each observe.
#
# Saves to logs/episode_N/step_S.png when --debug-overlay is active.
# One folder per episode, one image per observe step.
#
# Image contents:
#   - YOLO buildings (colored bboxes by category)
#   - Deploy zone positions (red numbered circles)
#   - YOLO troops (cyan circles)
#   - Text overlay: step, troops remaining, spells remaining

import os
import cv2
import numpy as np

# Building category colors (BGR)
BUILDING_COLORS = {
    'defense':   (0,   0,   255),  # red
    'resource':  (0,   200, 255),  # yellow
    'army':      (255, 165, 0),    # blue-orange
    'hero':      (255, 0,   200),  # magenta
    'trap':      (0,   128, 255),  # orange
    'wall':      (128, 128, 128),  # gray
    'other':     (0,   255, 0),    # green
}

DEFENSE_CLASSES = {
    'canon', 'mortier', 'tour_archer', 'tour_inferno', 'tour_enfer_mono',
    'tour_enfer_multiple', 'tesla', 'lance_air', 'balayeur_air',
    'aigle_artilleur', 'catapulte_erratique', 'arcX_sol', 'arcX_sol_air',
    'monolithe', 'scattershot', 'tour_ricochet',
}
RESOURCE_CLASSES = {
    'mine_or', 'collecteur_elixir', 'coffre_or', 'reserve_elixir',
    'foreuse_elixir_sombre', 'reserve_elixir_sombre',
}


def _building_color(cls_name):
    if cls_name in DEFENSE_CLASSES:
        return BUILDING_COLORS['defense']
    if cls_name in RESOURCE_CLASSES:
        return BUILDING_COLORS['resource']
    if cls_name in ('caserne', 'caserne_sombre', 'usine_sort', 'usine_siege'):
        return BUILDING_COLORS['army']
    if cls_name == 'hotel_de_ville':
        return (0, 0, 200)
    if cls_name in ('mur', 'rempart'):
        return BUILDING_COLORS['wall']
    return BUILDING_COLORS['other']


def save_debug_overlay(screenshot_pil, step, episode,
                       buildings=None, deploy_positions=None,
                       troop_positions=None, remaining_troops=None,
                       troop_types=None, output_root='logs'):
    """
    Saves an annotated debug image for the current observe step.

    Args:
        screenshot_pil : PIL.Image — current frame
        step           : int — current step number
        episode        : int — current episode number
        buildings      : list of {class, bbox, center}
        deploy_positions: list of (x, y) ADB positions
        troop_positions : dict {name: (x, y, conf)} from TroopFinder
        remaining_troops: np.array — troop counts
        troop_types     : list of dicts with 'name' and 'role'
        output_root     : root logs directory

    Returns:
        path of saved file, or None on error
    """
    try:
        ep_dir = os.path.join(output_root, f'episode_{episode:04d}')
        os.makedirs(ep_dir, exist_ok=True)

        img = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale_x = w / 1920
        scale_y = h / 1080

        def sx(x): return int(x * scale_x)
        def sy(y): return int(y * scale_y)

        # ── 1. YOLO buildings ─────────────────────────────────────
        if buildings:
            for b in buildings:
                x1, y1, x2, y2 = b['bbox']
                color = _building_color(b.get('class', ''))
                cv2.rectangle(img, (sx(x1), sy(y1)), (sx(x2), sy(y2)), color, 1)

        # ── 2. Deploy positions (red numbered circles) ────────────
        if deploy_positions:
            for i, (px, py) in enumerate(deploy_positions):
                cv2.circle(img, (sx(px), sy(py)), 8, (0, 0, 220), -1)
                cv2.putText(img, str(i), (sx(px) - 5, sy(py) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # ── 3. Troop bar positions (cyan, from YOLO detector) ─────
        if troop_positions:
            for name, (tx, ty, conf) in troop_positions.items():
                cv2.circle(img, (sx(tx), sy(ty)), 10, (255, 200, 0), 2)
                cv2.putText(img, name[:6], (sx(tx) - 18, sy(ty) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)

        # ── 4. Text overlay (top-left) ────────────────────────────
        lines = [f'Ep {episode}  Step {step}']

        if remaining_troops is not None and troop_types is not None:
            troops_txt = '  '.join(
                f"{t['name']}:{int(remaining_troops[i])}"
                for i, t in enumerate(troop_types)
                if int(remaining_troops[i]) > 0 and t['role'] != 'spell'
            )
            spells_txt = '  '.join(
                f"{t['name']}:{int(remaining_troops[i])}"
                for i, t in enumerate(troop_types)
                if int(remaining_troops[i]) > 0 and t['role'] == 'spell'
            )
            if troops_txt:
                lines.append(f'Troops: {troops_txt}')
            if spells_txt:
                lines.append(f'Spells: {spells_txt}')

        if buildings:
            lines.append(f'Buildings: {len(buildings)}')
        if deploy_positions:
            lines.append(f'Deploy pts: {len(deploy_positions)}')

        for i, line in enumerate(lines):
            y_pos = 20 + i * 18
            cv2.putText(img, line, (6, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, line, (6, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── 5. Legend (bottom-right) ──────────────────────────────
        legend = [
            ('Buildings', (0, 255, 0)),
            ('Defense',   (0, 0, 255)),
            ('Deploy pt', (0, 0, 220)),
            ('Troop bar', (255, 200, 0)),
        ]
        for j, (label, color) in enumerate(legend):
            ly = h - 10 - j * 16
            cv2.circle(img, (w - 80, ly - 4), 5, color, -1)
            cv2.putText(img, label, (w - 70, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        fname = f'step_{step:03d}.jpg'
        path = os.path.join(ep_dir, fname)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    except Exception as e:
        print(f" WARNING: debug overlay failed: {e}")
        return None
