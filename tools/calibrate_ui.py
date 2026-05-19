# scripts/rl/calibrate_ui.py
# UI calibrator for ClashAI.
#
# Records the positions of all UI buttons by guiding the user.
# Coordinates are saved in ui_positions.json and used
# by all modules (brain, chat monitor, gdc navigator, etc.).
#
# Usage:
# python scripts/rl/calibrate_ui.py (full calibration)
# python scripts/rl/calibrate_ui.py --only chat (recalibrate chat only)
# python scripts/rl/calibrate_ui.py --show (display current positions)
#
# Method:
# 1. The script takes an ADB screenshot
# 2. It displays it in an OpenCV window
# 3. The user clicks on the button in the image
# 4. Coordinates are converted to ADB and saved

import os
import sys
import json
import time
import subprocess
import io

from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))

POSITIONS_FILE = os.path.join(project_root, 'ui_positions.json')

# Re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# All buttons to calibrate, grouped by context
# Each entry: (json_key, description, required_screen, pre_delay)
# pre_delay = seconds to wait BEFORE capturing (to get into position)
BUTTONS_TO_CALIBRATE = {
    'village': [
        ('chat_open', ' Bouton pour OUVRIR le chat du clan', 'village_home', 0),
        ('chat_close_tap', 'ERROR: Tapez EN DEHORS du chat pour le FERMER', 'chat_clan', 0),
        ('attack_button', ' Bouton ATTAQUER (en bas à gauche)', 'village_home', 0),
    ],
    'chat': [
        ('chat_input', ' Barre de saisie "Message de clan..." (en bas du chat)', 'chat_clan', 0),
        ('chat_send', ' Bouton ENVOYER le message (flèche verte)', 'chat_clan', 0),
    ],
    'matchmaking': [
        ('find_match', 'Bouton TROUVER UNE PARTIE', 'recherche_adversaire', 0),
        ('start_attack', ' Bouton LANCER L\'ATTAQUE (pour confirmer)', 'prep_attaque', 0),
    ],
    'results': [
        ('return_home', 'Bouton RENTRER AU VILLAGE (après une attaque)', 'resultats_attaque', 20),
    ],
    'gdc': [
        # Full GdC navigation flow in order of use:
        # 1. From the village → access the GdC menu
        ('gdc_open', 'Bouton pour ACCÉDER AU MENU GDC depuis le village', 'village_home', 0),
        # 2. "War ended" screen → view the map
        ('gdc_war_ended_see_map', ' Bouton VOIR LA CARTE (écran guerre terminée)', None, 10),
        # 3. On the map → switch to enemies
        ('gdc_enemy_map', 'Bouton CARTE ENNEMIE (voir les ennemis)', 'gdc_ally', 0),
        # 4. On the map → switch to allies
        ('gdc_ally_map', ' Bouton CARTE ALLIÉE (voir les alliés)', 'gdc_enemy', 0),
        # 5. When an enemy village is clicked → popup with "Attack"
        ('gdc_attack_target', ' Bouton ATTAQUER dans le popup de cible GdC', None, 15),
        # 6. NEXT arrow (→) in the target popup (village n+1)
        ('gdc_village_next', ' Flèche SUIVANT (droite) dans le popup village', None, 0),
        # 7. PREVIOUS arrow (←) in the target popup (village n-1)
        ('gdc_village_prev', ' Flèche PRÉCÉDENT (gauche) dans le popup village', None, 0),
        # 8. From the GdC menu → back to village
        ('gdc_return_home', 'Bouton RETOUR AU VILLAGE depuis le menu GdC', None, 0),
    ],
    'general': [
        ('open_profil', ' Bouton pour OUVRIR le profil (depuis le village)', 'village_home', 0),
        ('close_profil', 'ERROR: Bouton pour FERMER le profil', 'profil', 0),
    ],
    'retreat': [
        # Retreat (surrender) buttons during combat
        # 1. The white flag in the top-right corner during combat
        ('ff_button', 'Bouton RETRAITE (drapeau blanc, en haut à droite pendant le combat)', 'phase_attaque', 0),
        # 2. The CONFIRMATION button in the popup
        ('confirm_ff', 'Bouton CONFIRMER la retraite (dans la popup de confirmation)', None, 0),
    ],
    'cdc': [
        ('cdc_confirmation', 'bouton confirmation de demande de renfort', 'village_home', 0),
    ],
}

# Default positions (fallback if not calibrated)
DEFAULT_POSITIONS = {
    'chat_open': (47, 400),
    'chat_close_tap': (1400, 400),
    'chat_input': (300, 1010),
    'chat_send': (490, 1010),
    'attack_button': (80, 830),
    'find_match': (960, 700),
    'start_attack': (960, 700),
    'return_home': (960, 800),
    'gdc_open': (47, 560),
    'gdc_war_ended_see_map': (960, 700),
    'gdc_enemy_map': (1700, 540),
    'gdc_ally_map': (200, 540),
    'gdc_attack_target': (900, 660),
    'gdc_village_next': (1050, 680),
    'gdc_village_prev': (100, 680),
    'gdc_return_home': (80, 780),
    'open_profil': (40, 50),
    'close_profil': (1270, 90),
    'close_menu': (1340, 95),
    'close_popup': (1300, 100),
    'ff_button': (1850, 550),
    'confirm_ff': (700, 550),
}


# =============================================================================
# CLICK CAPTURE (screenshot + OpenCV window)
# =============================================================================

def _adb_screenshot():
    """Captures the ADB screen and returns a PIL image."""
    try:
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0 or len(result.stdout) < 100:
            return None
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception:
        return None


_click_result = {'x': None, 'y': None, 'done': False}


def _mouse_callback(event, x, y, flags, param):
    """OpenCV callback to capture mouse click."""
    import cv2
    if event == cv2.EVENT_LBUTTONDOWN and not _click_result['done']:
        _click_result['x'] = x
        _click_result['y'] = y
        _click_result['done'] = True


def capture_click(description=""):
    """
    Takes an ADB screenshot, displays it in an OpenCV window,
    and waits for the user to click on it.

    Click coordinates are converted to ADB coordinates (1920x1080).

    Returns:
        (x, y) in ADB coordinates, or None if Escape / window closed
    """
    import cv2
    import numpy as np

    global _click_result
    _click_result = {'x': None, 'y': None, 'done': False}

    # 1. ADB screenshot
    img_pil = _adb_screenshot()
    if img_pil is None:
        print(" ERROR: Impossible de capturer l'écran")
        return None

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv.shape[:2]

    # 2. Resize if too large for the screen
    max_display_w = 1280
    max_display_h = 720
    scale = min(max_display_w / img_w, max_display_h / img_h, 1.0)

    if scale < 1.0:
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)
        display_img = cv2.resize(img_cv, (display_w, display_h))
    else:
        display_img = img_cv.copy()
        scale = 1.0

    # 3. Instruction text
    label = f"CLIQUEZ : {description}" if description else "CLIQUEZ sur le bouton"
    cv2.putText(display_img, label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_img, "Echap = passer", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # 4. Display the window
    window_name = "ClashAI Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_callback)
    cv2.imshow(window_name, display_img)

    # 5. Wait for click or Escape
    while not _click_result['done']:
        key = cv2.waitKey(100)
        if key == 27:
            cv2.destroyAllWindows()
            return None
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except cv2.error:
            return None

    cv2.destroyAllWindows()

    # 6. Convert display coordinates → ADB
    click_x = _click_result['x']
    click_y = _click_result['y']

    adb_x = int(click_x / scale * ADB_WIDTH / img_w)
    adb_y = int(click_y / scale * ADB_HEIGHT / img_h)

    adb_x = max(0, min(ADB_WIDTH - 1, adb_x))
    adb_y = max(0, min(ADB_HEIGHT - 1, adb_y))

    return (adb_x, adb_y)


# =============================================================================
# LOADING / SAVING
# =============================================================================

def load_positions():
    """Loads positions from the JSON file."""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                data = json.load(f)
            # Convert lists to tuples
            return {k: tuple(v) for k, v in data.items()}
        except (json.JSONDecodeError, Exception):
            pass
    return {}


def save_positions(positions):
    """Saves positions to the JSON file."""
    # Convert tuples to lists for JSON
    data = {k: list(v) for k, v in positions.items()}
    
    tmp_path = POSITIONS_FILE + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Atomic rename
    if os.path.exists(POSITIONS_FILE):
        os.replace(tmp_path, POSITIONS_FILE)
    else:
        os.rename(tmp_path, POSITIONS_FILE)


def get_position(key):
    """
    Retrieves a calibrated position, with fallback to the default value.

    This is the function that all modules should call.

    Args:
        key: str (e.g. 'chat_open', 'attack_button', etc.)

    Returns:
        (x, y) tuple
    """
    positions = load_positions()
    if key in positions:
        return positions[key]
    if key in DEFAULT_POSITIONS:
        return DEFAULT_POSITIONS[key]
    return (960, 540)


# =============================================================================
# CALIBRATION
# =============================================================================


def calibrate(groups=None):
    """
    Starts the interactive calibration.

    Args:
        groups: list of groups to calibrate (None = all)
    """
    positions = load_positions()
    
    print(f"\n{'='*60}")
    print(f" ClashAI — Calibration de l'interface")
    print(f"{'='*60}")
    print(f"\n Pour chaque bouton, un screenshot s'affiche dans une fenêtre.")
    print(f" Cliquez sur le bouton dans l'image.")
    print(f" Echap = passer un bouton.")
    print(f" Les positions sont sauvegardées dans ui_positions.json\n")
    
    if groups is None:
        groups = list(BUTTONS_TO_CALIBRATE.keys())
    
    calibrated = 0
    skipped = 0
    
    for group_name in groups:
        if group_name not in BUTTONS_TO_CALIBRATE:
            print(f" WARNING: Groupe '{group_name}' inconnu")
            continue
        
        buttons = BUTTONS_TO_CALIBRATE[group_name]
        print(f"\n  {group_name.upper()} ")
        
        for key, description, required_screen, delay in buttons:
            current = positions.get(key)
            current_str = f" (actuel: {current})" if current else ""
            
            print(f"\n {description}{current_str}")
            
            if required_screen:
                print(f" WARNING: Assurez-vous d'être sur l'écran : {required_screen}")
            
            # Wait delay if needed (to get into position)
            if delay > 0:
                print(f" {delay}s pour vous mettre en position...")
                for remaining in range(delay, 0, -5):
                    print(f" {remaining}s...", end='\r')
                    time.sleep(min(5, remaining))
                print(f" C'est parti ! ")
            
            print(f" → Un screenshot va s'afficher, CLIQUEZ sur le bouton")
            print(f" (Echap = passer)")
            
            # Screenshot + click in the OpenCV window
            pos = capture_click(description=description)
            
            if pos is not None:
                x, y = pos
                positions[key] = (x, y)
                print(f" {key} = ({x}, {y})")
                calibrated += 1
            else:
                if current:
                    print(f"  Gardé l'ancienne valeur : {current}")
                else:
                    default = DEFAULT_POSITIONS.get(key, (960, 540))
                    positions[key] = default
                    print(f"  Valeur par défaut : {default}")
                skipped += 1
    
    # Save
    save_positions(positions)
    
    print(f"\n{'='*60}")
    print(f" Calibration terminée !")
    print(f" {calibrated} boutons calibrés, {skipped} passés")
    print(f" Sauvegardé dans : {POSITIONS_FILE}")
    print(f"{'='*60}\n")
    
    return positions


def show_positions():
    """Displays all calibrated positions."""
    positions = load_positions()
    
    print(f"\nPositions UI ({POSITIONS_FILE}) :\n")
    
    if not positions:
        print(" (aucune position calibrée)")
        print(f" Lancez : python scripts/rl/calibrate_ui.py")
        return
    
    for group_name, buttons in BUTTONS_TO_CALIBRATE.items():
        print(f"  {group_name.upper()} ")
        for key, description, _, _ in buttons:
            pos = positions.get(key)
            default = DEFAULT_POSITIONS.get(key)
            if pos:
                is_default = pos == default
                marker = " (défaut)" if is_default else " "
                print(f" {key:20s} = ({pos[0]:4d}, {pos[1]:4d}){marker}")
            else:
                print(f" {key:20s} = ??? ← non calibré")
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ClashAI UI Calibrator")
    parser.add_argument('--show', action='store_true',
                        help="Afficher les positions actuelles")
    parser.add_argument('--only', type=str, nargs='+',
                        help="Calibrer seulement certains groupes "
                             "(village, chat, matchmaking, results, gdc, general, retreat)")
    parser.add_argument('--reset', action='store_true',
                        help="Remettre toutes les positions par défaut")
    
    args = parser.parse_args()
    
    if args.show:
        show_positions()
    elif args.reset:
        save_positions(DEFAULT_POSITIONS)
        print(f"Positions remises par défaut dans {POSITIONS_FILE}")
        show_positions()
    else:
        calibrate(groups=args.only)