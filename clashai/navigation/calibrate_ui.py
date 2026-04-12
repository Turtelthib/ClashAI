# scripts/rl/calibrate_ui.py
# Calibrateur d'interface pour ClashAI.
#
# Enregistre les positions de tous les boutons UI en guidant l'utilisateur.
# Les coordonnées sont sauvegardées dans ui_positions.json et utilisées
# par tous les modules (brain, chat monitor, gdc navigator, etc.).
#
# Usage :
#   python -m clashai.navigation.calibrate_ui              (calibration complète)
#   python -m clashai.navigation.calibrate_ui --only chat   (recalibrer seulement le chat)
#   python -m clashai.navigation.calibrate_ui --show        (afficher les positions actuelles)
#
# Méthode :
#   1. Le script prend un screenshot ADB
#   2. Il l'affiche dans une fenêtre OpenCV
#   3. L'utilisateur clique sur le bouton dans l'image
#   4. Les coordonnées sont converties en ADB et sauvegardées

import os
import json
import time
import subprocess
import io

from PIL import Image

# =============================================================================
#                         CONFIGURATION
# =============================================================================

from clashai.paths import UI_POSITIONS_FILE as _UI_POS

POSITIONS_FILE = _UI_POS

ADB_WIDTH = 1920
ADB_HEIGHT = 1080

# Tous les boutons à calibrer, regroupés par contexte
# Chaque entrée : (clé_json, description, écran_requis, delay_avant)
# delay_avant = secondes d'attente AVANT de capturer (pour se mettre en situation)
BUTTONS_TO_CALIBRATE = {
    'village': [
        ('chat_open', '💬 Bouton pour OUVRIR le chat du clan', 'village_home', 0),
        ('chat_close_tap', '❌ Tapez EN DEHORS du chat pour le FERMER', 'chat_clan', 0),
        ('attack_button', '⚔️ Bouton ATTAQUER (en bas à gauche)', 'village_home', 0),
    ],
    'chat': [
        ('chat_input', '✏️ Barre de saisie "Message de clan..." (en bas du chat)', 'chat_clan', 0),
        ('chat_send', '➤ Bouton ENVOYER le message (flèche verte)', 'chat_clan', 0),
    ],
    'matchmaking': [
        ('find_match', '🔍 Bouton TROUVER UNE PARTIE', 'recherche_adversaire', 0),
        ('start_attack', '▶️ Bouton LANCER L\'ATTAQUE (pour confirmer)', 'prep_attaque', 0),
    ],
    'results': [
        ('return_home', '🏠 Bouton RENTRER AU VILLAGE (après une attaque)', 'resultats_attaque', 20),
    ],
    'gdc': [
        # Flow complet de navigation GdC dans l'ordre d'utilisation :
        # 1. Depuis le village → accéder au menu GdC
        ('gdc_open', '🏰 Bouton pour ACCÉDER AU MENU GDC depuis le village', 'village_home', 0),
        # 2. Écran "guerre terminée" → voir la carte
        ('gdc_war_ended_see_map', '🗺️ Bouton VOIR LA CARTE (écran guerre terminée)', None, 10),
        # 3. Sur la carte → basculer vers les ennemis
        ('gdc_enemy_map', '👁️ Bouton CARTE ENNEMIE (voir les ennemis)', 'gdc_ally', 0),
        # 4. Sur la carte → basculer vers les alliés
        ('gdc_ally_map', '🛡️ Bouton CARTE ALLIÉE (voir les alliés)', 'gdc_enemy', 0),
        # 5. Quand on a cliqué sur un village ennemi → popup avec "Attaquer"
        ('gdc_attack_target', '⚔️ Bouton ATTAQUER dans le popup de cible GdC', None, 15),
        # 6. Flèche SUIVANT (→) dans le popup de cible (village n+1)
        ('gdc_village_next', '➡️ Flèche SUIVANT (droite) dans le popup village', None, 0),
        # 7. Flèche PRÉCÉDENT (←) dans le popup de cible (village n-1)
        ('gdc_village_prev', '⬅️ Flèche PRÉCÉDENT (gauche) dans le popup village', None, 0),
        # 8. Depuis le menu GdC → retour au village
        ('gdc_return_home', '🏠 Bouton RETOUR AU VILLAGE depuis le menu GdC', None, 0),
    ],
    'general': [
        ('open_profil', '👤 Bouton pour OUVRIR le profil (depuis le village)', 'village_home', 0),
        ('close_profil', '❌ Bouton pour FERMER le profil', 'profil', 0),
    ],
    'retreat': [
        # Boutons de retraite (abandon) pendant un combat
        # 1. Le drapeau blanc en haut à droite pendant le combat
        ('ff_button', '🏳️ Bouton RETRAITE (drapeau blanc, en haut à droite pendant le combat)', 'phase_attaque', 0),
        # 2. Le bouton de CONFIRMATION dans la popup
        ('confirm_ff', '✅ Bouton CONFIRMER la retraite (dans la popup de confirmation)', None, 0),
    ],
}

# Positions par défaut (fallback si pas calibré)
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
#                    CAPTURE DU CLIC (screenshot + fenêtre OpenCV)
# =============================================================================

def _adb_screenshot():
    """Capture l'écran ADB et retourne une image PIL."""
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
    """Callback OpenCV pour capturer le clic souris."""
    import cv2
    if event == cv2.EVENT_LBUTTONDOWN and not _click_result['done']:
        _click_result['x'] = x
        _click_result['y'] = y
        _click_result['done'] = True


def capture_click(description=""):
    """
    Prend un screenshot ADB, l'affiche dans une fenêtre OpenCV,
    et attend que l'utilisateur clique dessus.
    
    Les coordonnées du clic sont converties en coordonnées ADB (1920×1080).
    
    Returns:
        (x, y) en coordonnées ADB, ou None si Echap / fenêtre fermée
    """
    import cv2
    import numpy as np

    global _click_result
    _click_result = {'x': None, 'y': None, 'done': False}

    # 1. Screenshot ADB
    img_pil = _adb_screenshot()
    if img_pil is None:
        print("   ❌ Impossible de capturer l'écran")
        return None

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv.shape[:2]

    # 2. Réduire si trop grand pour l'écran
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

    # 3. Texte d'instruction
    label = f"CLIQUEZ : {description}" if description else "CLIQUEZ sur le bouton"
    cv2.putText(display_img, label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_img, "Echap = passer", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # 4. Afficher la fenêtre
    window_name = "ClashAI Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_callback)
    cv2.imshow(window_name, display_img)

    # 5. Attendre le clic ou Echap
    while not _click_result['done']:
        key = cv2.waitKey(100)
        if key == 27:  # Echap
            cv2.destroyAllWindows()
            return None
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except cv2.error:
            return None

    cv2.destroyAllWindows()

    # 6. Convertir coordonnées display → ADB
    click_x = _click_result['x']
    click_y = _click_result['y']

    adb_x = int(click_x / scale * ADB_WIDTH / img_w)
    adb_y = int(click_y / scale * ADB_HEIGHT / img_h)

    adb_x = max(0, min(ADB_WIDTH - 1, adb_x))
    adb_y = max(0, min(ADB_HEIGHT - 1, adb_y))

    return (adb_x, adb_y)


# =============================================================================
#                    CHARGEMENT / SAUVEGARDE
# =============================================================================

def load_positions():
    """Charge les positions depuis le fichier JSON."""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                data = json.load(f)
            # Convertir les listes en tuples
            return {k: tuple(v) for k, v in data.items()}
        except (json.JSONDecodeError, Exception):
            pass
    return {}


def save_positions(positions):
    """Sauvegarde les positions dans le fichier JSON."""
    # Convertir les tuples en listes pour JSON
    data = {k: list(v) for k, v in positions.items()}
    
    tmp_path = POSITIONS_FILE + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Rename atomique
    if os.path.exists(POSITIONS_FILE):
        os.replace(tmp_path, POSITIONS_FILE)
    else:
        os.rename(tmp_path, POSITIONS_FILE)


def get_position(key):
    """
    Récupère une position calibrée, avec fallback sur la valeur par défaut.
    
    C'est cette fonction que tous les modules doivent appeler.
    
    Args:
        key: str (ex: 'chat_open', 'attack_button', etc.)
        
    Returns:
        (x, y) tuple
    """
    positions = load_positions()
    if key in positions:
        return positions[key]
    if key in DEFAULT_POSITIONS:
        return DEFAULT_POSITIONS[key]
    return (960, 540)  # Centre de l'écran en dernier recours


# =============================================================================
#                    CALIBRATION
# =============================================================================

def calibrate(groups=None):
    """
    Lance la calibration interactive.
    
    Args:
        groups: liste de groupes à calibrer (None = tous)
    """
    positions = load_positions()
    
    print(f"\n{'='*60}")
    print("  🎯 ClashAI — Calibration de l'interface")
    print(f"{'='*60}")
    print("\n  Pour chaque bouton, un screenshot s'affiche dans une fenêtre.")
    print("  Cliquez sur le bouton dans l'image.")
    print("  Echap = passer un bouton.")
    print("  Les positions sont sauvegardées dans ui_positions.json\n")
    
    if groups is None:
        groups = list(BUTTONS_TO_CALIBRATE.keys())
    
    calibrated = 0
    skipped = 0
    
    for group_name in groups:
        if group_name not in BUTTONS_TO_CALIBRATE:
            print(f"  ⚠️  Groupe '{group_name}' inconnu")
            continue
        
        buttons = BUTTONS_TO_CALIBRATE[group_name]
        print(f"\n  ── {group_name.upper()} ──")
        
        for key, description, required_screen, delay in buttons:
            current = positions.get(key)
            current_str = f" (actuel: {current})" if current else ""
            
            print(f"\n  📍 {description}{current_str}")
            
            if required_screen:
                print(f"     ⚠️  Assurez-vous d'être sur l'écran : {required_screen}")
            
            # Délai d'attente si nécessaire (pour se mettre en situation)
            if delay > 0:
                print(f"     ⏱️  {delay}s pour vous mettre en position...")
                for remaining in range(delay, 0, -5):
                    print(f"         {remaining}s...", end='\r')
                    time.sleep(min(5, remaining))
                print("         C'est parti !        ")
            
            print("     → Un screenshot va s'afficher, CLIQUEZ sur le bouton")
            print("       (Echap = passer)")
            
            # Screenshot + clic dans la fenêtre OpenCV
            pos = capture_click(description=description)
            
            if pos is not None:
                x, y = pos
                positions[key] = (x, y)
                print(f"     ✅ {key} = ({x}, {y})")
                calibrated += 1
            else:
                if current:
                    print(f"     ⏭️  Gardé l'ancienne valeur : {current}")
                else:
                    default = DEFAULT_POSITIONS.get(key, (960, 540))
                    positions[key] = default
                    print(f"     ⏭️  Valeur par défaut : {default}")
                skipped += 1
    
    # Sauvegarder
    save_positions(positions)
    
    print(f"\n{'='*60}")
    print("  ✅ Calibration terminée !")
    print(f"     {calibrated} boutons calibrés, {skipped} passés")
    print(f"     Sauvegardé dans : {POSITIONS_FILE}")
    print(f"{'='*60}\n")
    
    return positions


def show_positions():
    """Affiche toutes les positions calibrées."""
    positions = load_positions()
    
    print(f"\n📍 Positions UI ({POSITIONS_FILE}) :\n")
    
    if not positions:
        print("   (aucune position calibrée)")
        print("   Lancez : python -m clashai.navigation.calibrate_ui")
        return
    
    for group_name, buttons in BUTTONS_TO_CALIBRATE.items():
        print(f"  ── {group_name.upper()} ──")
        for key, description, _, _ in buttons:
            pos = positions.get(key)
            default = DEFAULT_POSITIONS.get(key)
            if pos:
                is_default = pos == default
                marker = " (défaut)" if is_default else " ✅"
                print(f"   {key:20s} = ({pos[0]:4d}, {pos[1]:4d}){marker}")
            else:
                print(f"   {key:20s} = ???  ← non calibré")
        print()


# =============================================================================
#                            MAIN
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
        print(f"✅ Positions remises par défaut dans {POSITIONS_FILE}")
        show_positions()
    else:
        calibrate(groups=args.only)