# scripts/rl/capture_combat.py
# Capture automatique de screenshots pendant les combats pour annotation YOLO.
#
# Ce script tourne EN PARALLÈLE du Brain ou de l'entraînement.
# Il prend des screenshots toutes les 2 secondes quand le CNN détecte
# l'écran "phase_attaque" (combat en cours).
#
# Les screenshots sont sauvegardés dans combat_captures/ avec un nom unique.
# Tu les annoteras ensuite avec un outil comme LabelImg, Roboflow ou CVAT.
#
# Usage :
#   python scripts/rl/capture_combat.py
#   python scripts/rl/capture_combat.py --interval 3 --max 500
#   python scripts/rl/capture_combat.py --output mon_dossier/

import os
import time
import argparse
from datetime import datetime

from clashai.paths import PROJECT_ROOT as project_root


# =============================================================================
#                         CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT_DIR = os.path.join(project_root, 'combat_captures')
DEFAULT_INTERVAL = 2.0      # Secondes entre chaque capture
DEFAULT_MAX_CAPTURES = 300   # Limite par session
COMBAT_STATES = ['phase_attaque']  # États CNN = combat en cours


# =============================================================================
#                         CAPTURE
# =============================================================================

def capture_combat_screenshots(output_dir=DEFAULT_OUTPUT_DIR,
                                interval=DEFAULT_INTERVAL,
                                max_captures=DEFAULT_MAX_CAPTURES):
    """
    Capture des screenshots pendant les combats.
    
    Attend que le CNN détecte 'phase_attaque', puis capture
    toutes les N secondes jusqu'à ce que l'écran change.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les modèles
    print("📦 Chargement des modèles...")
    from clashai.navigation import game_loop as gl
    models = gl.load_models()
    
    total_captured = 0
    combat_count = 0
    
    print(f"\n{'='*60}")
    print("  📸 Capture de combat pour annotation YOLO")
    print(f"  Dossier  : {output_dir}")
    print(f"  Interval : {interval}s")
    print(f"  Max      : {max_captures}")
    print(f"{'='*60}\n")
    print("En attente d'un combat...\n")
    
    try:
        while total_captured < max_captures:
            # Attendre qu'un combat commence
            img = gl.adb_screenshot()
            if img is None:
                time.sleep(1)
                continue
            
            state, conf = gl.classify_screen(img, models)
            
            if state not in COMBAT_STATES:
                time.sleep(2)
                continue
            
            # Combat détecté !
            combat_count += 1
            combat_captures = 0
            print(f"⚔️  Combat #{combat_count} détecté !")
            
            while total_captured < max_captures:
                img = gl.adb_screenshot()
                if img is None:
                    time.sleep(1)
                    continue
                
                state, conf = gl.classify_screen(img, models)
                
                if state not in COMBAT_STATES:
                    print(f"   Combat #{combat_count} terminé "
                          f"({combat_captures} captures)")
                    break
                
                # Sauvegarder le screenshot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"combat_{combat_count:03d}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                
                total_captured += 1
                combat_captures += 1
                
                if combat_captures % 10 == 0:
                    print(f"   📸 {combat_captures} captures "
                          f"(total: {total_captured}/{max_captures})")
                
                time.sleep(interval)
            
            print("   En attente du prochain combat...\n")
    
    except KeyboardInterrupt:
        print("\n⛔ Arrêt")
    
    print(f"\n{'='*60}")
    print("  📸 Capture terminée")
    print(f"  Combats   : {combat_count}")
    print(f"  Captures  : {total_captured}")
    print(f"  Dossier   : {output_dir}")
    print(f"{'='*60}")
    print("\n📝 Prochaine étape :")
    print("   1. Ouvre les images dans un outil d'annotation (LabelImg, CVAT, Roboflow)")
    print("   2. Annote les troupes avec des bounding boxes")
    print("   3. Exporte en format YOLO (txt)")
    print("   4. Entraîne le modèle avec train_yolo_troops.py")


# =============================================================================
#                         MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture de screenshots de combat pour annotation YOLO"
    )
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Dossier de sortie")
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                        help="Intervalle entre captures (secondes)")
    parser.add_argument('--max', type=int, default=DEFAULT_MAX_CAPTURES,
                        help="Nombre max de captures")
    
    args = parser.parse_args()
    
    capture_combat_screenshots(
        output_dir=args.output,
        interval=args.interval,
        max_captures=args.max,
    )
