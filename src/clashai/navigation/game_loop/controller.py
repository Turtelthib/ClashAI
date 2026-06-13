# clashai/navigation/game_loop/controller.py
# CLI controller: button coords, state dispatch, test/live loops, main().

import os
import sys
import time
import argparse

import cv2
import numpy as np
from PIL import Image

from clashai.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ADB_DELAY_SCREENSHOT, ADB_DELAY_NAVIGATION, ADB_DELAY_MATCHMAKING,
)
from clashai.paths import ADB_DEVICE
from clashai.navigation.game_loop.constants import SCREEN_CONFIDENCE_THRESHOLD
from clashai.navigation.game_loop.models import load_models
from clashai.navigation.game_loop.analysis import (
    classify_screen, analyze_village, get_village_summary,
)
from clashai.navigation.game_loop.adb_io import (
    adb_screenshot, adb_tap, adb_swipe, adb_key, adb_check_connection,
)


BUTTONS = {
    # Main village (village_home)
    'attaquer': (80, 950),

    # Attack selection menu (recherche_adversaire)
    'trouver_partie': (235, 780),

    # Army preparation screen (prep_attaque)
    'lancer_attaque': (1630, 930),
    'fermer_prep': (1355, 95),

    # Scout enemy village (phase_attaque — before deployment)
    'suivant': (1640, 850),
    'terminer_bataille': (75, 650),

    # Attack results
    'fin_combat': (960, 800),

    # General navigation
    'retour': (50, 50),
    'centre_ecran': (960, 540),
}


def handle_state(state, confidence, models, img_pil=None):
    """
    Decides what to do based on the detected state.

    Game flow:
    village_home → [tap Attack] → recherche_adversaire → [tap Find a match]
    → prep_attaque → [tap green Attack] → chargement → phase_attaque (scout)
    → [analyze village, RL decides] → resultats_attaque → [tap close] → village_home
    """
    result = {
        'state': state,
        'confidence': confidence,
        'action': None,
        'buildings': None,
        'summary': None,
    }

    if state == 'village_home':
        # → We are home. Open the attack menu.
        result['action'] = 'tap_attaquer'
        print("Village principal détecté → Ouverture du menu attaque")

    elif state == 'recherche_adversaire':
        # → Combat / Ranked combat menu. Start matchmaking.
        result['action'] = 'tap_trouver_partie'
        print("Menu attaque détecté → Recherche d'un adversaire")

    elif state == 'prep_attaque':
        # → Army screen with the green Attack button. Launch!
        result['action'] = 'tap_lancer_attaque'
        print(" Préparation armée détectée → Lancement de l'attaque")

    elif state == 'phase_attaque':
        if img_pil is not None:
            print("Village ennemi en vue → Attente 3s (décorations)...")
            time.sleep(3)
            # Take a fresh capture after the wait
            fresh_img = adb_screenshot()
            if fresh_img is not None:
                img_pil = fresh_img
            print("Analyse des bâtiments...")
            buildings = analyze_village(img_pil, models)
            summary = get_village_summary(buildings)
            result['buildings'] = buildings
            result['summary'] = summary
            result['action'] = 'analyze_village'

            print(f" {summary['total']} bâtiments détectés")
            print(f"  {summary['defenses']} défenses")
            print(f"  {summary['ressources']} bâtiments de ressources")

            for cls, count in sorted(summary['details'].items()):
                print(f" {cls}: {count}")

            # → Later: the RL agent will decide here (attack or Next)
        else:
            result['action'] = 'need_screenshot'

    elif state == 'resultats_attaque':
        # → Combat over. Close and return to village.
        result['action'] = 'tap_fin_combat'
        print("* Résultats d'attaque détectés → Retour au village")
        # → Later: extract stars/% for the RL reward

    elif state == 'chargement':
        result['action'] = 'wait'
        print(" Chargement en cours...")

    elif state == 'chat_clan':
        result['action'] = 'close_menu'
        print(" Chat de clan détecté → Fermeture")

    elif state == 'menu_boutique':
        result['action'] = 'close_menu'
        print(" Boutique détectée → Fermeture")

    else:
        result['action'] = 'unknown'
        print(f" État inconnu : {state} ({confidence:.1%})")

    return result

def run_test(image_path, models):
    """
    Mode test : analyse une image statique.
    Fait exactement ce que fera le mode live, mais sans ADB.
    """
    print(f"\n{'='*60}")
    print(f" MODE TEST — Analyse de : {image_path}")
    print(f"{'='*60}\n")

    if not os.path.exists(image_path):
        print(f"ERROR: Image introuvable : {image_path}")
        return

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"ERROR: Impossible de lire : {image_path}")
        return

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    state, confidence = classify_screen(img_pil, models)
    print(f"État détecté : {state} ({confidence:.1%})")

    if confidence < SCREEN_CONFIDENCE_THRESHOLD:
        print(f"WARNING: Confiance trop basse ({confidence:.1%} < {SCREEN_CONFIDENCE_THRESHOLD:.0%})")
        print(" L'IA n'est pas sûre de l'état d'écran.")

    result = handle_state(state, confidence, models, img_pil)

    if result['buildings']:
        import random
        annotated = img_cv.copy()
        for b in result['buildings']:
            x1, y1, x2, y2 = b['bbox']
            random.seed(hash(b['class']) % 1000)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{b['class']} {b['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = "GameLoop_Result.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"\n Image annotée sauvegardée : {output_path}")

    print(f"\n{'='*60}")
    print(f" RÉSULTAT : état={state} | action={result['action']}")
    print(f"{'='*60}")

    return result


def run_live(models):
    """
    Live mode: autonomous ADB loop.
    Capture screen → Analyze → Decide → Act → Repeat.
    """
    print(f"\n{'='*60}")
    print(" MODE LIVE — Boucle autonome ADB")
    print(f"{'='*60}\n")

    print(f"Connecting to emulator: {ADB_DEVICE} ...")
    from clashai.adb import get_client
    client = get_client()
    connect_result = client.connect()
    print(f" -> {connect_result.stdout.strip()}")
    time.sleep(1)

    if not adb_check_connection():
        sys.exit(1)

    try:
        result = client.shell("wm size")
        print(f" Résolution : {result.stdout.strip()}")
    except:
        pass

    print("\nDémarrage de la boucle... (Ctrl+C pour arrêter)\n")

    frame_count = 0
    try:
        while True:
            frame_count += 1
            print(f"\n--- Frame #{frame_count} ---")

            img_pil = adb_screenshot()
            if img_pil is None:
                print("WARNING: Capture échouée, retry...")
                time.sleep(1)
                continue

            state, confidence = classify_screen(img_pil, models)
            print(f"État : {state} ({confidence:.1%})")

            result = handle_state(state, confidence, models, img_pil)

            action = result['action']

            if action == 'tap_attaquer':
                adb_tap(*BUTTONS['attaquer'])
                time.sleep(ADB_DELAY_NAVIGATION)

            elif action == 'tap_trouver_partie':
                adb_tap(*BUTTONS['trouver_partie'])
                time.sleep(ADB_DELAY_NAVIGATION)

            elif action == 'tap_lancer_attaque':
                adb_tap(*BUTTONS['lancer_attaque'])
                time.sleep(ADB_DELAY_MATCHMAKING)

            elif action == 'analyze_village':
                # Later: the RL agent will decide here (attack or skip)
                print(" → Analyse terminée. Skip pour l'instant...")
                adb_tap(*BUTTONS['suivant'])
                time.sleep(ADB_DELAY_NAVIGATION)

            elif action == 'tap_fin_combat':
                # Results → return to village
                adb_tap(*BUTTONS['fin_combat'])
                time.sleep(ADB_DELAY_NAVIGATION)

            elif action == 'close_menu':
                # Chat / boutique / popup → fermer
                adb_tap(*BUTTONS['retour'])
                time.sleep(ADB_DELAY_NAVIGATION)

            elif action == 'wait':
                # Chargement → attendre
                time.sleep(2)

            else:
                time.sleep(ADB_DELAY_SCREENSHOT)

    except KeyboardInterrupt:
        print(f"\n\nArrêt demandé. {frame_count} frames traitées.")
        print("Au revoir !")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ClashAI — Boucle principale de l'IA",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--test', type=str, nargs='+',
                        help='Mode test : analyse une ou plusieurs images\n'
                             'Ex: python scripts/game_loop.py --test test_img/screen1.png test_img/screen2.png')
    parser.add_argument('--live', action='store_true',
                        help='Mode live : boucle autonome avec ADB')

    args = parser.parse_args()

    # If no argument provided, show help
    if not args.test and not args.live:
        parser.print_help()
        print("\nExemples :")
        print(" python scripts/game_loop.py --test test_img/village.png")
        print(" python scripts/game_loop.py --test test_img/*.png")
        print(" python scripts/game_loop.py --live")
        sys.exit(0)

    # Load models
    models = load_models()

    if args.test:
        # Mode test : analyser chaque image
        for img_path in args.test:
            run_test(img_path, models)
    elif args.live:
        # Mode live : boucle ADB
        run_live(models)


if __name__ == "__main__":
    main()
