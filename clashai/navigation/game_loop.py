# scripts/game_loop.py
# Main loop of the ClashAI agent
# Usage:
# Test mode (static images): python scripts/game_loop.py --test test_img/my_screen.png
# Live mode (ADB): python scripts/game_loop.py --live

import os
import sys
import json
import time
import argparse
import subprocess
import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from clashai.perception.screen_classifier import MyCustomCNN


# =============================================================================
# CONFIGURATION
# =============================================================================

from clashai.paths import PROJECT_ROOT, WEIGHTS_DIR
# WEIGHTS_DIR imported from clashai.paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seuils de confiance
SCREEN_CONFIDENCE_THRESHOLD = 0.60
BUILDING_CONFIDENCE_THRESHOLD = 0.50
YOLO_CONF = 0.25
YOLO_IOU = 0.50

# ADB delays (in seconds)
ADB_DELAY_TAP = 0.1
ADB_DELAY_SCREENSHOT = 0.2
ADB_DELAY_NAVIGATION = 1.0
ADB_DELAY_MATCHMAKING = 3.0


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models():
    """Loads the 3 models: screen CNN, YOLO, building CNN."""
    models = {}

    # --- 1) Screen state CNN ---
    print(" Loading screen state CNN...")
    screen_classes_path = os.path.join(WEIGHTS_DIR, 'screen_classes.json')
    screen_weights_path = os.path.join(WEIGHTS_DIR, 'screen_cnn.pth')

    if not os.path.exists(screen_classes_path) or not os.path.exists(screen_weights_path):
        print("ERROR: ERREUR : screen_cnn.pth ou screen_classes.json introuvable !")
        print("👉 Lancez d'abord 'python scripts/train_screen_cnn.py'")
        sys.exit(1)

    with open(screen_classes_path) as f:
        models['screen_classes'] = json.load(f)

    screen_cnn = MyCustomCNN(num_classes=len(models['screen_classes'])).to(DEVICE)
    screen_cnn.load_state_dict(torch.load(screen_weights_path, map_location=DEVICE))
    screen_cnn.eval()
    models['screen_cnn'] = screen_cnn
    print(f" {len(models['screen_classes'])} états chargés : {models['screen_classes']}")

    # --- 2) YOLO Detection ---
    print("Loading YOLO11...")
    yolo_path = os.path.join(WEIGHTS_DIR, 'best.pt')
    if not os.path.exists(yolo_path):
        # Fallback: look in runs/
        yolo_path = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'FinishedTrain', 'weights', 'best.pt')
    if not os.path.exists(yolo_path):
        print("ERROR: ERREUR : best.pt introuvable !")
        sys.exit(1)

    models['yolo'] = YOLO(yolo_path)
    print(" YOLO chargé")

    # --- 3) Building CNN ---
    print("Loading building CNN...")
    building_classes_path = os.path.join(WEIGHTS_DIR, 'classes.json')
    building_weights_path = os.path.join(WEIGHTS_DIR, 'building_cnn.pth')

    if not os.path.exists(building_classes_path) or not os.path.exists(building_weights_path):
        print("ERROR: ERREUR : building_cnn.pth ou classes.json introuvable !")
        sys.exit(1)

    with open(building_classes_path) as f:
        models['building_classes'] = json.load(f)

    building_cnn = MyCustomCNN(num_classes=len(models['building_classes'])).to(DEVICE)
    building_cnn.load_state_dict(torch.load(building_weights_path, map_location=DEVICE))
    building_cnn.eval()
    models['building_cnn'] = building_cnn
    print(f" {len(models['building_classes'])} building classes loaded")

    print(f"\nTous les modèles sont chargés sur {DEVICE}\n")
    return models


# =============================================================================
# TRANSFORMS
# =============================================================================

screen_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

building_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def classify_screen(img_pil, models):
    """
    Determines the current screen state.
    Returns (state, confidence).
    """
    tensor = screen_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = models['screen_cnn'](tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][idx].item()

    state = models['screen_classes'][idx]
    return state, confidence


def analyze_village(img_pil, models):
    """
    Detects and classifies all buildings in the image.
    Returns a list of dicts {class, confidence, bbox, center}.
    """
    # YOLO detection
    img_np = np.array(img_pil)
    results = models['yolo'].predict(img_np, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    
    buildings = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp
        h, w = img_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop and CNN classification
        crop = img_pil.crop((x1, y1, x2, y2))
        tensor = building_transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = models['building_cnn'](tensor)
            probs = torch.softmax(outputs, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][idx].item()

        label = models['building_classes'][idx]

        # Filter out useless classes and low confidence
        if label in ('useless', 'ignore'):
            continue
        if confidence < BUILDING_CONFIDENCE_THRESHOLD:
            continue

        buildings.append({
            'class': label,
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
        })

    return buildings


def get_village_summary(buildings):
    """Human-readable summary of detected buildings."""
    counts = {}
    for b in buildings:
        counts[b['class']] = counts.get(b['class'], 0) + 1

    # Sort by type: defenses first, then resources, then others
    defenses = ['hdv', 'tour_enfer_mono', 'tour_enfer_multiple', 'aigle_artilleur',
                'catapulte_erratique', 'arcX_sol', 'arcX_sol_air', 'monolithe',
                'tour_archere', 'canon', 'mortier', 'multi_mortier', 'tour_sorcier',
                'defense_antiaerienne', 'prop_air', 'tesla', 'canon_ricochet',
                'cracheur_feu', 'tour_runique_rage', 'tour_runique_poison',
                'tour_runique_invisible', 'tour_multi_equipe_rapide', 'tour_bombe',
                'tour_archere_multiple', 'tour_multi_equipe_lente', 'tour_archere_rapide',
                'canon_double', 'tour_vengeuse', 'super_tour_sorcier', 'gigabombe',
                'tour_runique_seisme', 'cabane_ouvrier_arme']

    ressources = ['reserve_or', 'reserve_elixir', 'reserve_noire', 'ressources']

    summary = {
        'total': len(buildings),
        'defenses': sum(counts.get(d, 0) for d in defenses),
        'ressources': sum(counts.get(r, 0) for r in ressources),
        'details': counts
    }
    return summary


def adb_check_connection():
    """Checks that ADB is connected to a device."""
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.replace('\r', '')
        lines = output.strip().split('\n')
        # Look for lines containing "device" but not "devices" (header)
        devices = [l.strip() for l in lines if '\tdevice' in l or ' device' in l]
        if devices:
            device_name = devices[0].split()[0]
            print(f"📱 ADB connecté : {device_name}")
            return True
        else:
            print("ERROR: Aucun appareil ADB détecté.")
            print(f" (sortie adb: {repr(result.stdout[:200])})")
            print("👉 Lancez le Developer Emulator et exécutez : adb connect localhost:6520")
            return False
    except FileNotFoundError:
        print("ERROR: ADB n'est pas installé ou pas dans le PATH.")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: ADB ne répond pas (timeout).")
        return False


def adb_screenshot():
    """Captures the screen via ADB and returns a PIL image.
    Uses raw format for better speed."""
    try:
        # PNG format (slower but more reliable)
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0 or len(result.stdout) < 100:
            return None
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception as e:
        print(f"WARNING: Erreur capture : {e}")
        return None


def adb_tap(x, y):
    """Performs a tap at position (x, y)."""
    subprocess.run(["adb", "shell", f"input tap {x} {y}"], timeout=5)
    time.sleep(ADB_DELAY_TAP)


def adb_swipe(x1, y1, x2, y2, duration_ms=300):
    """Performs a swipe."""
    subprocess.run(
        ["adb", "shell", f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"],
        timeout=5
    )
    time.sleep(ADB_DELAY_TAP)


def adb_key(keycode):
    """Envoie une touche (ex: KEYCODE_BACK)."""
    subprocess.run(["adb", "shell", f"input keyevent {keycode}"], timeout=5)
    time.sleep(ADB_DELAY_TAP)


# =============================================================================
# NAVIGATION LOGIC
# =============================================================================
# Coordinates calibrated for Google Play Games Developer Emulator
# Resolution: 1920x1080 (adb shell wm size)

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Button coordinates (calibrated on real captures)
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
        print("⚔ Préparation armée détectée → Lancement de l'attaque")

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
            print(f" ⚔ {summary['defenses']} défenses")
            print(f" 💰 {summary['ressources']} bâtiments de ressources")

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
        print("⏳ Chargement en cours...")

    elif state == 'chat_clan':
        result['action'] = 'close_menu'
        print("💬 Chat de clan détecté → Fermeture")

    elif state == 'menu_boutique':
        result['action'] = 'close_menu'
        print("🛒 Boutique détectée → Fermeture")

    else:
        result['action'] = 'unknown'
        print(f"❓ État inconnu : {state} ({confidence:.1%})")

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
        print(f"\n🖼 Image annotée sauvegardée : {output_path}")

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

    print("📱 Connexion à Google Play Games Developer Emulator...")
    connect_result = subprocess.run(["adb", "connect", "localhost:6520"],
                                    capture_output=True, text=True, timeout=5)
    print(f" → {connect_result.stdout.strip()}")
    time.sleep(1)

    if not adb_check_connection():
        sys.exit(1)

    try:
        result = subprocess.run(
            ["adb", "shell", "wm", "size"],
            capture_output=True, text=True, timeout=5
        )
        print(f"📐 Résolution : {result.stdout.strip()}")
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