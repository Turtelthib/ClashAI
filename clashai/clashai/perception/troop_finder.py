# scripts/rl/troop_finder.py
# Détecte et sélectionne les troupes dans la barre du bas via template matching.
# Robuste : fonctionne quel que soit l'ordre des slots, les héros manquants,
# les troupes d'événement, etc.
#
# v2 : Multi-scale matching + scroll de la barre si des troupes manquent
#
# Setup (une seule fois) :
#   1. python scripts/rl/troop_finder.py --extract
#      → Capture la barre de troupes et la sauvegarde
#   2. Ouvre troop_templates/_barre_complete.png dans un éditeur
#   3. Découpe chaque icône et sauvegarde avec le bon nom :
#      golem.png, pekka.png, sorcier.png, etc.
#
# Utilisation dans le code :
#   finder = TroopFinder()
#   finder.update(screenshot_pil)   # Analyse la barre actuelle
#   finder.select("golem")          # Clique sur le golem
#   finder.select("rage")           # Clique sur le sort de rage

import os
import sys
import subprocess
import io
import time

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

from clashai.paths import TROOP_TEMPLATES_DIR

TEMPLATES_DIR = TROOP_TEMPLATES_DIR

# Zone de la barre de troupes dans l'écran (1920x1080)
BAR_TOP = 950
BAR_BOTTOM = 1080
BAR_LEFT = 0
BAR_RIGHT = 1920

# Seuil de confiance pour le template matching
MATCH_THRESHOLD = 0.45  # Abaissé de 0.50 à 0.45 pour attraper plus de troupes

# Multi-scale : échelles à tester si le match direct échoue
MATCH_SCALES = [1.0, 0.9, 1.1, 0.85, 1.15]


# =============================================================================
#                         FONCTIONS ADB
# =============================================================================

def adb_tap(x, y, delay=0.05):
    """Tap ADB."""
    subprocess.run(["adb", "shell", f"input tap {x} {y}"],
                   capture_output=True, timeout=5)
    time.sleep(delay)


def adb_swipe(x1, y1, x2, y2, duration_ms=300):
    """Swipe ADB."""
    subprocess.run(
        ["adb", "shell", f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"],
        capture_output=True, timeout=5
    )
    time.sleep(0.3)


def adb_screenshot():
    """Capture l'écran et retourne une image PIL."""
    try:
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0 or len(result.stdout) < 100:
            return None
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception as e:
        print(f"⚠️  Erreur capture : {e}")
        return None


# =============================================================================
#                      TROOP FINDER
# =============================================================================

class TroopFinder:
    """
    Trouve et sélectionne les troupes dans la barre du bas
    en utilisant le template matching d'OpenCV.
    """

    def __init__(self, templates_dir=TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self.templates = {}       # nom → image template (numpy array)
        self.positions = {}       # nom → (x, y, confidence)
        self._load_templates()

    def _load_templates(self):
        """Charge tous les templates depuis le dossier troop_templates/."""
        if not os.path.exists(self.templates_dir):
            print(f"⚠️  Dossier templates introuvable : {self.templates_dir}")
            print("   Lancez d'abord : python scripts/rl/troop_finder.py --extract")
            return

        count = 0
        for filename in os.listdir(self.templates_dir):
            if not filename.endswith('.png'):
                continue
            if filename.startswith('_'):  # Ignorer _barre_complete.png
                continue

            name = os.path.splitext(filename)[0]
            path = os.path.join(self.templates_dir, filename)
            template = cv2.imread(path)

            if template is not None:
                self.templates[name] = template
                count += 1

        if count > 0:
            print(f"📦 {count} templates de troupes chargés : {sorted(self.templates.keys())}")
        else:
            print(f"⚠️  Aucun template trouvé dans {self.templates_dir}")

    def _match_template_multiscale(self, bar_region, template):
        """
        Template matching multi-échelle.
        Retourne (max_val, max_loc, best_scale) ou (0, None, None) si rien.
        """
        best_val = 0
        best_loc = None
        _best_scale = None
        best_tw = 0
        best_th = 0

        for scale in MATCH_SCALES:
            th, tw = template.shape[:2]
            new_h = int(th * scale)
            new_w = int(tw * scale)

            if new_h > bar_region.shape[0] or new_w > bar_region.shape[1]:
                continue
            if new_h < 10 or new_w < 10:
                continue

            resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(bar_region, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                _best_scale = scale
                best_tw = new_w
                best_th = new_h

        return best_val, best_loc, best_tw, best_th

    def update(self, screenshot_pil):
        """
        Analyse la barre de troupes actuelle et trouve la position de chaque troupe.
        Utilise le multi-scale matching pour plus de robustesse.

        Args:
            screenshot_pil: image PIL de l'écran complet
        """
        screen = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        bar_region = screen[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]

        self.positions = {}

        for name, template in self.templates.items():
            best_val, best_loc, best_tw, best_th = self._match_template_multiscale(
                bar_region, template)

            if best_val >= MATCH_THRESHOLD and best_loc is not None:
                match_x = BAR_LEFT + best_loc[0] + best_tw // 2
                match_y = BAR_TOP + best_loc[1] + best_th // 2
                self.positions[name] = (match_x, match_y, best_val)

        found = len(self.positions)
        total = len(self.templates)
        print(f"🔍 Troupes détectées : {found}/{total}")

        for name, (x, y, conf) in sorted(self.positions.items(), key=lambda item: item[1][0]):
            print(f"   {name:<25s} → ({x:4d}, {y:4d})  conf: {conf:.2f}")

        # Lister les troupes manquantes
        missing = set(self.templates.keys()) - set(self.positions.keys())
        if missing:
            print(f"   ⚠️  Non trouvées : {sorted(missing)}")

    def update_with_scroll(self, scroll_attempts=2):
        """
        Comme update() mais scrolle la barre si des troupes manquent.
        Utile quand l'armée a beaucoup de types différents.

        Args:
            scroll_attempts: nombre de scrolls à essayer
        """
        # Premier scan sans scroll
        img = adb_screenshot()
        if img is None:
            return
        self.update(img)

        # Si toutes les troupes sont trouvées, on arrête
        if len(self.positions) >= len(self.templates):
            return

        missing = set(self.templates.keys()) - set(self.positions.keys())
        if not missing:
            return

        # Scroller la barre vers la droite et rescanner
        for attempt in range(scroll_attempts):
            print(f"   📜 Scroll de la barre (tentative {attempt+1})...")
            # Swipe de droite à gauche dans la barre pour voir les troupes cachées
            adb_swipe(1400, 1020, 600, 1020, 300)
            time.sleep(0.5)

            img = adb_screenshot()
            if img is None:
                continue

            screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bar_region = screen[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]

            # Scanner uniquement les troupes manquantes
            newly_found = 0
            for name in list(missing):
                if name not in self.templates:
                    continue

                best_val, best_loc, best_tw, best_th = self._match_template_multiscale(
                    bar_region, self.templates[name])

                if best_val >= MATCH_THRESHOLD and best_loc is not None:
                    match_x = BAR_LEFT + best_loc[0] + best_tw // 2
                    match_y = BAR_TOP + best_loc[1] + best_th // 2
                    self.positions[name] = (match_x, match_y, best_val)
                    missing.discard(name)
                    newly_found += 1
                    print(f"   ✅ Trouvé après scroll : {name} ({best_val:.2f})")

            if not missing:
                break

        # Rescroller au début pour remettre la barre en position initiale
        if scroll_attempts > 0:
            adb_swipe(600, 1020, 1400, 1020, 300)
            time.sleep(0.3)

        print(f"🔍 Total après scroll : {len(self.positions)}/{len(self.templates)}")

    def select(self, troop_name, delay=0.15):
        """
        Sélectionne une troupe en cliquant sur sa position.

        Returns:
            True si trouvée et cliquée, False sinon.
        """
        if troop_name not in self.positions:
            # Ne pas spammer les warnings
            return False

        x, y, conf = self.positions[troop_name]
        adb_tap(x, y, delay=delay)
        return True

    def get_position(self, troop_name):
        """Retourne (x, y) d'une troupe ou None."""
        if troop_name in self.positions:
            x, y, _ = self.positions[troop_name]
            return (x, y)
        return None

    def is_available(self, troop_name):
        """Vérifie si une troupe est visible dans la barre."""
        return troop_name in self.positions


# =============================================================================
#                 EXTRACTION DES TEMPLATES
# =============================================================================

def extract_bar():
    """
    Capture la barre de troupes et la sauvegarde.
    L'utilisateur découpe ensuite les icônes manuellement.
    """
    print("📸 Extraction de la barre de troupes...")
    print("   Assure-toi d'être sur l'écran d'un village ennemi")
    print("   (avec la barre de troupes visible en bas)\n")

    img_pil = adb_screenshot()
    if img_pil is None:
        print("❌ Impossible de capturer l'écran")
        return

    screen = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    bar = screen[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]

    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    bar_path = os.path.join(TEMPLATES_DIR, '_barre_complete.png')
    cv2.imwrite(bar_path, bar)

    full_path = os.path.join(TEMPLATES_DIR, '_screenshot_complet.png')
    cv2.imwrite(full_path, screen)

    print(f"✅ Barre sauvegardée : {bar_path}")
    print(f"✅ Screenshot complet : {full_path}")
    print()
    print("📝 PROCHAINE ÉTAPE :")
    print(f"   1. Ouvre {bar_path} dans un éditeur d'images")
    print("   2. Découpe chaque icône de troupe séparément")
    print(f"   3. Sauvegarde chaque icône dans {TEMPLATES_DIR}/ avec le bon nom :")
    print("      golem.png, pekka.png, sorcier.png, sorciere.png,")
    print("      archere.png, lance_buche.png, roi.png, reine.png,")
    print("      grand_gardien.png, championne.png, rage.png, soin.png, gel.png")
    print("   4. Supprime les fichiers commençant par _ (référence)")


def auto_crop_bar():
    """Découpe automatiquement la barre en slots réguliers."""
    bar_path = os.path.join(TEMPLATES_DIR, '_barre_complete.png')
    if not os.path.exists(bar_path):
        print("❌ Barre non trouvée. Lance --extract d'abord.")
        return

    bar = cv2.imread(bar_path)
    h, w = bar.shape[:2]

    start_x = 115
    spacing = 85
    icon_w = 75

    print(f"✂️  Découpage automatique de la barre ({w}x{h})")

    slot = 0
    x = start_x
    while x + icon_w < w:
        icon = bar[:, x:x + icon_w]
        if icon.mean() > 20:
            filename = f"slot_{slot:02d}.png"
            path = os.path.join(TEMPLATES_DIR, filename)
            cv2.imwrite(path, icon)
            print(f"   Slot {slot:2d} → {filename} (x={x})")
            slot += 1
        x += spacing

    print(f"\n✅ {slot} slots extraits.")
    print("   Renomme chaque slot_XX.png avec le vrai nom de la troupe.")


# =============================================================================
#                              TEST
# =============================================================================

def test_finder():
    """Test le TroopFinder sur un screenshot actuel."""
    print("🧪 Test du TroopFinder...\n")

    finder = TroopFinder()
    if not finder.templates:
        print("❌ Pas de templates. Lance --extract et découpe les icônes d'abord.")
        return

    img_pil = adb_screenshot()
    if img_pil is None:
        print("❌ Impossible de capturer l'écran")
        return

    # D'abord un scan normal
    finder.update(img_pil)

    # Puis avec scroll si des troupes manquent
    if len(finder.positions) < len(finder.templates):
        print("\n📜 Tentative avec scroll de la barre...")
        finder.update_with_scroll()

    if not finder.positions:
        print("\n❌ Aucune troupe détectée.")
        return

    print("\n🎯 Test de sélection (Entrée pour cliquer, 'q' pour quitter) :")
    for name in sorted(finder.positions.keys()):
        response = input(f"   Sélectionner '{name}' ? [Entrée/q] ")
        if response.lower() == 'q':
            break
        finder.select(name)
        print(f"   → Tap envoyé sur {name}")


# =============================================================================
#                              MAIN
# =============================================================================

if __name__ == "__main__":
    if '--extract' in sys.argv:
        extract_bar()
    elif '--auto-crop' in sys.argv:
        auto_crop_bar()
    elif '--test' in sys.argv:
        test_finder()
    else:
        print("TroopFinder — Détection visuelle des troupes dans la barre")
        print()
        print("Usage :")
        print("  python scripts/rl/troop_finder.py --extract     Capturer la barre de troupes")
        print("  python scripts/rl/troop_finder.py --auto-crop   Découper automatiquement en slots")
        print("  python scripts/rl/troop_finder.py --test        Tester la détection")