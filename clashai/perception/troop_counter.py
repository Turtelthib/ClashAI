# scripts/rl/troop_count_reader.py
# Lit le nombre de troupes (x2, x6, x10) affiché sur chaque icône
# dans la barre d'attaque.
#
# Méthode :
#   1. TroopFinder donne la position de chaque troupe dans la barre
#   2. On crop la zone "xN" en haut à gauche de l'icône
#   3. On isole le texte blanc par seuillage
#   4. On lit les chiffres par template matching (mêmes templates que reward_reader)
#
# Usage :
#   from clashai.perception.troop_counter import read_troop_counts
#   counts = read_troop_counts(screenshot_pil, troop_finder)
#   # counts = {'golem': 2, 'sorcier': 6, 'sorciere': 10, ...}

import os
import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

from clashai.paths import REWARD_DIGITS_DIR

# Dossier des templates de chiffres (partagé avec reward_reader)
DIGITS_DIR = REWARD_DIGITS_DIR

# La zone "xN" est en haut à gauche de chaque icône dans la barre
# Offset relatif à la position du template match (centre de l'icône)
# Les icônes font ~80x80px dans la barre, le "xN" est en haut-gauche
COUNT_OFFSET_X = -35   # Décalage X depuis le centre de l'icône
COUNT_OFFSET_Y = -45   # Décalage Y depuis le centre (vers le haut)
COUNT_WIDTH = 45        # Largeur de la zone à cropper
COUNT_HEIGHT = 25       # Hauteur de la zone à cropper

# Seuil pour le texte blanc
WHITE_THRESHOLD = 200   # V > 200 = texte blanc

# Template matching
DIGIT_MATCH_THRESHOLD = 0.65


# =============================================================================
#                    CHARGEMENT DES TEMPLATES
# =============================================================================

_digit_templates = None

def _load_digit_templates():
    """Charge les templates de chiffres 0-9."""
    global _digit_templates
    if _digit_templates is not None:
        return _digit_templates

    _digit_templates = {}

    if not os.path.exists(DIGITS_DIR):
        print(f"⚠️  Dossier digits introuvable : {DIGITS_DIR}")
        return _digit_templates

    for digit in range(10):
        path = os.path.join(DIGITS_DIR, f'{digit}.png')
        if os.path.exists(path):
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                _digit_templates[digit] = tmpl

    return _digit_templates


# =============================================================================
#                    LECTURE DES COMPTEURS
# =============================================================================

def _read_count_from_region(region_bgr):
    """
    Lit un nombre (1-99) depuis une petite image contenant "xN" ou "xNN".

    Args:
        region_bgr: image BGR de la zone contenant le texte

    Returns:
        count: int ou None si non lisible
    """
    if region_bgr is None or region_bgr.size == 0:
        return None

    templates = _load_digit_templates()
    if not templates:
        return None

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)

    # Isoler le texte blanc (les chiffres sont blancs sur fond sombre)
    _, binary = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Upscale pour le template matching (les chiffres sont petits)
    scale = 2.0
    binary_large = cv2.resize(binary, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    # Chercher les chiffres par template matching
    found_digits = []  # (x_position, digit_value)

    for digit, tmpl in templates.items():
        # Essayer plusieurs tailles de template
        for tmpl_scale in [0.4, 0.5, 0.6, 0.7, 0.8]:
            h_t = max(1, int(tmpl.shape[0] * tmpl_scale))
            w_t = max(1, int(tmpl.shape[1] * tmpl_scale))

            if h_t >= binary_large.shape[0] or w_t >= binary_large.shape[1]:
                continue

            tmpl_resized = cv2.resize(tmpl, (w_t, h_t))

            # Binariser le template aussi
            _, tmpl_bin = cv2.threshold(tmpl_resized, 128, 255, cv2.THRESH_BINARY)

            result = cv2.matchTemplate(binary_large, tmpl_bin, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= DIGIT_MATCH_THRESHOLD)

            for my, mx in zip(locations[0], locations[1]):
                # Vérifier qu'on n'a pas déjà un chiffre à cette position
                duplicate = False
                for fx, fd in found_digits:
                    if abs(mx - fx) < w_t * 0.6:
                        duplicate = True
                        # Garder le meilleur match
                        if result[my, mx] > fd[1]:
                            found_digits.remove((fx, fd))
                            found_digits.append((mx, (digit, result[my, mx])))
                        break
                if not duplicate:
                    found_digits.append((mx, (digit, result[my, mx])))

    if not found_digits:
        return None

    # Trier par position X (gauche → droite)
    found_digits.sort(key=lambda x: x[0])

    # Construire le nombre
    number = 0
    for _, (digit, conf) in found_digits:
        number = number * 10 + digit

    # Sanity check
    if number < 1 or number > 99:
        return None

    return number


def read_troop_counts(screenshot_pil, troop_finder):
    """
    Lit le nombre de chaque troupe depuis la barre d'attaque.

    Args:
        screenshot_pil: PIL Image du screenshot complet
        troop_finder: TroopFinder avec les positions déjà mises à jour

    Returns:
        counts: dict {nom_troupe: nombre} pour les troupes détectées
    """
    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    counts = {}

    for name, (tx, ty, conf) in troop_finder.positions.items():
        # Convertir la position ADB en coordonnées image
        ix = int(tx * w / 1920)
        iy = int(ty * h / 1080)

        # Zone du "xN" en haut-gauche de l'icône
        x1 = ix + int(COUNT_OFFSET_X * w / 1920)
        y1 = iy + int(COUNT_OFFSET_Y * h / 1080)
        x2 = x1 + int(COUNT_WIDTH * w / 1920)
        y2 = y1 + int(COUNT_HEIGHT * h / 1080)

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        region = img_cv[y1:y2, x1:x2]
        count = _read_count_from_region(region)

        if count is not None:
            counts[name] = count

    return counts


# =============================================================================
#                            TEST
# =============================================================================

def test_reader(image_path=None):
    """Test le lecteur de compteurs."""
    from clashai.perception.troop_finder import TroopFinder

    print("🧪 Test du Troop Count Reader\n")

    if image_path:
        img_pil = Image.open(image_path).convert("RGB")
    else:
        from clashai.navigation.game_loop import adb_screenshot
        img_pil = adb_screenshot()
        if img_pil is None:
            print("❌ Impossible de capturer l'écran")
            return

    finder = TroopFinder()
    finder.update(img_pil)

    counts = read_troop_counts(img_pil, finder)

    print("\n📊 Compteurs lus :")
    for name, count in sorted(counts.items()):
        print(f"   {name}: x{count}")

    # Troupes détectées mais pas de compteur lu
    for name in finder.positions:
        if name not in counts:
            print(f"   {name}: ??? (non lu)")


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else None
    test_reader(img)
