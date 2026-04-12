# scripts/rl/reward_reader.py
# Lit les étoiles (0-3) et le pourcentage de destruction
# sur l'écran de résultats d'attaque.
#
# Méthode :
#   - Étoiles : détection HSV (argenté = haute luminosité + faible saturation)
#   - Pourcentage : template matching par chiffre (0-9), exclusion du % détecté
#
# INDÉPENDANT DE LA RÉSOLUTION : fonctionne en 1920x1080, 2000x923, etc.
# Plus besoin de star_earned.png — les étoiles sont détectées par couleur.
#
# Setup (une seule fois) :
#   Mettre les templates 0.png à 9.png + pct.png dans reward_templates/digits/
#
# Usage :
#   results = read_attack_results()
#   print(results['stars'], results['percentage'], results['reward'])

import os
import sys
import subprocess
import io

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

from clashai.paths import REWARD_TEMPLATES_DIR, REWARD_DIGITS_DIR, DEBUG_DIR

TEMPLATES_DIR = REWARD_TEMPLATES_DIR
DIGITS_DIR = REWARD_DIGITS_DIR

# Seuils pour le digit matching
DIGIT_MATCH_THRESHOLD = 0.60
PCT_MATCH_THRESHOLD = 0.50

# Seuils pour les étoiles (HSV)
STAR_MIN_AREA = 1000       # Taille min d'une étoile en pixels
STAR_MAX_ASPECT = 2.5      # Ratio max largeur/hauteur
STAR_SATURATION_MAX = 60   # Saturation max (argenté = pas coloré)
STAR_VALUE_MIN = 180       # Luminosité min (argenté = lumineux)


# =============================================================================
#                         FONCTIONS ADB
# =============================================================================

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
#                     ISOLATION DU VERT (Partagée)
# =============================================================================

def isolate_green(img_bgr, green_thresh=20, bright_thresh=70):
    """
    Isole le texte vert de CoC dans une image BGR.
    Retourne un masque binaire (255 = vert, 0 = fond).
    """
    b, g, r = cv2.split(img_bgr)
    green_diff = cv2.subtract(g, cv2.max(r, b))
    _, mask_diff = cv2.threshold(green_diff, green_thresh, 255, cv2.THRESH_BINARY)
    _, mask_bright = cv2.threshold(g, bright_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_diff, mask_bright)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# =============================================================================
#                    LECTURE DES ÉTOILES (HSV)
# =============================================================================

def count_stars(img_cv, debug=False):
    """
    Compte les étoiles gagnées par filtrage HSV.

    Logique :
    - Étoiles gagnées = argenté/blanc = haute luminosité + faible saturation
    - Étoiles perdues = noires/sombres (invisibles au filtre)
    - Le bandeau doré "Victoire" est éliminé car il est saturé (doré ≠ argenté)

    Indépendant de la résolution : les zones sont en pourcentage de l'image.
    Plus besoin de template star_earned.png.
    """
    h, w = img_cv.shape[:2]

    # Zone des étoiles : tiers supérieur, centre de l'image
    sy1 = int(h * 0.03)
    sy2 = int(h * 0.35)
    sx1 = int(w * 0.25)
    sx2 = int(w * 0.65)

    region = img_cv[sy1:sy2, sx1:sx2]
    rh, rw = region.shape[:2]

    if debug:
        debug_dir = DEBUG_DIR
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, 'star_region.png'), region)

    # Filtrage HSV : argenté = saturation faible + valeur haute
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, STAR_VALUE_MIN), (180, STAR_SATURATION_MAX, 255))

    # Nettoyage morphologique
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Composantes connectées
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    # Filtrer : les étoiles sont grandes, dans le haut, et ~carrées
    star_candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cy = centroids[i][1]
        cx = centroids[i][0]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area > STAR_MIN_AREA and cy < rh * 0.70:
            aspect = bw / bh if bh > 0 else 99
            if 1.0 / STAR_MAX_ASPECT < aspect < STAR_MAX_ASPECT:
                star_candidates.append((cx, cy, area, bw, bh))

    # NMS spatial : éliminer les détections trop proches
    star_candidates.sort(key=lambda c: -c[2])
    kept = []
    for sc in star_candidates:
        cx = sc[0]
        is_dup = any(abs(cx - k[0]) < rw * 0.10 for k in kept)
        if not is_dup:
            kept.append(sc)

    stars = min(len(kept), 3)

    if debug:
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for cx, cy, area, bw, bh in kept:
            cv2.circle(debug_img, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            cv2.putText(debug_img, f"a={area}", (int(cx) - 20, int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, 'star_mask.png'), debug_img)
        print(f"   ⭐ Étoiles : {stars} ({len(star_candidates)} candidats, {len(kept)} après NMS)")

    return stars


# =============================================================================
#         DÉTECTION DYNAMIQUE DE LA ZONE POURCENTAGE
# =============================================================================

def find_pct_region(img_cv):
    """
    Trouve dynamiquement la zone du pourcentage vert.
    Cherche le gros texte vert dans le tiers supérieur de l'image.
    Indépendant de la résolution.

    Returns:
        (x1, y1, x2, y2) en coordonnées absolues, ou None si non trouvé.
    """
    h, w = img_cv.shape[:2]

    # Zone de recherche : tiers supérieur, tiers central
    search_y1 = 0
    search_y2 = h // 3
    search_x1 = w // 3
    search_x2 = 2 * w // 3

    search_region = img_cv[search_y1:search_y2, search_x1:search_x2]

    # Isolation du vert avec seuils stricts (gros texte seulement)
    mask = isolate_green(search_region, green_thresh=25, bright_thresh=100)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Composantes connectées
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    # Filtrer les composantes : chiffres = assez grands, pas trop larges
    digit_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 80:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if bh > 10 and bw < 60:
                digit_components.append((x, y, bw, bh, area))

    if not digit_components:
        return None

    # Grouper les composantes proches en Y (même ligne de texte)
    median_y = np.median([c[1] for c in digit_components])
    close = [c for c in digit_components if abs(c[1] - median_y) < 15]
    if not close:
        close = digit_components

    # Bounding box englobant + padding
    x1 = min(c[0] for c in close)
    y1 = min(c[1] for c in close)
    x2 = max(c[0] + c[2] for c in close)
    y2 = max(c[1] + c[3] for c in close)

    pad = 5
    return (search_x1 + x1 - pad, search_y1 + y1 - pad,
            search_x1 + x2 + pad, search_y1 + y2 + pad)


# =============================================================================
#              LECTURE DU POURCENTAGE (Digit Template Matching)
# =============================================================================

def load_digit_templates():
    """
    Charge les templates de chiffres 0-9 et % depuis reward_templates/digits/.
    Retourne (digits_dict, pct_mask) où digits_dict = {int: mask} et
    pct_mask = masque du symbole %.
    """
    digit_templates = {}
    pct_mask = None

    if not os.path.exists(DIGITS_DIR):
        print(f"⚠️  Dossier digits introuvable : {DIGITS_DIR}")
        return digit_templates, pct_mask

    for digit in range(10):
        path = os.path.join(DIGITS_DIR, f'{digit}.png')
        if not os.path.exists(path):
            print(f"⚠️  Template manquant : {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        mask = isolate_green(img)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 1
            mask = mask[max(0, y - pad):y + h + pad * 2,
                        max(0, x - pad):x + w + pad * 2]

        digit_templates[digit] = mask

    # Charger le template %
    pct_path = os.path.join(DIGITS_DIR, 'pct.png')
    if os.path.exists(pct_path):
        pct_img = cv2.imread(pct_path)
        if pct_img is not None:
            pct_mask = isolate_green(pct_img)
            coords = cv2.findNonZero(pct_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                pad = 1
                pct_mask = pct_mask[max(0, y - pad):y + h + pad * 2,
                                    max(0, x - pad):x + w + pad * 2]

    if digit_templates:
        print(f"📦 {len(digit_templates)} digit templates chargés"
              f"{' + %' if pct_mask is not None else ''}")

    return digit_templates, pct_mask


def read_percentage(img_cv, debug=False):
    """
    Lit le pourcentage par digit template matching.

    Approche :
    1. Trouver dynamiquement la zone du texte vert (indépendant de la résolution)
    2. Localiser le symbole "%" pour délimiter la zone des chiffres
    3. Matcher les chiffres 0-9 uniquement à GAUCHE du %
    4. Lire de gauche à droite pour former le nombre
    """
    if debug:
        debug_dir = DEBUG_DIR
        os.makedirs(debug_dir, exist_ok=True)

    # 1. Trouver la zone du pourcentage dynamiquement
    pct_coords = find_pct_region(img_cv)
    if pct_coords is None:
        if debug:
            print("   ⚠️  Zone PCT non trouvée !")
        return -1

    rx1, ry1, rx2, ry2 = pct_coords
    region = img_cv[ry1:ry2, rx1:rx2]
    region_mask = isolate_green(region)
    region_h, region_w = region_mask.shape[:2]

    if debug:
        cv2.imwrite(os.path.join(debug_dir, 'pct_region.png'), region)
        cv2.imwrite(os.path.join(debug_dir, 'pct_green_mask.png'), region_mask)

    # Vérifier qu'il y a du contenu vert
    green_pixels = np.sum(region_mask > 0)
    if green_pixels < 30:
        if debug:
            print("   ⚠️  Trop peu de pixels verts dans la zone PCT")
        return -1

    # 2. Charger les templates
    digit_templates, pct_tmpl = load_digit_templates()
    if not digit_templates:
        print("   ❌ Pas de digit templates disponibles !")
        return -1

    # Hauteur du contenu vert
    coords = cv2.findNonZero(region_mask)
    if coords is None:
        return -1
    _, _, _, content_h = cv2.boundingRect(coords)

    # 3. Localiser le "%" pour délimiter la zone des chiffres
    digits_right_limit = region_w  # Par défaut : toute la largeur

    if pct_tmpl is not None:
        pct_h, pct_w = pct_tmpl.shape[:2]
        base_scale_pct = content_h / pct_h
        best_pct_conf = 0
        best_pct_x = region_w

        for s in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            scale = base_scale_pct * s
            new_h = int(pct_h * scale)
            new_w = int(pct_w * scale)
            if new_h > region_h or new_w > region_w or new_h < 5 or new_w < 3:
                continue

            resized = cv2.resize(pct_tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(region_mask, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_pct_conf:
                best_pct_conf = max_val
                best_pct_x = max_loc[0]

        if best_pct_conf >= PCT_MATCH_THRESHOLD:
            digits_right_limit = best_pct_x - 2

        if debug:
            print(f"   🔍 % détecté : conf={best_pct_conf:.2f}, x={best_pct_x}, "
                  f"limite chiffres={digits_right_limit}")

    # 4. Matcher les chiffres uniquement à gauche du %
    digit_region = region_mask[:, :max(1, digits_right_limit)]

    all_matches = []
    for digit, tmpl_mask in digit_templates.items():
        tmpl_h, tmpl_w = tmpl_mask.shape[:2]
        base_scale = content_h / tmpl_h

        for s in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            scale = base_scale * s
            new_h = max(5, int(tmpl_h * scale))
            new_w = max(3, int(tmpl_w * scale))

            if new_h > digit_region.shape[0] or new_w > digit_region.shape[1]:
                continue
            if new_h < 5 or new_w < 3:
                continue

            resized = cv2.resize(tmpl_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(digit_region, resized, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= DIGIT_MATCH_THRESHOLD)

            for my, mx in zip(locations[0], locations[1]):
                conf = result[my, mx]
                cx = mx + new_w // 2
                all_matches.append((cx, mx, mx + new_w, digit, conf, new_w, my, new_h))

    if not all_matches:
        if debug:
            print("   ⚠️  Aucun chiffre détecté")
        return -1

    # 5. NMS basé sur l'overlap des bounding boxes
    all_matches.sort(key=lambda m: -m[4])
    kept = []
    for match in all_matches:
        cx, x_left, x_right, digit, conf, mw, my, mh = match

        is_dup = False
        for existing in kept:
            ex_left, ex_right, emw = existing[1], existing[2], existing[5]
            overlap_start = max(x_left, ex_left)
            overlap_end = min(x_right, ex_right)
            overlap = max(0, overlap_end - overlap_start)
            min_width = min(mw, emw)

            if overlap > min_width * 0.4:
                is_dup = True
                break

        if not is_dup:
            kept.append(match)

    # 6. Lire de gauche à droite
    kept.sort(key=lambda m: m[0])
    digits_found = [m[3] for m in kept]

    if debug:
        debug_img = cv2.cvtColor(region_mask, cv2.COLOR_GRAY2BGR)
        # Ligne rouge = limite droite (début du %)
        if digits_right_limit < region_w:
            cv2.line(debug_img, (digits_right_limit, 0),
                     (digits_right_limit, region_h), (0, 0, 255), 1)
        for cx, x_left, x_right, digit, conf, mw, my, mh in kept:
            cv2.rectangle(debug_img, (x_left, my), (x_right, my + mh), (0, 255, 0), 1)
            cv2.putText(debug_img, f"{digit}({conf:.2f})", (x_left, max(12, my - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, 'pct_digit_matches.png'), debug_img)

        print(f"   🔍 Matches brut: {len(all_matches)}, après NMS: {len(kept)}")
        for cx, x_left, x_right, digit, conf, mw, my, mh in kept:
            print(f"      Digit {digit} à x={cx}, conf={conf:.3f}")

    if not digits_found:
        return -1

    # Limiter à 3 chiffres max
    if len(digits_found) > 3:
        digits_found = digits_found[:3]

    # Construire le nombre
    number = 0
    for d in digits_found:
        number = number * 10 + d

    # Validation : 0 à 100
    if number > 100:
        if len(digits_found) >= 3 and digits_found[:3] == [1, 0, 0]:
            number = 100
        elif len(digits_found) >= 2:
            n2 = digits_found[0] * 10 + digits_found[1]
            if 0 <= n2 <= 100:
                number = n2
                if debug:
                    print(f"   🧠 Correction : {digits_found} → {n2}")
            else:
                number = digits_found[0]
        else:
            number = digits_found[0]

    if debug:
        print(f"   📊 Pourcentage lu : {number}%")

    return number


def read_percentage_from_stars(stars):
    """Estimation du pourcentage basée sur les étoiles (fallback)."""
    estimates = {0: 15, 1: 55, 2: 75, 3: 100}
    return estimates.get(stars, 0)


# =============================================================================
#                    LECTURE COMPLÈTE
# =============================================================================

def read_attack_results(img_pil=None, debug=False):
    """Lit les résultats d'attaque complets avec correction logique."""
    if img_pil is None:
        img_pil = adb_screenshot()
        if img_pil is None:
            return {'stars': 0, 'percentage': 0, 'reward': 0, 'success': False}

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 1. Lire le pourcentage (digit template matching)
    percentage = read_percentage(img_cv, debug=debug)

    # 2. Compter les étoiles (HSV)
    stars = count_stars(img_cv, debug=debug)

    # Fallback si le template matching a raté
    if percentage < 0:
        print("   ⚠️  Digit matching échoué, estimation depuis les étoiles...")
        percentage = read_percentage_from_stars(stars)

    # ---------------------------------------------------------
    # 🧠 CORRECTION LOGIQUE
    # ---------------------------------------------------------

    # Fix OCR 6→1 : le template matching confond souvent le 6 avec le 1.
    # Résultat : 60-69% lu comme 10-19%. On détecte ce cas par croisement
    # avec les étoiles : 2★ nécessite au minimum 50%.
    # Pour 1★, 16% est techniquement possible (TH détruit à 16%) mais
    # statistiquement c'est presque toujours un 6X% mal lu.
    if 10 <= percentage <= 19:
        corrected_pct = percentage + 50  # 16 → 66, 13 → 63, etc.
        if stars >= 2:
            # 2★ + <50% = impossible → correction certaine
            print(f"   🧠 Fix OCR 6→1 : {percentage}% impossible avec {stars}★"
                  f" → corrigé en {corrected_pct}%")
            percentage = corrected_pct
        elif stars == 1 and corrected_pct <= 100:
            # 1★ + 1X% = suspect → correction probable
            print(f"   🧠 Fix OCR 6→1 : {percentage}% suspect avec {stars}★"
                  f" → corrigé en {corrected_pct}%")
            percentage = corrected_pct

    if percentage == 100:
        stars = 3
    elif 0 <= percentage < 100 and stars == 3:
        print("   🧠 Correction : Impossible 3 étoiles sans 100%. Réduction à 2.")
        stars = 2
    elif percentage >= 50 and stars == 0:
        print("   🧠 Correction : >= 50% garantit 1 étoile minimum.")
        stars = 1

    # Garde-fou final : 2★ nécessite ≥50%
    if stars >= 2 and percentage < 50:
        print(f"   🧠 Correction : {stars}★ mais {percentage}% → forcé à 50%")
        percentage = 50

    reward = calculate_reward(stars, percentage)

    return {
        'stars': stars,
        'percentage': percentage,
        'reward': reward,
        'success': True,
    }


def calculate_reward(stars, percentage):
    """Calcule la récompense pour l'agent RL."""
    reward = (stars * 100) + percentage

    if stars >= 1:
        reward += 50
    if stars == 0:
        reward -= 50
    if stars == 3 and percentage == 100:
        reward += 50

    return reward


# =============================================================================
#                 EXTRACTION DES TEMPLATES
# =============================================================================

def extract_result_screen():
    """Capture l'écran de résultats et sauvegarde les zones utiles."""
    print("📸 Extraction de l'écran de résultats...")
    print("   Assure-toi d'être sur l'écran de résultats d'attaque")
    print("   (avec les étoiles et 'Victoire' ou 'Défaite')\n")

    img_pil = adb_screenshot()
    if img_pil is None:
        print("❌ Impossible de capturer l'écran")
        return

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    full_path = os.path.join(TEMPLATES_DIR, '_screenshot_resultats.png')
    cv2.imwrite(full_path, img_cv)

    # Zone dynamique du pourcentage
    pct_coords = find_pct_region(img_cv)
    if pct_coords:
        x1, y1, x2, y2 = pct_coords
        pct_region = img_cv[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(TEMPLATES_DIR, '_zone_pourcentage.png'), pct_region)

    print(f"✅ Screenshots sauvegardés dans {TEMPLATES_DIR}/")
    print()
    print("📝 PROCHAINE ÉTAPE :")
    print("   Mettre les templates 0-9 + pct.png dans reward_templates/digits/")
    print("   Puis lancer --test pour vérifier")


# =============================================================================
#                            TEST
# =============================================================================

def test_reward_reader(image_path=None):
    """Test le reward reader."""
    print("🧪 Test du Reward Reader\n")

    if image_path and os.path.exists(image_path):
        print(f"   Image : {image_path}")
        img_pil = Image.open(image_path).convert("RGB")
    else:
        print("   Capture ADB en cours...")
        img_pil = adb_screenshot()
        if img_pil is None:
            print("❌ Impossible de capturer l'écran")
            return

    results = read_attack_results(img_pil, debug=True)

    print(f"\n{'=' * 40}")
    print(f"⭐ Étoiles : {results['stars']}/3")
    print(f"📊 Pourcentage : {results['percentage']}%")
    print(f"🏆 Récompense RL : {results['reward']}")
    print(f"{'=' * 40}")

    print("\n📁 Images de debug dans le dossier debug_reward/")


def test_digits_only(image_path=None):
    """Test uniquement la lecture des chiffres."""
    print("🔢 Test du Digit Template Matching\n")

    if image_path and os.path.exists(image_path):
        print(f"   Image : {image_path}")
        img_pil = Image.open(image_path).convert("RGB")
    else:
        print("   Capture ADB en cours...")
        img_pil = adb_screenshot()
        if img_pil is None:
            print("❌ Impossible de capturer l'écran")
            return

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    pct = read_percentage(img_cv, debug=True)
    print(f"\n📊 Résultat : {pct}%")


# =============================================================================
#                              MAIN
# =============================================================================

if __name__ == "__main__":
    if '--extract' in sys.argv:
        extract_result_screen()
    elif '--test' in sys.argv:
        img_path = None
        for arg in sys.argv[2:]:
            if os.path.exists(arg):
                img_path = arg
                break
        test_reward_reader(img_path)
    elif '--test-digits' in sys.argv:
        img_path = None
        for arg in sys.argv[2:]:
            if os.path.exists(arg):
                img_path = arg
                break
        test_digits_only(img_path)
    else:
        print("Reward Reader — Lecture des résultats d'attaque")
        print()
        print("Usage :")
        print("  python scripts/rl/reward_reader.py --extract      (capturer l'écran)")
        print("  python scripts/rl/reward_reader.py --test         (tester tout)")
        print("  python scripts/rl/reward_reader.py --test-digits  (tester les chiffres seuls)")
        print()
        print("Setup :")
        print("  1. Mets les templates 0-9 + pct.png dans reward_templates/digits/")
        print("  2. Lance --test pour vérifier")
        print()
        print("Note : Plus besoin de star_earned.png !")
        print("       Les étoiles sont détectées par couleur (HSV).")