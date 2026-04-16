# clashai/social/clan_castle.py
# Gestion automatique du château de clan pour ClashAI V4.1.
#
# Demande des troupes de renfort avant chaque attaque,
# comme tout joueur humain le ferait.
#
# Flow :
#   1. YOLO bâtiments localise le château de clan dans le village
#   2. Check si "PLEIN" est affiché au-dessus
#   3. Si pas plein : tap CC → template matching bouton "Demande"
#      dans la barre du bas → tap "Envoyer" (position calibrée)
#   4. Cooldown 15 min entre chaque demande
#
# Setup :
#   1. uv run python -m clashai.social.clan_castle --capture
#      → Capture la barre et crop le bouton "Demande"
#      → Sauvegarder dans templates/clan_castle/request.png
#   2. uv run python -m clashai.navigation.calibrate_ui
#      → Calibrer la position cdc_confirmation (bouton vert "Envoyer")
#
# Usage :
#   manager = ClanCastleManager(building_detector)
#   manager.request_if_needed(screenshot_fn, tap_fn)

import os
import time

import cv2
import numpy as np
from PIL import Image

from clashai.paths import PROJECT_ROOT


# =============================================================================
#                         CONFIGURATION
# =============================================================================

# Dossier des templates
CC_TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates', 'clan_castle')

# Cooldown entre deux demandes (secondes)
REQUEST_COOLDOWN = 15 * 60  # 15 minutes

# Délais ADB (secondes)
DELAY_AFTER_TAP_CC = 1.5       # Attendre la barre d'options
DELAY_AFTER_TAP_REQUEST = 1.0  # Attendre le popup "Demander des renforts"
DELAY_AFTER_CONFIRM = 0.8      # Attendre après "Envoyer"
DELAY_CLOSE = 0.5

# Template matching (barre du bas — position instable)
MATCH_THRESHOLD = 0.60
MATCH_SCALES = [1.0, 0.9, 1.1, 0.85, 1.15]

# Zone de la barre d'options du bâtiment (bas de l'écran)
BAR_TOP = 880
BAR_BOTTOM = 1080
BAR_LEFT = 0
BAR_RIGHT = 1920

# Zone au-dessus du château pour détecter "PLEIN"
FULL_TEXT_OFFSET_Y = -80
FULL_TEXT_ZONE_W = 140
FULL_TEXT_ZONE_H = 40
FULL_TEXT_WHITE_THRESHOLD = 200

# Fallback position pour "Envoyer" (si pas calibré)
# Estimé depuis le screenshot : bouton vert centré-droit du popup
FALLBACK_CONFIRM_POS = (850, 575)


# =============================================================================
#                    TEMPLATE MATCHING
# =============================================================================

def _load_template(name):
    """Charge un template depuis templates/clan_castle/."""
    for ext in ('.png', '.jpg', '.bmp'):
        path = os.path.join(CC_TEMPLATES_DIR, f'{name}{ext}')
        if os.path.exists(path):
            tmpl = cv2.imread(path)
            if tmpl is not None:
                return tmpl
    return None


def _find_template(screenshot_cv, template_cv, region=None,
                   threshold=MATCH_THRESHOLD):
    """
    Cherche un template dans un screenshot avec multi-scale matching.

    Returns:
        (x, y, confidence) position ADB du centre, ou None
    """
    if region:
        top, bottom, left, right = region
        search_area = screenshot_cv[top:bottom, left:right]
        offset_x, offset_y = left, top
    else:
        search_area = screenshot_cv
        offset_x, offset_y = 0, 0

    search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_cv, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]

    best_val = 0
    best_loc = None
    best_scale = 1.0

    for scale in MATCH_SCALES:
        new_w = int(tw * scale)
        new_h = int(th * scale)
        if new_w < 10 or new_h < 10:
            continue
        tmpl_scaled = cv2.resize(tmpl_gray, (new_w, new_h)) if scale != 1.0 else tmpl_gray

        sh, sw = search_gray.shape[:2]
        if new_h > sh or new_w > sw:
            continue

        result = cv2.matchTemplate(
            search_gray, tmpl_scaled, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale

    if best_val < threshold or best_loc is None:
        return None

    scaled_tw = int(tw * best_scale)
    scaled_th = int(th * best_scale)
    cx = best_loc[0] + scaled_tw // 2 + offset_x
    cy = best_loc[1] + scaled_th // 2 + offset_y

    return (cx, cy, best_val)


def _get_confirm_position():
    """
    Position du bouton "Envoyer" dans le popup de confirmation.
    Popup centré → position stable → calibrate_ui.
    """
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position('cdc_confirmation')
    except (ImportError, KeyError, FileNotFoundError):
        return FALLBACK_CONFIRM_POS


# =============================================================================
#                    CLAN CASTLE MANAGER
# =============================================================================

class ClanCastleManager:
    """
    Gère les demandes de troupes du château de clan.

    - YOLO bâtiments → localise le CC dans le village
    - Template matching → trouve le bouton "Demande" (barre instable)
    - calibrate_ui → bouton "Envoyer" (popup stable)
    - Cooldown 15 min entre chaque demande
    """

    def __init__(self, building_detector=None, verbose=True):
        self._building_detector = building_detector
        self.verbose = verbose
        self._last_request_time = 0
        self._total_requests = 0

        # Template pour le bouton "Demande" dans la barre du bas
        self._tmpl_request = _load_template('request')

        if verbose:
            status = "ok" if self._tmpl_request is not None else "MANQUANT"
            print(f"   🏰 CC Manager — template request: {status}")
            if self._tmpl_request is None:
                print(f"      → uv run python -m clashai.social.clan_castle --capture")

    # -----------------------------------------------------------------
    #  API principale
    # -----------------------------------------------------------------

    def request_if_needed(self, screenshot_fn, tap_fn):
        """Demande des troupes si cooldown passé et CC pas plein."""
        if self._tmpl_request is None:
            return False
        if not self._cooldown_ready():
            return False

        img = screenshot_fn()
        if img is None:
            return False

        cc_pos = self._find_clan_castle(img)
        if cc_pos is None:
            if self.verbose:
                print("   ⚠️ Château de clan non trouvé")
            return False

        if self._is_castle_full(img, cc_pos):
            if self.verbose:
                print("   🏰 CC PLEIN — pas de demande")
            return False

        return self._do_request(cc_pos, tap_fn, screenshot_fn)

    def time_until_next_request(self):
        elapsed = time.time() - self._last_request_time
        return max(0, REQUEST_COOLDOWN - elapsed)

    # -----------------------------------------------------------------
    #  Cooldown
    # -----------------------------------------------------------------

    def _cooldown_ready(self):
        if self._last_request_time == 0:
            return True
        elapsed = time.time() - self._last_request_time
        if elapsed >= REQUEST_COOLDOWN:
            return True
        if self.verbose:
            remaining = (REQUEST_COOLDOWN - elapsed) / 60
            print(f"   ⏳ CC cooldown: encore {remaining:.0f}min")
        return False

    # -----------------------------------------------------------------
    #  Localisation du château de clan (YOLO)
    # -----------------------------------------------------------------

    def _find_clan_castle(self, screenshot_pil):
        if self._building_detector is None:
            return None
        detections = self._building_detector.detect(screenshot_pil)
        for d in detections:
            if d.class_name == 'clan_castle':
                if self.verbose:
                    print(f"   🏰 CC à ({d.x}, {d.y}) conf={d.conf:.2f}")
                return (d.x, d.y)
        return None

    # -----------------------------------------------------------------
    #  Détection "PLEIN"
    # -----------------------------------------------------------------

    def _is_castle_full(self, screenshot_pil, cc_pos):
        """
        Vérifie si le texte "PLEIN" est affiché au-dessus du CC.
        En cas de doute, retourne False (on essaie de demander).
        """
        try:
            img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            cx, cy = cc_pos
            x1 = max(0, cx - FULL_TEXT_ZONE_W // 2)
            x2 = min(w, cx + FULL_TEXT_ZONE_W // 2)
            y1 = max(0, cy + FULL_TEXT_OFFSET_Y - FULL_TEXT_ZONE_H // 2)
            y2 = max(0, cy + FULL_TEXT_OFFSET_Y + FULL_TEXT_ZONE_H // 2)
            if y2 <= y1 or x2 <= x1:
                return False
            zone = img_cv[y1:y2, x1:x2]
            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            ratio = np.sum(gray > FULL_TEXT_WHITE_THRESHOLD) / max(gray.size, 1)
            if ratio > 0.15:
                if self.verbose:
                    print(f"   🔍 'PLEIN' détecté (ratio: {ratio:.1%})")
                return True
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Erreur détection PLEIN: {e}")
        return False

    # -----------------------------------------------------------------
    #  Exécution de la demande
    # -----------------------------------------------------------------

    def _do_request(self, cc_pos, tap_fn, screenshot_fn):
        """
        1. Tap CC → barre d'options
        2. Template matching → bouton "Demande" (barre instable)
        3. Tap "Envoyer" (popup stable → calibrate_ui)
        """
        if self.verbose:
            print(f"   🏰 Demande de troupes CC...")

        # 1. Tap sur le château de clan
        tap_fn(cc_pos[0], cc_pos[1])
        time.sleep(DELAY_AFTER_TAP_CC)

        # 2. Screenshot + template matching "Demande" dans la barre
        img = screenshot_fn()
        if img is None:
            self._close_menu(tap_fn)
            return False

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = _find_template(
            img_cv, self._tmpl_request,
            region=(BAR_TOP, BAR_BOTTOM, BAR_LEFT, BAR_RIGHT),
        )

        if result is None:
            if self.verbose:
                print("   ⚠️ Bouton 'Demande' non trouvé dans la barre")
            self._close_menu(tap_fn)
            return False

        req_x, req_y, conf = result
        if self.verbose:
            print(f"   🔍 'Demande' à ({req_x}, {req_y}) conf={conf:.2f}")

        # 3. Tap "Demande" → popup "Demander des renforts"
        tap_fn(req_x, req_y)
        time.sleep(DELAY_AFTER_TAP_REQUEST)

        # 4. Tap "Envoyer" (position calibrée — popup stable)
        confirm_pos = _get_confirm_position()
        tap_fn(confirm_pos[0], confirm_pos[1])
        time.sleep(DELAY_AFTER_CONFIRM)

        if self.verbose:
            print(f"   📤 'Envoyer' à ({confirm_pos[0]}, {confirm_pos[1]})")

        # 5. Fermer + cooldown
        self._close_menu(tap_fn)
        self._last_request_time = time.time()
        self._total_requests += 1

        if self.verbose:
            print(f"   ✅ Troupes demandées ! (total: {self._total_requests})")
        return True

    def _close_menu(self, tap_fn):
        tap_fn(960, 400)
        time.sleep(DELAY_CLOSE)


# =============================================================================
#                            SETUP & TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    import subprocess
    import io

    parser = argparse.ArgumentParser(description="ClashAI CC Manager")
    parser.add_argument('--capture', action='store_true',
                        help="Capture la barre du CC pour créer le template")
    parser.add_argument('--test', action='store_true',
                        help="Test le template matching sur l'écran actuel")
    args = parser.parse_args()

    if args.capture:
        print("📸 Capture de la barre du château de clan\n")
        print("   1. Tap sur ton château de clan en jeu")
        print("   2. Attends que la barre apparaisse en bas")
        input("\n   → Entrée quand c'est prêt...")

        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0:
            print("   ❌ Erreur ADB")
        else:
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            os.makedirs(CC_TEMPLATES_DIR, exist_ok=True)
            bar = img_cv[BAR_TOP:BAR_BOTTOM, BAR_LEFT:BAR_RIGHT]
            bar_path = os.path.join(CC_TEMPLATES_DIR, '_barre_complete.png')
            cv2.imwrite(bar_path, bar)
            full_path = os.path.join(CC_TEMPLATES_DIR, '_screenshot.png')
            cv2.imwrite(full_path, img_cv)

            print(f"\n   ✅ Barre → {bar_path}")
            print(f"   ✅ Screenshot → {full_path}")
            print(f"\n   Maintenant :")
            print(f"   1. Ouvre {bar_path}")
            print(f"   2. Crop le bouton 'Demande' → request.png")
            print(f"   3. Sauvegarde dans {CC_TEMPLATES_DIR}/")
            print(f"\n   Puis calibrer le bouton 'Envoyer' :")
            print(f"   uv run python -m clashai.navigation.calibrate_ui")
            print(f"   → ajouter cdc_confirmation (bouton vert 'Envoyer')")

    elif args.test:
        print("🧪 Test template matching CC\n")
        mgr = ClanCastleManager(building_detector=None)
        if mgr._tmpl_request is None:
            print("   ❌ Template 'request' manquant")
            print(f"   → uv run python -m clashai.social.clan_castle --capture")
        else:
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True, timeout=5
            )
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            match = _find_template(
                img_cv, mgr._tmpl_request,
                region=(BAR_TOP, BAR_BOTTOM, BAR_LEFT, BAR_RIGHT),
            )
            if match:
                print(f"   ✅ Bouton 'Demande' à ({match[0]}, {match[1]}) "
                      f"conf={match[2]:.3f}")
            else:
                print("   ❌ Bouton non trouvé (ouvre d'abord le menu CC)")

            confirm = _get_confirm_position()
            print(f"   📍 Position 'Envoyer' : {confirm}")

    else:
        print("🏰 ClashAI CC Manager")
        print("  --capture   Capture la barre pour le template")
        print("  --test      Test le template matching")
        print(f"\nTemplates : {CC_TEMPLATES_DIR}")
        req = _load_template('request')
        print(f"  request.png : {'✅' if req is not None else '❌ manquant'}")
        confirm = _get_confirm_position()
        is_fallback = confirm == FALLBACK_CONFIRM_POS
        tag = "⚠️ fallback" if is_fallback else "✅ calibré"
        print(f"  cdc_confirmation : {confirm} ({tag})")