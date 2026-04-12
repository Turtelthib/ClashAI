# scripts/rl/gdc_navigator.py
# Navigation automatique en Guerre de Clans (GdC) pour ClashAI.
#
# Ce module orchestre une attaque GdC complète :
#   1. Depuis le village → ouvrir le menu clan → onglet GdC
#   2. Passer sur la carte ennemie
#   3. Scroller jusqu'à la cible n°X
#   4. Sélectionner la cible → scouter → confirmer l'attaque
#   5. L'agent V3 prend le relais pour le combat
#   6. Retour au village après le combat
#
# Usage :
#   navigator = GdCNavigator(models)
#   success = navigator.attack_target(3)  # Attaquer le n°3 ennemi
#
# Usage avec l'agent V3 :
#   navigator = GdCNavigator(models)
#   success = navigator.navigate_to_target(3)  # Juste naviguer
#   if success:
#       # L'agent V3 gère l'attaque via environment_v3
#       env = ClashEnvV3(models)
#       obs, mask = env.reset()  # Reprend depuis phase_attaque
#       ...

import os
import time
import subprocess
import io

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

ADB_WIDTH = 1920
ADB_HEIGHT = 1080

# --- Positions des boutons UI ---
# Chargées dynamiquement depuis ui_positions.json
# Calibrées via : python scripts/rl/calibrate_ui.py
def _get_ui_pos(name):
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position(name)
    except ImportError:
        defaults = {
            'chat_open': (47, 400),
            'chat_close_tap': (1400, 400),
            'gdc_open': (47, 560),
            'gdc_war_ended_see_map': (960, 700),
            'gdc_enemy_map': (1700, 540),
            'gdc_ally_map': (200, 540),
            'gdc_attack_target': (900, 660),
            'gdc_village_next': (1050, 680),
            'gdc_village_prev': (100, 680),
            'gdc_return_home': (80, 780),
            'attack_button': (80, 830),
            'start_attack': (960, 700),
            'open_profil': (40, 50),
            'close_profil': (1270, 90),
            'close_menu': (1340, 95),
            'close_popup': (1300, 100),
            'return_home': (960, 800),
        }
        return defaults.get(name, (960, 400))

# Zone où les numéros de cibles ennemies apparaissent
# (la liste des ennemis avec leur #)
TARGET_LIST_ZONE = {
    'left': 100,
    'right': 1820,
    'top': 150,
    'bottom': 850,
}

# Positions Y approximatives des cibles visibles à l'écran
# (environ 5-6 cibles visibles à la fois dans la liste GdC)
VISIBLE_TARGETS_PER_SCREEN = 5

# Vitesse de scroll pour naviguer dans la liste
SCROLL_DISTANCE = 400  # pixels par swipe
SCROLL_DURATION = 300   # ms

# Temps d'attente entre les actions
WAIT_NAVIGATION = 1.5
WAIT_MENU_LOAD = 2.0
WAIT_SCROLL = 1.0
WAIT_TARGET_LOAD = 2.0
WAIT_MATCHMAKING = 4.0

MAX_RETRIES = 15


# =============================================================================
#                    FONCTIONS ADB
# =============================================================================

def _adb_screenshot():
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


def _adb_tap(x, y, delay=0.15):
    subprocess.run(["adb", "shell", f"input tap {x} {y}"],
                   capture_output=True, timeout=5)
    time.sleep(delay)


def _adb_swipe(x1, y1, x2, y2, duration_ms=300):
    subprocess.run(
        ["adb", "shell", f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"],
        capture_output=True, timeout=5
    )
    time.sleep(0.5)


# =============================================================================
#                    DÉTECTION DE NUMÉRO OCR
# =============================================================================

def _detect_target_numbers(screenshot_pil):
    """
    Détecte les numéros de cibles visibles sur l'écran GdC ennemi.
    
    Dans CoC, chaque ennemi a un numéro (#1, #2, ..., #50) affiché
    à côté de son nom dans la liste de guerre.
    
    Returns:
        targets: dict {numéro: (x_center, y_center)} des cibles visibles
    """
    try:
        from clashai.social.clan_chat_monitor import _init_ocr

    except ImportError:
        return {}

    img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    zone = img_cv[TARGET_LIST_ZONE['top']:TARGET_LIST_ZONE['bottom'],
                   TARGET_LIST_ZONE['left']:TARGET_LIST_ZONE['right']]

    engine, etype = _init_ocr()
    if engine is None:
        return {}

    targets = {}

    if etype == 'easyocr':
        results = engine.readtext(zone, paragraph=False)
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            # Chercher des numéros (#1, #2, 1., 2., etc.)
            import re
            match = re.search(r'#?(\d{1,2})', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 50:
                    # Position au centre de la bbox
                    cx = int((bbox[0][0] + bbox[2][0]) / 2) + TARGET_LIST_ZONE['left']
                    cy = int((bbox[0][1] + bbox[2][1]) / 2) + TARGET_LIST_ZONE['top']
                    targets[num] = (cx, cy)

    elif etype == 'tesseract':
        import pytesseract
        data = pytesseract.image_to_data(zone, output_type=pytesseract.Output.DICT)
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
            import re
            match = re.search(r'#?(\d{1,2})', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 50:
                    x = data['left'][i] + data['width'][i] // 2 + TARGET_LIST_ZONE['left']
                    y = data['top'][i] + data['height'][i] // 2 + TARGET_LIST_ZONE['top']
                    targets[num] = (x, y)

    return targets


# =============================================================================
#                    GDC NAVIGATOR
# =============================================================================

class GdCNavigator:
    """
    Navigue dans l'interface de Guerre de Clans et sélectionne une cible.
    """

    def __init__(self, models, verbose=True):
        self.models = models
        self.verbose = verbose

        # Imports de game_loop
        from clashai.navigation import game_loop as gl
        self._classify_screen = gl.classify_screen
        self._adb_screenshot_gl = gl.adb_screenshot

    def _get_screen_state(self):
        """Retourne (state, confidence, img_pil)."""
        img = self._adb_screenshot_gl()
        if img is None:
            return None, 0.0, None
        state, conf = self._classify_screen(img, self.models)
        return state, conf, img

    def _navigate_to_state(self, target, max_retries=MAX_RETRIES):
        """Navigation générique vers un état."""
        for attempt in range(max_retries):
            state, conf, img = self._get_screen_state()
            if state is None:
                time.sleep(1)
                continue

            if self.verbose and attempt % 3 == 0:
                print(f"   📍 GdC nav: {state} ({conf:.0%}) → cible: {target}")

            if state == target:
                return True, img

            # Navigation contextuelle
            if state == 'village_home':
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)
            elif state == 'chat_clan':
                # Fermer le chat puis ouvrir le clan
                _adb_tap(*_get_ui_pos('chat_close_tap'))
                time.sleep(0.5)
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)
            elif state == 'gdc_ally':
                if target == 'gdc_enemy':
                    _adb_tap(*_get_ui_pos('gdc_enemy_map'))
                    time.sleep(WAIT_NAVIGATION)
                elif target == 'village_home':
                    _adb_tap(*_get_ui_pos('gdc_return_home'))
                    time.sleep(WAIT_NAVIGATION)
                else:
                    _adb_tap(*_get_ui_pos('gdc_war_ended_see_map'))
                    time.sleep(WAIT_NAVIGATION)
            elif state == 'gdc_enemy':
                if target == 'village_home':
                    _adb_tap(*_get_ui_pos('gdc_return_home'))
                    time.sleep(WAIT_NAVIGATION)
                elif target == 'gdc_ally':
                    _adb_tap(*_get_ui_pos('gdc_ally_map'))
                    time.sleep(WAIT_NAVIGATION)
                elif target == 'phase_attaque':
                    return True, img
                else:
                    return True, img
            elif state == 'phase_attaque':
                return True, img
            elif state == 'resultats_attaque':
                _adb_tap(*_get_ui_pos('return_home'))
                time.sleep(WAIT_NAVIGATION)
            elif state == 'profil':
                _adb_tap(*_get_ui_pos('close_profil'))
                time.sleep(0.5)
                _adb_tap(1800, 500)  # Tap bord droit (safety)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'menu_boutique':
                _adb_tap(*_get_ui_pos('close_menu'))
                time.sleep(WAIT_NAVIGATION)
            elif state == 'popup_offre':
                _adb_tap(*_get_ui_pos('close_popup'))
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chargement':
                time.sleep(2)
            else:
                _adb_tap(960, 400)
                time.sleep(WAIT_NAVIGATION)

        return False, None

    def navigate_to_war_map(self):
        """
        Depuis n'importe quel écran, navigue jusqu'à la carte
        ennemie de la GdC.
        
        Gère l'écran "guerre terminée" qui apparaît quand la dernière
        GdC est finie — clique sur "Voir la carte" automatiquement.
        
        Returns:
            success: bool
        """
        if self.verbose:
            print("\n🗺️  Navigation vers la carte GdC ennemie...")

        # Étape 1 : Aller au village si pas déjà sur un écran GdC
        state, _, _ = self._get_screen_state()
        if state not in ('village_home', 'gdc_ally', 'gdc_enemy', 'gdc_ended'):
            success, _ = self._navigate_to_state('village_home')
            if not success:
                if self.verbose:
                    print("   ❌ Impossible de revenir au village")
                return False

        # Étape 2 : Ouvrir le menu GdC depuis le village
        state, _, _ = self._get_screen_state()
        if state == 'village_home':
            if self.verbose:
                print("   📋 Ouverture du menu GdC...")
            _adb_tap(*_get_ui_pos('gdc_open'))
            time.sleep(WAIT_MENU_LOAD)

        # Étape 3 : Gérer les écrans possibles après l'ouverture
        for attempt in range(MAX_RETRIES):
            state, conf, img = self._get_screen_state()
            
            if self.verbose and attempt % 2 == 0:
                print(f"   📍 GdC nav: {state} ({conf:.0%}) → cible: gdc_enemy")

            if state == 'gdc_enemy':
                if self.verbose:
                    print("   ✅ Carte ennemie GdC atteinte")
                return True

            elif state == 'gdc_ally':
                # Sur la carte alliée → basculer vers ennemis
                if self.verbose:
                    print("   🔄 Carte alliée → passage aux ennemis")
                _adb_tap(*_get_ui_pos('gdc_enemy_map'))
                time.sleep(WAIT_NAVIGATION)

            elif state == 'chat_clan':
                # Le chat s'est ouvert au lieu du menu GdC
                _adb_tap(*_get_ui_pos('chat_close_tap'))
                time.sleep(0.5)
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)

            elif state == 'village_home':
                # Retour au village inattendu → réessayer
                _adb_tap(*_get_ui_pos('gdc_open'))
                time.sleep(WAIT_MENU_LOAD)

            elif state == 'chargement':
                time.sleep(2)

            elif state == 'gdc_ended':
                # Écran "guerre terminée" → cliquer sur "Voir la carte"
                if self.verbose:
                    print("   📋 Guerre terminée → clic 'Voir la carte'")
                _adb_tap(*_get_ui_pos('gdc_war_ended_see_map'))
                time.sleep(WAIT_NAVIGATION)

            else:
                # État inconnu → tenter le bouton "Voir la carte" en fallback
                if self.verbose:
                    print(f"   ❓ État inconnu ({state} {conf:.0%}) "
                          f"→ tentative bouton 'Voir la carte'")
                _adb_tap(*_get_ui_pos('gdc_war_ended_see_map'))
                time.sleep(WAIT_NAVIGATION)

        if self.verbose:
            print("   ❌ Navigation GdC échouée après max retries")
        return False

    def select_target(self, target_number):
        """
        Sélectionne la cible n°X sur la carte ennemie.
        
        Méthode fiable sans OCR :
        1. Tape sur un village pour ouvrir un popup
        2. Appuie sur "précédent" (←) plein de fois → arrive au #1
        3. Appuie sur "suivant" (→) exactement (N-1) fois → arrive au #N
        
        Pas d'OCR = pas d'erreur de lecture.
        
        Args:
            target_number: int (1-50)
            
        Returns:
            success: bool
        """
        if self.verbose:
            print(f"\n🎯 Recherche de la cible #{target_number}...")

        # On doit être sur gdc_enemy
        state, _, _ = self._get_screen_state()
        if state != 'gdc_enemy':
            if self.verbose:
                print(f"   ⚠️  Pas sur la carte ennemie (état: {state})")
            return False

        # --- Étape 1 : Ouvrir un popup en tapant sur un village ---
        village_tap_positions = [
            (700, 450), (500, 400), (900, 500),
            (600, 350), (800, 550), (960, 400),
        ]

        popup_opened = False
        for tx, ty in village_tap_positions:
            _adb_tap(tx, ty)
            time.sleep(1.0)

            img = _adb_screenshot()
            if img is not None and self._check_attack_popup(img):
                popup_opened = True
                break

        if not popup_opened:
            if self.verbose:
                print("   ❌ Impossible d'ouvrir un popup de village")
            return False

        # --- Étape 2 : Aller au village #1 (tout à gauche) ---
        # Appuyer sur "précédent" suffisamment de fois
        # Max 30 villages en GdC classique, 15 en ligue
        max_prev = 30
        if self.verbose:
            print(f"   ⬅️  Retour au village #1 ({max_prev}x prev)...")
        
        for i in range(max_prev):
            _adb_tap(*_get_ui_pos('gdc_village_prev'))
            time.sleep(0.3)
        
        time.sleep(0.5)

        # --- Étape 3 : Avancer de (N-1) villages ---
        steps_needed = target_number - 1
        
        if steps_needed > 0:
            if self.verbose:
                print(f"   ➡️  Navigation : {steps_needed}x next → cible #{target_number}")
            
            for i in range(steps_needed):
                _adb_tap(*_get_ui_pos('gdc_village_next'))
                time.sleep(0.4)
                
                # Log de progression tous les 5 villages
                if self.verbose and (i + 1) % 5 == 0:
                    print(f"      #{i + 2}...")
        
        time.sleep(0.5)

        # Vérifier qu'on a toujours un popup ouvert
        img = _adb_screenshot()
        if img is not None and self._check_attack_popup(img):
            if self.verbose:
                print(f"   ✅ Cible #{target_number} sélectionnée !")
            return True
        else:
            if self.verbose:
                print("   ⚠️  Popup perdu après navigation")
            return False

    def _check_attack_popup(self, img_pil):
        """
        Vérifie si le popup de sélection de cible est affiché.
        Détecte le bouton vert "Attaquer" dans la moitié basse de l'écran.
        """
        import cv2
        import numpy as np

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        roi = img_cv[h // 2:, :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Bouton vert "Attaquer"
        mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
        green_pixels = cv2.countNonZero(mask)

        return green_pixels > 500

    def _read_popup_number(self, img_pil):
        """
        Lit le numéro de la cible dans le popup de sélection.
        
        Le popup affiche "3. NomDuJoueur" ou "3. ほりほり" en haut.
        On cherche un nombre au début d'une ligne dans la zone du popup.
        
        Returns:
            int ou None
        """
        import cv2
        import numpy as np
        import re

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # Zone serrée : juste la ligne titre du popup
        # Le titre "N. Joueur" apparaît à ~64-74% de la hauteur, centré
        popup_zone = img_cv[int(h * 0.64):int(h * 0.74),
                            int(w * 0.25):int(w * 0.60)]

        try:
            from clashai.social.clan_chat_monitor import _init_ocr
            engine, etype = _init_ocr()
            if engine is None:
                return None

            if etype == 'easyocr':
                results = engine.readtext(popup_zone, paragraph=False)
                for (bbox, text, conf) in results:
                    if conf < 0.2:
                        continue
                    text = text.strip()
                    # Patterns possibles :
                    # "3. ほりほり" → "3."
                    # "3.ほりほり" → "3."
                    # "#3" → "#3"
                    # "3 . nom" → "3"
                    # On cherche juste un nombre de 1-2 chiffres
                    match = re.match(r'#?(\d{1,2})', text)
                    if match:
                        num = int(match.group(1))
                        if 1 <= num <= 50:
                            return num

            elif etype == 'tesseract':
                import pytesseract
                text = pytesseract.image_to_string(popup_zone)
                for line in text.split('\n'):
                    line = line.strip()
                    match = re.match(r'#?(\d{1,2})', line)
                    if match:
                        num = int(match.group(1))
                        if 1 <= num <= 50:
                            return num

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Erreur OCR popup: {e}")

        return None

    def launch_attack(self):
        """
        Depuis l'écran avec le popup de cible, lance l'attaque.
        """
        if self.verbose:
            print("   ⚔️  Lancement de l'attaque GdC...")

        for attempt in range(15):
            # D'abord : vérifier l'état via CNN
            img = _adb_screenshot()
            state, conf, _ = self._get_screen_state()

            if self.verbose:
                print(f"   📍 Attaque: {state} ({conf:.0%})")

            if state == 'phase_attaque':
                if self.verbose:
                    print("   ✅ Phase d'attaque atteinte")
                return True

            elif state == 'prep_attaque':
                # Écran de préparation → cliquer sur le gros "Attaquer"
                if self.verbose:
                    print("   📋 Préparation → clic Attaquer")
                _adb_tap(*_get_ui_pos('start_attack'))
                time.sleep(WAIT_MATCHMAKING)

            elif state == 'gdc_enemy':
                # Encore sur la carte → vérifier si popup visible
                if img is not None and self._check_attack_popup(img):
                    atk_pos = _get_ui_pos('gdc_attack_target')
                    if self.verbose:
                        print(f"   📍 Popup visible → clic Attaquer "
                              f"à ({atk_pos[0]}, {atk_pos[1]})")
                    _adb_tap(*atk_pos)
                    time.sleep(WAIT_NAVIGATION)
                else:
                    if self.verbose:
                        print("   ⚠️  Pas de popup, tap village")
                    _adb_tap(700, 450)
                    time.sleep(1.0)

            elif state == 'chargement':
                time.sleep(2)

            else:
                if self.verbose:
                    print(f"   ❓ État {state}, tap confirmation")
                _adb_tap(960, 600)
                time.sleep(WAIT_NAVIGATION)

        if self.verbose:
            print("   ❌ Impossible d'atteindre la phase d'attaque")
        return False

        if self.verbose:
            print("   ❌ Impossible d'atteindre la phase d'attaque")
        return False

    def attack_target(self, target_number):
        """
        Séquence complète : naviguer → sélectionner → attaquer.
        
        Note : cette méthode amène jusqu'à phase_attaque.
        L'agent V3 (environment_v3) doit prendre le relais pour
        le deploy + combat.
        
        Args:
            target_number: int (1-50)
            
        Returns:
            success: bool (True si prêt à attaquer)
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"  GdC : Attaque cible #{target_number}")
            print(f"{'='*50}")

        # 1. Naviguer vers la carte ennemie
        if not self.navigate_to_war_map():
            return False

        # 2. Sélectionner la cible
        if not self.select_target(target_number):
            # Retour au village en cas d'échec
            self._navigate_to_state('village_home')
            return False

        # 3. Lancer l'attaque
        if not self.launch_attack():
            self._navigate_to_state('village_home')
            return False

        if self.verbose:
            print(f"\n   ✅ Prêt à attaquer la cible #{target_number} !")
            print("   → L'agent V3 prend le relais pour le combat")

        return True

    def return_to_village(self):
        """Retour au village après une attaque GdC."""
        return self._navigate_to_state('village_home')


# =============================================================================
#                    ORCHESTRATEUR GDC
# =============================================================================

class GdCOrchestrator:
    """
    Orchestre une attaque GdC complète :
    chat monitor → navigation GdC → agent V3 → retour village.
    
    Usage :
        orchestrator = GdCOrchestrator(models)
        orchestrator.run()  # Boucle infinie : surveille le chat et attaque
    """

    def __init__(self, models, bot_name='mini_pekka', verbose=True):
        self.models = models
        self.verbose = verbose

        from clashai.social.clan_chat_monitor import ClanChatMonitor
        self._chat_monitor = ClanChatMonitor(bot_name=bot_name, verbose=verbose)
        self._navigator = GdCNavigator(models, verbose=verbose)

    def handle_command(self, command):
        """
        Exécute une commande reçue du chat.
        
        Args:
            command: dict {'type': 'attack', 'target': 3, ...}
        """
        if command['type'] == 'attack':
            target = command['target']
            if self.verbose:
                print(f"\n🎯 Commande reçue : attaquer #{target} en GdC")

            success = self._navigator.attack_target(target)

            if success:
                # On est en phase_attaque → lancer l'agent V3
                self._run_attack()
            else:
                if self.verbose:
                    print(f"   ❌ Navigation vers cible #{target} échouée")

            # Retour au village dans tous les cas
            self._navigator.return_to_village()

        elif command['type'] == 'status':
            if self.verbose:
                print("   📊 Status demandé (pas d'action)")

    def _run_attack(self):
        """
        Lance l'agent V3 pour une attaque depuis phase_attaque.
        """
        if self.verbose:
            print("\n   🤖 Lancement de l'attaque V3...")

        try:
            from clashai.combat.environment import ClashEnvV3
            from clashai.combat.agent import PPOAgentV3

            env = ClashEnvV3(models=self.models, verbose=self.verbose)

            # L'agent : charger le meilleur checkpoint
            agent = PPOAgentV3()
            weights_dir = os.path.join(
                os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )),
                'weights', 'rl'
            )
            best_path = os.path.join(weights_dir, 'agent_v3_best.pth')
            checkpoint_path = os.path.join(weights_dir, 'agent_v3_checkpoint.pth')

            if os.path.exists(best_path):
                agent.load(best_path)
            elif os.path.exists(checkpoint_path):
                agent.load(checkpoint_path)
            else:
                if self.verbose:
                    print("   ⚠️  Pas de checkpoint, mode heuristique")

            # Reset (reprend depuis phase_attaque)
            obs, mask = env.reset()
            grid, vector = obs

            # Heuristique ou RL selon si on a un checkpoint
            heuristic_mode = not os.path.exists(best_path) and not os.path.exists(checkpoint_path)

            if heuristic_mode:
                actions = env.get_heuristic_sequence()
                for action in actions:
                    obs, mask, reward, done, info = env.step(action)
                    if done:
                        break
            else:
                from clashai.combat.agent import MAX_STEPS_PER_EPISODE
                for step in range(MAX_STEPS_PER_EPISODE):
                    action, _, _ = agent.select_action(grid, vector, mask)
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break

            if self.verbose:
                stars = info.get('stars', '?')
                pct = info.get('percentage', '?')
                print(f"\n   📊 Résultat GdC : {stars}⭐ {pct}%")

            env.close()

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Erreur attaque V3 : {e}")
                import traceback
                traceback.print_exc()

    def run(self, monitor_interval=30):
        """
        Boucle principale : surveille le chat et exécute les commandes.
        
        Args:
            monitor_interval: secondes entre chaque check du chat
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print("  ClashAI GdC Orchestrator")
            print(f"  Bot: @{self._chat_monitor.bot_name}")
            print(f"  Interval: {monitor_interval}s")
            print(f"{'='*50}\n")

        self._chat_monitor.monitor_loop(
            classify_screen_fn=self._navigator._classify_screen,
            models=self.models,
            callback=self.handle_command,
            interval=monitor_interval,
        )


# =============================================================================
#                            MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClashAI GdC Navigator")
    parser.add_argument('--attack', type=int,
                        help="Attaquer la cible n°X en GdC")
    parser.add_argument('--navigate', type=int,
                        help="Naviguer vers la cible sans attaquer")
    parser.add_argument('--monitor', action='store_true',
                        help="Lancer le monitoring du chat")
    parser.add_argument('--bot-name', type=str, default='mini_pekka')
    parser.add_argument('--interval', type=int, default=30)

    args = parser.parse_args()

    # Charger les modèles
    current_dir = os.path.dirname(os.path.abspath(__file__))
        
    from clashai.navigation import game_loop
    models = game_loop.load_models()

    if args.attack:
        nav = GdCNavigator(models)
        success = nav.attack_target(args.attack)
        if success:
            print("✅ Phase d'attaque atteinte — l'agent V3 peut prendre le relais")
        else:
            print("❌ Navigation échouée")

    elif args.navigate:
        nav = GdCNavigator(models)
        if nav.navigate_to_war_map():
            success = nav.select_target(args.navigate)
            if success:
                print(f"✅ Cible #{args.navigate} sélectionnée")
            else:
                print(f"❌ Cible #{args.navigate} non trouvée")

    elif args.monitor:
        orchestrator = GdCOrchestrator(models, bot_name=args.bot_name)
        orchestrator.run(monitor_interval=args.interval)

    else:
        print("Usage :")
        print("  --attack 3      Attaquer la cible #3 en GdC")
        print("  --navigate 5    Naviguer vers la cible #5 (sans attaquer)")
        print("  --monitor       Surveiller le chat et exécuter les commandes")
        print("  --bot-name X    Nom du bot (défaut: mini_pekka)")
        print("  --interval N    Intervalle de monitoring en secondes")