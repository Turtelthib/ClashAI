# scripts/rl/environment_v3.py
# Environnement V3 pour ClashAI — IA réactive mid-combat.
#
# Changement majeur vs V2 :
#   L'épisode a maintenant 2 phases :
#
#   Phase DEPLOY (identique V2) :
#     L'agent pose ses troupes une par une, attend, lance des sorts.
#     Se termine quand l'agent fait "done".
#
#   Phase COMBAT (nouveau) :
#     L'agent continue à prendre des décisions PENDANT le combat :
#     - Activer les capacités des héros (rage du roi, cloak reine, etc.)
#     - Lancer les sorts restants (avec screenshot mid-combat → ciblage)
#     - Observer (wait_combat = attendre 2-3s et re-screenshotter)
#     Se termine quand le combat est fini (écran résultats) ou max steps.
#
# Usage :
#   env = ClashEnvV3(models)
#   obs, mask = env.reset()
#   while True:
#       action = agent.select_action(obs, mask)
#       obs, mask, reward, done, info = env.step(action)
#       if done: break

import time
import random

import numpy as np
import cv2

# Imports du projet

from clashai.combat.state_encoder import encode_state, find_best_attack_side
from clashai.perception.deploy_zone import (get_full_perimeter_positions)
from clashai.perception.reward_reader import read_attack_results
from clashai.combat.spell_caster import SpellCaster
from clashai.perception.troop_counter import read_troop_counts
from clashai.combat.combat_observer import CombatObserver, COMBAT_FEATURES_SIZE
from clashai.combat.hero_ability import HeroAbilityManager, HERO_NAMES

from clashai.combat.agent import (
    TROOP_TYPES, NUM_TROOP_TYPES, NUM_POSITIONS,
    TOTAL_ACTIONS, ACTION_WAIT_SHORT, ACTION_WAIT_LONG, ACTION_DONE,
    ACTION_WAIT_COMBAT, MAX_STEPS_PER_EPISODE, MAX_COMBAT_STEPS,
    GRID_CHANNELS, GRID_SIZE, VILLAGE_FEATURES, VECTOR_SIZE,
    decode_action, compute_action_mask, get_initial_troop_counts,
    get_troop_counts_from_finder, TROOP_NAME_TO_IDX,
)

# Import optionnel du zoom
try:
    from clashai.navigation.zoom_control import zoom_out as _zoom_out_fn
    ZOOM_AVAILABLE = True
except (ImportError, OSError):
    ZOOM_AVAILABLE = False


# =============================================================================
#                         CONFIGURATION
# =============================================================================

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Temps d'attente
WAIT_DECORATIONS = 3.0
WAIT_BATTLE_MAX = 195.0
WAIT_BATTLE_CHECK = 5.0
WAIT_RESULT_SCREEN = 5.0
WAIT_NAVIGATION = 1.5
WAIT_MATCHMAKING = 4.0

# Délais entre actions
DELAY_DEPLOY = 0.05
DELAY_SWITCH_TROOP = 0.15
DELAY_WAIT_SHORT = 0.5
DELAY_WAIT_LONG = 2.0
DELAY_WAIT_COMBAT = 2.5      # Observation pendant le combat
DELAY_ABILITY = 0.3           # Après activation d'une ability

MAX_NAV_RETRIES = 20

# Reward
REWARD_STAR_BONUS = 100
REWARD_ZERO_STAR_PENALTY = -50
REWARD_THREE_STAR_BONUS = 50
REWARD_FIRST_STAR_BONUS = 50

# Retraite intelligente
#
# Seule méthode fiable : seuil bas.
#   Si barres vertes <= GREEN_DEAD_THRESHOLD pendant N checks → troupes mortes.
#   Les barres vertes qui passent orange (troupes blessées) ne comptent PAS
#   comme mortes — une troupe blessée se bat encore.
#
# IMPORTANT : la "méthode plateau stable" a été retirée car elle causait
# des faux positifs catastrophiques. En combat normal, les barres vertes
# diminuent naturellement (vert → orange quand blessé), ce que le plateau
# interprétait à tort comme "troupes mortes". Résultat : surrender à 88% 2★.
#
GREEN_DEAD_THRESHOLD = 2           # Strictement en dessous = mort
NO_TROOPS_CHECKS_THRESHOLD = 3    # Checks consécutifs sous le seuil
NO_TROOPS_MIN_WAIT = 5.0          # Attente min après surrender (secondes)
# Quand on détecte 0 troupes pendant _wait_for_battle_end, le jeu va finir
# tout seul dans quelques secondes — pas besoin d'attendre 30s.

# Reward shaping V3
REWARD_ABILITY_TIMING_GOOD = 3.0    # Ability utilisée au bon moment
REWARD_ABILITY_TIMING_BAD = -2.0    # Ability trop tôt / trop tard
REWARD_FREEZE_ON_INFERNO = 5.0      # Gel ciblé sur tour d'enfer


# =============================================================================
#                    ENVIRONNEMENT V3
# =============================================================================

class ClashEnvV3:
    """
    Environnement bi-phase pour Clash of Clans.

    Phase 1 (DEPLOY) : L'agent pose ses troupes.
    Phase 2 (COMBAT) : L'agent réagit au combat en cours.
    """

    def __init__(self, models, verbose=True):
        self.models = models
        self.verbose = verbose

        # Imports game_loop
        from clashai.navigation import game_loop as gl
        self._classify_screen = gl.classify_screen
        self._analyze_village = gl.analyze_village
        self._adb_screenshot = gl.adb_screenshot
        self._adb_tap = gl.adb_tap
        self._buttons = gl.BUTTONS

        # Charger les positions UI calibrées (une seule fois)
        try:
            from clashai.navigation.calibrate_ui import get_position
            self._ui = {
                'chat_open': get_position('chat_open'),
                'chat_close': get_position('chat_close_tap'),
                'close_profil': get_position('close_profil'),
                'close_menu': get_position('close_menu'),
                'close_popup': get_position('close_popup'),
                'gdc_return': get_position('gdc_return_home'),
                'ff_button': get_position('ff_button'),
                'confirm_ff': get_position('confirm_ff'),
            }
        except ImportError:
            self._ui = {
                'chat_open': (47, 400),
                'chat_close': (1400, 400),
                'close_profil': (1270, 90),
                'close_menu': (1340, 95),
                'close_popup': (1300, 100),
                'gdc_return': (80, 780),
                'ff_button': (1850, 550),
                'confirm_ff': (700, 550),
            }

        # État de l'épisode
        self._grid = None
        self._features = None
        self._buildings = None
        self._remaining_troops = None
        self._deploy_map = None
        self._step_count = 0
        self._deploy_positions = None
        self._spell_positions = None
        self._village_center = None
        self._last_troop_name = None
        self._episode_count = 0

        # Phase tracking (NOUVEAU V3)
        self._phase = 'deploy'           # 'deploy' ou 'combat'
        self._combat_step_count = 0      # Steps dans la phase combat
        self._combat_features = None     # Dernières features combat observées

        # Retraite intelligente
        self._no_troops_count = 0        # Checks consécutifs avec 0 troupes vivantes

        # Modules V3
        self._troop_detector = self._try_load_troop_detector()
        self._combat_observer = CombatObserver(verbose=self.verbose,
                                                troop_detector=self._troop_detector)
        self._hero_manager = HeroAbilityManager(verbose=self.verbose)
        self._spell_caster = SpellCaster(verbose=self.verbose)

        # TroopFinder (template matching barre de troupes)
        from clashai.perception.troop_finder import TroopFinder
        self._troop_finder = TroopFinder()

        # Reward shaping
        self._shaping_history = []
        self._tanks_deployed = 0
        self._troops_deployed = 0
        self._spells_deployed = 0
        self._last_deploy_pos = None
        self._step_rewards = []

        if self.verbose and type(self).__name__ == 'ClashEnvV3':
            print("\n🎮 ClashEnv V3 initialisé")
            print(f"   Actions     : {TOTAL_ACTIONS} "
                  f"(280 deploy + 3 ctrl + 5 abilities + 1 wait_combat)")
            print(f"   Vector      : {VECTOR_SIZE} dims")
            print("   Phases      : deploy → combat")
            print(f"   Max steps   : {MAX_STEPS_PER_EPISODE} "
                  f"(dont {MAX_COMBAT_STEPS} combat)")

    # -----------------------------------------------------------------
    #                 COMPORTEMENT HUMAIN
    # -----------------------------------------------------------------

    @staticmethod
    def _try_load_troop_detector():
        """Tente de charger le TroopDetector YOLO. Retourne None si indisponible."""
        try:
            from clashai.perception.troop_detector import TroopDetector, YOLO_TROOPS_PATH
            import os
            if os.path.exists(YOLO_TROOPS_PATH):
                detector = TroopDetector(verbose=True)
                print("   ✅ TroopDetector YOLO chargé (mode V4)")
                return detector
            else:
                print(f"   ⚠️  YOLO troupes introuvable ({YOLO_TROOPS_PATH}), fallback barres de vie")
        except ImportError:
            print("   ⚠️  TroopDetector non disponible, fallback barres de vie")
        return None

    def _human_idle(self):
        """Simule un comportement humain entre épisodes."""
        wait_time = random.uniform(15, 60)
        if self.verbose:
            print(f"   😴 Pause humaine ({wait_time:.0f}s)...")

        elapsed = 0
        while elapsed < wait_time:
            action = random.choices(
                ['wait', 'zoom_in', 'zoom_out', 'small_scroll'],
                weights=[0.5, 0.15, 0.15, 0.2], k=1
            )[0]

            if action in ('zoom_in', 'zoom_out'):
                try:
                    from clashai.navigation.zoom_control import zoom_in, zoom_out
                    fn = zoom_in if action == 'zoom_in' else zoom_out
                    fn(scrolls=random.randint(2, 5))
                except ImportError:
                    pass
                pause = random.uniform(1.5, 4.0)
            elif action == 'small_scroll':
                x1 = random.randint(400, 1500)
                y1 = random.randint(200, 600)
                dx, dy = random.randint(-150, 150), random.randint(-100, 100)
                try:
                    import subprocess
                    subprocess.run(
                        ["adb", "shell",
                         f"input swipe {x1} {y1} {x1+dx} {y1+dy} "
                         f"{random.randint(200, 500)}"],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass
                pause = random.uniform(2.0, 5.0)
            else:
                pause = random.uniform(2.0, 6.0)

            time.sleep(pause)
            elapsed += pause

    # -----------------------------------------------------------------
    #                    OBSERVATION
    # -----------------------------------------------------------------

    def _get_obs(self):
        """Construit l'observation complète (grid, vector)."""
        step_norm = np.array(
            [self._step_count / MAX_STEPS_PER_EPISODE],
            dtype=np.float32
        )
        phase_indicator = np.array(
            [1.0 if self._phase == 'combat' else 0.0],
            dtype=np.float32
        )

        # Combat features (0 pendant deploy, mis à jour pendant combat)
        combat_feats = (self._combat_features
                        if self._combat_features is not None
                        else np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32))

        # Hero ability status
        hero_status = self._hero_manager.get_status_vector()

        vector = np.concatenate([
            self._features,                    # (8,)
            self._remaining_troops / 10.0,     # (14,) normalisé
            self._deploy_map,                  # (20,)
            step_norm,                         # (1,)
            combat_feats,                      # (15,)  NOUVEAU V3
            hero_status,                       # (5,)   NOUVEAU V3
            phase_indicator,                   # (1,)   NOUVEAU V3
        ])

        return self._grid, vector

    def _get_mask(self):
        """Construit le masque d'actions selon la phase."""
        hero_mask = self._hero_manager.get_ability_mask()
        return compute_action_mask(
            self._remaining_troops,
            phase=self._phase,
            hero_ability_mask=hero_mask
        )

    # -----------------------------------------------------------------
    #                    NAVIGATION ADB
    # -----------------------------------------------------------------

    def _get_screen_state(self):
        img_pil = self._adb_screenshot()
        if img_pil is None:
            return None, 0.0, None
        state, confidence = self._classify_screen(img_pil, self.models)
        return state, confidence, img_pil

    def _navigate_to(self, target_state, timeout_retries=MAX_NAV_RETRIES):
        last_state = None
        stuck_count = 0
        MAX_STUCK = 4

        for attempt in range(timeout_retries):
            state, confidence, img_pil = self._get_screen_state()
            if state is None:
                time.sleep(1)
                continue

            if self.verbose and attempt % 3 == 0:
                print(f"   📍 État: {state} ({confidence:.0%}) "
                      f"[cible: {target_state}]")

            if state == last_state:
                stuck_count += 1
            else:
                stuck_count = 0
            last_state = state

            if stuck_count >= MAX_STUCK:
                if self.verbose:
                    print(f"   🔄 Bloqué sur '{state}' → récupération")
                self._recovery_sequence()
                stuck_count = 0
                continue

            if confidence < 0.55 and state != target_state:
                time.sleep(1.0)
                continue

            if state == target_state:
                return True, img_pil

            # Navigation selon l'état
            if state == 'village_home':
                self._adb_tap(*self._buttons['attaquer'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'recherche_adversaire':
                self._adb_tap(*self._buttons['trouver_partie'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'prep_attaque':
                self._adb_tap(*self._buttons['lancer_attaque'])
                time.sleep(WAIT_MATCHMAKING)
            elif state == 'resultats_attaque':
                self._return_to_village()
            elif state == 'chargement':
                time.sleep(2)
            elif state in ('gdc_ally', 'gdc_enemy', 'gdc_ended'):
                self._adb_tap(*self._ui['gdc_return'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'profil':
                self._adb_tap(*self._ui['close_profil'])
                time.sleep(0.5)
                self._adb_tap(1800, 500)
                time.sleep(0.5)
                self._adb_tap(30, 500)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'popup_offre':
                self._adb_tap(*self._ui['close_popup'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chat_clan':
                self._adb_tap(*self._ui['chat_close'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'menu_boutique':
                self._adb_tap(*self._ui['close_menu'])
                time.sleep(WAIT_NAVIGATION)
            else:
                time.sleep(WAIT_NAVIGATION)

        return False, None

    def _recovery_sequence(self):
        if self.verbose:
            print("   🚑 Séquence de récupération...")
        for x, y, wait in [
            (30, 540, 0.5), (1340, 95, 0.5), (1800, 500, 0.5),
            (30, 540, 0.5), (1270, 90, 0.5), (30, 540, 1.0),
        ]:
            self._adb_tap(x, y)
            time.sleep(wait)
            state, _, _ = self._get_screen_state()
            if state in ('village_home', 'phase_attaque', 'prep_attaque'):
                return

    def _zoom_out(self):
        if ZOOM_AVAILABLE:
            try:
                _zoom_out_fn(scrolls=15)
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Dézoom échoué : {e}")

    def _find_green_button(self, img_pil):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        bottom_half = img_cv[h // 2:, :]
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 100, 120), (85, 255, 255))
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best_area, best_cx, best_cy = 0, None, None
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area > 5000 and bw > 150 and bw / max(bh, 1) > 1.5:
                if area > best_area:
                    best_area = area
                    best_cx = int(centroids[i][0])
                    best_cy = int(centroids[i][1]) + h // 2
        return (best_cx, best_cy) if best_cx else None

    def _return_to_village(self, max_retries=10):
        last_state = None
        stuck_count = 0
        for attempt in range(max_retries):
            state, conf, img_pil = self._get_screen_state()
            if state == last_state:
                stuck_count += 1
            else:
                stuck_count = 0
            last_state = state
            if stuck_count >= 3:
                self._recovery_sequence()
                stuck_count = 0
                continue
            if state == 'village_home':
                if self.verbose:
                    print("   ✅ Retour au village confirmé")
                return True
            elif state in ('resultats_attaque', None) and img_pil is not None:
                btn_pos = self._find_green_button(img_pil)
                if btn_pos:
                    self._adb_tap(btn_pos[0], btn_pos[1])
                else:
                    for btn_y in [800, 760, 840, 720]:
                        self._adb_tap(960, btn_y)
                        time.sleep(0.3)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chargement':
                time.sleep(2)
            elif state in ('gdc_ally', 'gdc_enemy', 'gdc_ended'):
                self._adb_tap(*self._ui['gdc_return'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'profil':
                self._adb_tap(*self._ui['close_profil'])
                time.sleep(0.5)
                self._adb_tap(1800, 500)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'chat_clan':
                self._adb_tap(*self._ui['chat_close'])
                time.sleep(0.5)
                self._adb_tap(30, 540)
                time.sleep(WAIT_NAVIGATION)
            elif state == 'menu_boutique':
                self._adb_tap(*self._ui['close_menu'])
                time.sleep(WAIT_NAVIGATION)
            elif state == 'popup_offre':
                self._adb_tap(*self._ui['close_popup'])
                time.sleep(WAIT_NAVIGATION)
            else:
                self._adb_tap(30, 540) 
                time.sleep(WAIT_NAVIGATION)
        return False

    # -----------------------------------------------------------------
    #                    RETRAITE (FF)
    # -----------------------------------------------------------------

    def _surrender(self):
        """
        Abandonne le combat en appuyant sur le drapeau blanc puis confirme.
        
        Appelé quand la retraite intelligente détecte 0 troupes vivantes.
        Flow : drapeau blanc → attente popup → confirmer → écran résultats.
        
        Returns:
            success: bool (True si on a pu appuyer sur les boutons)
        """
        if self.verbose:
            print("   🏳️ Abandon du combat...")

        # 1. Appuyer sur le drapeau blanc (FF button)
        self._adb_tap(*self._ui['ff_button'])
        time.sleep(1.0)  # Attendre la popup de confirmation

        # 2. Appuyer sur le bouton de confirmation
        self._adb_tap(*self._ui['confirm_ff'])
        time.sleep(0.5)

        if self.verbose:
            print("   🏳️ Retraite confirmée, attente écran résultats...")

        return True

    # -----------------------------------------------------------------
    #                    RESCAN BARRE DE TROUPES
    # -----------------------------------------------------------------

    # Fréquence de rescan automatique (tous les N steps en phase deploy)
    RESCAN_EVERY_N_STEPS = 8

    def _rescan_troop_bar(self):
        """
        Rescanne la barre de troupes avec un screenshot frais.
        
        Utilise la SATURATION des icônes pour déterminer si une troupe
        est encore disponible (colorée) ou épuisée (grisée).
        """
        img = self._adb_screenshot()
        if img is None:
            return

        self._troop_finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Mettre à jour les compteurs via saturation
        ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}

        for troop_name_raw, (tx, ty, conf) in self._troop_finder.positions.items():
            troop_name = ALIAS_MAP.get(troop_name_raw, troop_name_raw)
            if troop_name not in TROOP_NAME_TO_IDX:
                continue

            idx = TROOP_NAME_TO_IDX[troop_name]
            is_active = self._is_slot_active(img_cv, tx, ty)

            if not is_active:
                # Grisé = x0 → forcer le compteur à 0
                self._remaining_troops[idx] = 0
            elif self._remaining_troops[idx] <= 0:
                # Coloré mais compteur à 0 → désynchronisé → corriger
                self._remaining_troops[idx] = 1.0

        # Troupes disparues du TroopFinder → entièrement déployées
        available_names = set()
        for name_raw in self._troop_finder.positions:
            available_names.add(ALIAS_MAP.get(name_raw, name_raw))
        for i, t in enumerate(TROOP_TYPES):
            if t['name'] not in available_names and self._remaining_troops[i] > 0:
                if t['role'] != 'spell':
                    self._remaining_troops[i] = 0

        # OCR des compteurs pour affiner les counts
        try:
            real_counts = read_troop_counts(img, self._troop_finder)
            for name, count in real_counts.items():
                real_name = ALIAS_MAP.get(name, name)
                if real_name in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[real_name]
                    self._remaining_troops[idx] = float(count)
        except Exception:
            pass

        if self.verbose:
            remaining = int(np.sum(self._remaining_troops))
            print(f"      🔄 Rescan barre : {remaining} troupes restantes")

    # -----------------------------------------------------------------
    #                    CLEANUP TROUPES RESTANTES
    # -----------------------------------------------------------------

    def _cleanup_remaining_troops(self):
        """
        Déploie toutes les troupes encore disponibles dans la barre.
        
        Détection par SATURATION : dans CoC, les icônes de troupes
        épuisées (x0) sont grisées (saturation ≈ 0-18), tandis que
        les troupes encore disponibles sont colorées (saturation > 40).
        
        Bien plus fiable que les compteurs internes.
        """
        if self.verbose:
            print("\n   🧹 Cleanup : rescan de la barre...")

        img = self._adb_screenshot()
        if img is None:
            return

        self._troop_finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}
        SPELL_NAMES = {t['name'] for t in TROOP_TYPES if t['role'] == 'spell'}

        # Rescan OCR compteurs
        try:
            real_counts = read_troop_counts(img, self._troop_finder)
            for name, count in real_counts.items():
                real_name = ALIAS_MAP.get(name, name)
                if real_name in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[real_name]
                    self._remaining_troops[idx] = float(count)
        except Exception:
            pass

        if self.verbose:
            remaining = int(np.sum([
                self._remaining_troops[i] for i, t in enumerate(TROOP_TYPES)
                if t['role'] not in ('spell',)
            ]))
            print(f"      🔄 Rescan barre : {remaining} troupes restantes")

        # Positions de deploy (spread autour du centre)
        center_idx = NUM_POSITIONS // 2
        spread_positions = []
        for offset in [-2, -1, 0, 1, 2]:
            p_idx = (center_idx + offset) % NUM_POSITIONS
            if self._deploy_positions and p_idx < len(self._deploy_positions):
                spread_positions.append(self._deploy_positions[p_idx])
        if not spread_positions:
            spread_positions = [self._village_center or (960, 500)]

        # Identifier les troupes COLORÉES (non grisées) dans la barre
        troops_to_deploy = []
        for troop_name_raw, (tx, ty, conf) in self._troop_finder.positions.items():
            troop_name = ALIAS_MAP.get(troop_name_raw, troop_name_raw)

            if troop_name in SPELL_NAMES:
                continue
            if troop_name not in TROOP_NAME_TO_IDX:
                continue

            # Test de saturation : coloré = encore disponible
            if self._is_slot_active(img_cv, tx, ty):
                idx = TROOP_NAME_TO_IDX[troop_name]
                role = TROOP_TYPES[idx]['role']
                count = max(int(self._remaining_troops[idx]), 1)
                troops_to_deploy.append((troop_name_raw, troop_name, role, count))
                if self.verbose:
                    print(f"      📦 {troop_name} ({role}) x{count}")

        if not troops_to_deploy:
            if self.verbose:
                print("   🧹 Cleanup : rien à déployer")
            return

        total = sum(c for _, _, _, c in troops_to_deploy)
        if self.verbose:
            print(f"      → {total} troupes à déployer")

        # Déployer dans l'ordre tactique : tank → ranged → melee → siege → hero
        role_order = {'tank': 0, 'ranged': 1, 'melee': 2, 'siege': 3, 'hero': 4}
        troops_to_deploy.sort(key=lambda t: role_order.get(t[2], 99))

        deployed_count = 0
        MAX_ROUNDS = 5  # Max de cycles check-tap par troupe

        for troop_name_raw, troop_name, role, count in troops_to_deploy:
            if not self._troop_finder.select(troop_name_raw):
                continue

            time.sleep(DELAY_SWITCH_TROOP)
            idx = TROOP_NAME_TO_IDX[troop_name]
            tx, ty, _ = self._troop_finder.positions[troop_name_raw]
            taps_done = 0

            # Boucle : taper par batch → check saturation → stop si gris
            for round_i in range(MAX_ROUNDS):
                # Taper 3 fois
                for tap_i in range(3):
                    pos = spread_positions[(taps_done + tap_i) % len(spread_positions)]
                    self._adb_tap(pos[0], pos[1])
                    time.sleep(DELAY_DEPLOY)
                taps_done += 3

                # Check saturation (screenshot léger, pas de TroopFinder.update)
                check_img = self._adb_screenshot()
                if check_img is None:
                    break
                check_cv = cv2.cvtColor(np.array(check_img), cv2.COLOR_RGB2BGR)

                if not self._is_slot_active(check_cv, tx, ty):
                    # Slot grisé → tout est déployé
                    break

            deployed_count += taps_done
            self._remaining_troops[idx] = 0
            self._last_troop_name = None

            if self.verbose:
                status = 'grisé ✅' if taps_done < MAX_ROUNDS * 3 else 'max atteint'
                print(f"      🧹 {troop_name} → {taps_done} taps ({status})")

        if self.verbose:
            remaining = int(np.sum([
                self._remaining_troops[i] for i, t in enumerate(TROOP_TYPES)
                if t['role'] not in ('spell',)
            ]))
            print(f"   🧹 Cleanup terminé : {deployed_count} actions"
                  f" ({remaining} encore en compteur)")

    # Seuil de saturation pour distinguer actif (coloré) vs épuisé (grisé)
    SLOT_SATURATION_THRESHOLD = 40

    def _is_slot_active(self, img_cv, x, y):
        """
        Vérifie si un slot de troupe est coloré (actif) ou grisé (x0).
        
        Dans CoC :
        - Troupes disponibles : icône colorée, saturation moyenne > 40
        - Troupes épuisées (x0) : icône grisée, saturation < 20
        
        Args:
            img_cv: image BGR complète
            x, y: position ADB du centre de l'icône
            
        Returns:
            True si la troupe est encore disponible
        """
        h, w = img_cv.shape[:2]
        ix = int(x * w / SCREEN_WIDTH)
        iy = int(y * h / SCREEN_HEIGHT)

        # Zone d'échantillonnage : centre de l'icône (évite le texte x0 en haut)
        sample_half_w = int(15 * w / SCREEN_WIDTH)
        sample_top = int(20 * h / SCREEN_HEIGHT)
        sample_bot = int(5 * h / SCREEN_HEIGHT)

        y1 = max(0, iy - sample_top)
        y2 = min(h, iy + sample_bot)
        x1 = max(0, ix - sample_half_w)
        x2 = min(w, ix + sample_half_w)

        region = img_cv[y1:y2, x1:x2]
        if region.size == 0:
            return False

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        avg_saturation = float(np.mean(hsv[:, :, 1]))

        return avg_saturation > self.SLOT_SATURATION_THRESHOLD

    # -----------------------------------------------------------------
    #                    EXÉCUTION D'ACTIONS
    # -----------------------------------------------------------------

    TROOP_ALIASES = {
        'lance_buche': ['lance_buche_vide'],
    }

    def _select_troop(self, troop_name):
        if troop_name == self._last_troop_name:
            return True
        if self._troop_finder.select(troop_name):
            self._last_troop_name = troop_name
            return True
        for alias in self.TROOP_ALIASES.get(troop_name, []):
            if self._troop_finder.select(alias):
                self._last_troop_name = troop_name
                return True
        if self.verbose:
            print(f"      ⚠️  {troop_name} non trouvé dans la barre")
        self._last_troop_name = None
        return False

    def _execute_action(self, action_idx):
        """
        Exécute une action dans le jeu via ADB.
        Gère les deux phases (deploy et combat).
        """
        action_type, troop_idx, pos_idx = decode_action(action_idx)

        if action_type == 'deploy':
            troop = TROOP_TYPES[troop_idx]
            troop_name = troop['name']
            is_spell = troop['role'] == 'spell'
            tap_pos = None  # Position réelle du tap (pour le log)
            deploy_success = False

            if self._select_troop(troop_name):
                time.sleep(DELAY_SWITCH_TROOP)
                deploy_success = True

                if is_spell:
                    # Sorts intelligents via SpellCaster
                    combat_img = self._adb_screenshot()
                    if combat_img is not None:
                        # V4: si YOLO dispo, observer + cibler via YOLO
                        if self._combat_observer.has_yolo:
                            _, raw = self._combat_observer.observe(
                                combat_img, self._village_center, phase='combat')
                            targets = self._spell_caster.analyze_from_yolo(
                                raw, self._village_center)
                        else:
                            targets = self._spell_caster.analyze_battlefield(
                                combat_img, self._village_center)
                        spell_target_map = {'soin': 'heal', 'rage': 'rage', 'gel': 'freeze'}
                        target_key = spell_target_map.get(troop_name, 'heal')
                        x, y = targets[target_key]
                    else:
                        positions = self._spell_positions
                        if positions and pos_idx < len(positions):
                            x, y = positions[pos_idx]
                        else:
                            x, y = self._village_center or (960, 500)

                    self._adb_tap(x, y)
                    tap_pos = (x, y)
                    time.sleep(0.3)
                    self._last_troop_name = None
                else:
                    # Troupes normales
                    positions = self._deploy_positions
                    if positions and pos_idx < len(positions):
                        x, y = positions[pos_idx]
                        self._adb_tap(x, y)
                        time.sleep(DELAY_DEPLOY)

                    # Tracker héros pour les abilities
                    if troop['role'] == 'hero':
                        self._hero_manager.mark_deployed(troop_name)

            # --- Compteur : décrémenter SEULEMENT si le deploy a réussi ---
            # Avant ce fix, le compteur descendait même quand _select_troop
            # retournait False → l'agent pensait avoir tout posé mais la
            # barre affichait encore des troupes. Le masque d'actions se
            # fermait alors qu'il restait des troupes à déployer.
            if deploy_success:
                self._remaining_troops[troop_idx] = max(
                    0, self._remaining_troops[troop_idx] - 1
                )
            else:
                # La troupe n'a pas pu être sélectionnée. On rescanne la
                # barre pour mettre à jour les positions du TroopFinder.
                self._deploy_failed_count = getattr(
                    self, '_deploy_failed_count', 0) + 1
                if self._deploy_failed_count >= 3:
                    self._rescan_troop_bar()
                    self._deploy_failed_count = 0

            if pos_idx is not None:
                self._deploy_map[pos_idx] += 0.2

            if is_spell and tap_pos:
                return f"🧪{troop_name} → ({tap_pos[0]}, {tap_pos[1]})"
            elif is_spell:
                return f"🧪{troop_name} → (sélection échouée)"
            else:
                return f"{troop_name} → pos {pos_idx}"

        elif action_type == 'wait_short':
            time.sleep(DELAY_WAIT_SHORT)
            self._last_troop_name = None
            return "attendre 0.5s"

        elif action_type == 'wait_long':
            time.sleep(DELAY_WAIT_LONG)
            self._last_troop_name = None
            return "attendre 2.0s"

        elif action_type == 'done':
            if self._phase == 'deploy':
                # Transition deploy → combat
                return "DONE (deploy → combat)"
            else:
                # Fin de la phase combat
                return "DONE (fin combat actif)"

        elif action_type == 'ability':
            # NOUVEAU V3 : Activation capacité héros
            hero_name = HERO_NAMES[troop_idx]

            # Si l'icône n'est pas encore détectée, tenter un scan frais
            if hero_name not in self._hero_manager._icon_positions:
                screenshot = self._adb_screenshot()
                if screenshot is not None and self._hero_manager.has_templates():
                    self._hero_manager.scan(screenshot)

            success = self._hero_manager.activate(hero_name, self._adb_tap)
            time.sleep(DELAY_ABILITY)
            if success:
                return f"⚡ {hero_name} ability activée"
            else:
                return f"⚠️ {hero_name} ability échouée (icône non trouvée)"

        elif action_type == 'wait_combat':
            # NOUVEAU V3 : Observer le combat
            time.sleep(DELAY_WAIT_COMBAT)
            # Prendre un screenshot frais et mettre à jour les combat features
            self._update_combat_observation()
            return f"👁️ observe ({DELAY_WAIT_COMBAT}s)"

        return "???"

    def _update_combat_observation(self):
        """
        Prend un screenshot mid-combat et met à jour :
        - Les combat features (CombatObserver)
        - Les positions des icônes d'ability (HeroAbilityManager.scan)
        - Le compteur de retraite intelligente (0 troupes consécutifs)
        """
        screenshot = self._adb_screenshot()
        if screenshot is None:
            return

        # Construire le dict des sorts restants
        spells_remaining = {}
        for i, t in enumerate(TROOP_TYPES):
            if t['role'] == 'spell':
                spells_remaining[t['name']] = int(self._remaining_troops[i])

        features, raw_data = self._combat_observer.observe(
            screenshot,
            village_center_adb=self._village_center,
            spells_remaining=spells_remaining,
            phase=self._phase
        )

        self._combat_features = features

        # --- Retraite intelligente ---
        # V4 : si YOLO dispo, compter les troupes détectées directement
        # V3 fallback : barres VERTES uniquement (les rouges sont ambiguës)
        if 'yolo_detections' in raw_data:
            yolo_troops = [d for d in raw_data['yolo_detections'] if d.is_troop]
            yolo_heroes = [d for d in raw_data['yolo_detections'] if d.is_hero]
            troops_alive = len(yolo_troops) + len(yolo_heroes)
        else:
            num_green = len(raw_data.get('green_positions', []))
            num_heroes = raw_data.get('num_heroes', 0)
            troops_alive = num_green + num_heroes

        if troops_alive <= GREEN_DEAD_THRESHOLD and self._phase == 'combat':
            self._no_troops_count += 1
            if self.verbose:
                print(f"      💀 Troupes sous le seuil : {troops_alive} "
                      f"({self._no_troops_count}/{NO_TROOPS_CHECKS_THRESHOLD})")
        else:
            self._no_troops_count = 0

        # Scanner les icônes d'ability héros (template matching)
        if self._phase == 'combat' and self._hero_manager.has_templates():
            self._hero_manager.scan(screenshot)

        # V4 : mettre à jour les positions YOLO des héros
        if 'hero_positions_named' in raw_data:
            self._hero_manager.update_battlefield_positions(
                raw_data['hero_positions_named'])

    # -----------------------------------------------------------------
    #                       RESET
    # -----------------------------------------------------------------

    def reset(self):
        """Commence un nouvel épisode."""
        self._episode_count += 1
        self._step_count = 0
        self._remaining_troops = np.zeros(NUM_TROOP_TYPES, dtype=np.float32)
        self._deploy_map = np.zeros(NUM_POSITIONS, dtype=np.float32)
        self._last_troop_name = None
        self._phase = 'deploy'
        self._combat_step_count = 0
        self._combat_features = np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32)
        self._no_troops_count = 0

        # Reset modules V3
        self._hero_manager.reset()
        self._deploy_failed_count = 0

        # Reset reward shaping
        self._shaping_history = []
        self._tanks_deployed = 0
        self._troops_deployed = 0
        self._spells_deployed = 0
        self._last_deploy_pos = None
        self._step_rewards = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  ÉPISODE #{self._episode_count} — Reset V3")
            print(f"{'='*60}")

        # Comportement humain entre épisodes
        if self._episode_count > 1:
            self._human_idle()

        # 1. Naviguer vers le village ennemi
        success, img_pil = self._navigate_to('phase_attaque')
        if not success:
            if self.verbose:
                print("❌ Impossible d'atteindre un village ennemi")
            self._grid = np.zeros(
                (GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
            )
            self._features = np.zeros(VILLAGE_FEATURES, dtype=np.float32)
            self._deploy_positions = [(960, 500)] * NUM_POSITIONS
            self._spell_positions = self._generate_spell_positions((960, 500))
            self._village_center = (960, 500)
            return self._get_obs(), self._get_mask()

        # 2. Attendre + dézoomer
        if self.verbose:
            print(f"   ⏳ Attente décorations ({WAIT_DECORATIONS}s)...")
        time.sleep(WAIT_DECORATIONS)
        self._zoom_out()

        # 3. Screenshot frais
        img_pil = self._adb_screenshot()
        if img_pil is None:
            self._grid = np.zeros(
                (GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
            )
            self._features = np.zeros(VILLAGE_FEATURES, dtype=np.float32)
            self._deploy_positions = [(960, 500)] * NUM_POSITIONS
            self._spell_positions = self._generate_spell_positions((960, 500))
            self._village_center = (960, 500)
            return self._get_obs(), self._get_mask()

        # 4. Détecter les troupes dans la barre
        self._troop_finder.update(img_pil)
        self._remaining_troops = get_troop_counts_from_finder(self._troop_finder)

        # 4b. OCR compteurs
        try:
            real_counts = read_troop_counts(img_pil, self._troop_finder)
            ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}
            for name, count in real_counts.items():
                real_name = ALIAS_MAP.get(name, name)
                if real_name in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[real_name]
                    if self._remaining_troops[idx] > 0:
                        self._remaining_troops[idx] = float(count)
            if self.verbose and real_counts:
                print(f"   📖 OCR compteurs : {dict(real_counts)}")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  OCR compteurs échoué : {e}")

        if self.verbose:
            detected = [TROOP_TYPES[i]['name'] for i in range(NUM_TROOP_TYPES)
                        if self._remaining_troops[i] > 0]
            total = int(np.sum(self._remaining_troops))
            print(f"   🎯 {len(detected)} types ({total} unités)")

        # 5. Zone de déploiement
        positions_all, center_adb, zone_ok = get_full_perimeter_positions(
            img_pil, num_points=NUM_POSITIONS
        )

        # Si la zone n'est pas bien détectée → retry avec un dézoom supplémentaire
        if not zone_ok or not positions_all or len(positions_all) < NUM_POSITIONS // 2:
            if self.verbose:
                print("   🔄 Zone mal détectée, retry avec dézoom supplémentaire...")
            self._zoom_out()
            time.sleep(1.0)
            img_retry = self._adb_screenshot()
            if img_retry is not None:
                img_pil = img_retry
                positions_all, center_adb, zone_ok = get_full_perimeter_positions(
                    img_pil, num_points=NUM_POSITIONS
                )
                # Aussi rescanner les troupes sur le nouveau screenshot
                self._troop_finder.update(img_pil)

        if zone_ok and positions_all and len(positions_all) >= NUM_POSITIONS // 2:
            while len(positions_all) < NUM_POSITIONS:
                positions_all.append(positions_all[-1])
            self._deploy_positions = positions_all[:NUM_POSITIONS]
        else:
            if self.verbose:
                print("   ⚠️  Zone non détectée, positions fallback")
            self._deploy_positions = self._generate_fallback_positions()
            center_adb = (960, 500)

        self._village_center = center_adb
        self._spell_positions = self._generate_spell_positions(center_adb)

        # 6. YOLO+CNN
        if self.verbose:
            print("   🔍 Analyse YOLO+CNN du village...")
        buildings = self._analyze_village(img_pil, self.models)
        if self.verbose:
            print(f"   🏰 {len(buildings)} bâtiments détectés")

        self._buildings = buildings
        state = encode_state(buildings)
        self._grid = state['grid']
        self._features = state['features']

        # V2 SpellCaster : enregistrer les positions des défenses dangereuses
        self._spell_caster.set_defense_positions(buildings)

        return self._get_obs(), self._get_mask()

    def _generate_fallback_positions(self):
        positions = []
        cx, cy = 960, 450
        radius = 350
        for i in range(NUM_POSITIONS):
            angle = 2 * np.pi * i / NUM_POSITIONS
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle) * 0.6)
            positions.append((max(80, min(1840, x)), max(80, min(850, y))))
        return positions

    def _generate_spell_positions(self, center_adb):
        cx, cy = center_adb
        positions = []
        for row in range(5):
            for col in range(4):
                x = cx + int((col - 1.5) * 80)
                y = cy + int((row - 2.0) * 50)
                positions.append((max(100, min(1820, x)), max(100, min(850, y))))
        return positions[:NUM_POSITIONS]

    # -----------------------------------------------------------------
    #                   REWARD SHAPING
    # -----------------------------------------------------------------

    def _compute_shaping_reward(self, action_idx):
        """
        Reward shaping V3.
        
        Phase DEPLOY : identique V2 (6 règles tactiques).
        Phase COMBAT : nouvelles règles pour les abilities et sorts.
        """
        action_type, troop_idx, pos_idx = decode_action(action_idx)
        reward = 0.0

        if self._phase == 'deploy':
            # === Règles V2 identiques ===
            if action_type == 'deploy':
                troop = TROOP_TYPES[troop_idx]
                role = troop['role']

                # Règle 1 : Tanks d'abord
                if role == 'tank' and self._troops_deployed < 4:
                    reward += 5.0

                # Règle 2 : Sorts avant troupes
                if role == 'spell':
                    troops_left = sum(
                        self._remaining_troops[i]
                        for i, t in enumerate(TROOP_TYPES)
                        if t['role'] not in ('spell',)
                        and self._remaining_troops[i] > 0
                    )
                    if troops_left > 3:
                        reward -= 8.0
                    elif troops_left > 0:
                        reward -= 3.0

                # Règle 3 : Héros avant tanks
                if role == 'hero' and self._tanks_deployed == 0:
                    reward -= 3.0

                # Règle 4 : Concentration
                if role not in ('spell',) and pos_idx is not None:
                    if self._last_deploy_pos is not None:
                        dist = abs(pos_idx - self._last_deploy_pos)
                        dist = min(dist, NUM_POSITIONS - dist)
                        if dist <= 3:
                            reward += 1.0
                        elif dist >= 8:
                            reward -= 1.0

                # Compteurs
                if role == 'tank':
                    self._tanks_deployed += 1
                if role == 'spell':
                    self._spells_deployed += 1
                else:
                    self._troops_deployed += 1
                    self._last_deploy_pos = pos_idx

            elif action_type == 'wait_long':
                # Règle 5 : Attente stratégique
                if self._tanks_deployed > 0 and self._troops_deployed < 6:
                    reward += 3.0

            elif action_type == 'done':
                # Règle 6 : Troupes non déployées
                troops_remaining = sum(
                    int(self._remaining_troops[i])
                    for i, t in enumerate(TROOP_TYPES)
                    if t['role'] not in ('spell',)
                )
                if troops_remaining > 0:
                    reward -= 2.0 * troops_remaining

        elif self._phase == 'combat':
            # === Nouvelles règles V3 ===

            if action_type == 'ability':
                hero_name = HERO_NAMES[troop_idx]

                # Règle 7 : Timing des abilities
                # Le roi devrait utiliser sa rage quand il est bas en HP
                # (approximé par le temps dans le combat)
                combat_progress = 0.0
                if self._combat_features is not None:
                    combat_progress = self._combat_features[1]  # progress

                if hero_name == 'roi':
                    # Le roi utilise sa rage en milieu/fin de combat
                    if 0.3 <= combat_progress <= 0.8:
                        reward += REWARD_ABILITY_TIMING_GOOD
                    elif combat_progress < 0.1:
                        reward += REWARD_ABILITY_TIMING_BAD  # Trop tôt

                elif hero_name == 'reine':
                    # La reine utilise son cloak quand il reste des défenses
                    if 0.2 <= combat_progress <= 0.7:
                        reward += REWARD_ABILITY_TIMING_GOOD

                elif hero_name == 'grand_gardien':
                    # Le GG utilise son tome quand les troupes prennent des dégâts
                    if self._combat_features is not None:
                        hurt_ratio = self._combat_features[10]
                        if hurt_ratio > 0.3:  # Beaucoup de troupes blessées
                            reward += REWARD_ABILITY_TIMING_GOOD + 2.0
                        elif hurt_ratio < 0.1:
                            reward += REWARD_ABILITY_TIMING_BAD

                elif hero_name in ('championne', 'prince_gargouille'):
                    # Timing flexible, léger bonus si utilisé
                    if combat_progress > 0.2:
                        reward += 1.0

            elif action_type == 'deploy' and TROOP_TYPES[troop_idx]['role'] == 'spell':
                # Règle 8 : Sort bien ciblé pendant le combat → petit bonus
                reward += 1.0

            elif action_type == 'wait_combat':
                # Règle 9 : Observer est neutre (pas de pénalité)
                # Mais si on observe trop sans agir → léger malus
                if self._combat_step_count > 10:
                    reward -= 0.5

        self._shaping_history.append((action_type, troop_idx, pos_idx, reward))
        return reward

    # -----------------------------------------------------------------
    #                        STEP
    # -----------------------------------------------------------------

    def step(self, action_idx):
        """
        Exécute une action.
        Gère la transition deploy → combat automatiquement.
        """
        self._step_count += 1

        # Shaping reward
        shaping = self._compute_shaping_reward(action_idx)
        self._step_rewards.append(shaping)

        # Exécuter
        action_desc = self._execute_action(action_idx)

        if self.verbose:
            phase_tag = "🏗️" if self._phase == 'deploy' else "⚔️"
            shaping_str = f" ({shaping:+.0f})" if shaping != 0 else ""
            print(f"   {phase_tag} Step {self._step_count:2d}: "
                  f"{action_desc}{shaping_str}")

        action_type, _, _ = decode_action(action_idx)

        # --- Rescan périodique de la barre pendant le deploy ---
        # Les icônes bougent quand des troupes sont posées.
        # Sans rescan, le TroopFinder tape au mauvais endroit.
        if (self._phase == 'deploy'
                and self._step_count % self.RESCAN_EVERY_N_STEPS == 0
                and action_type != 'done'):
            self._rescan_troop_bar()

        # --- Transition deploy → combat ---
        if action_type == 'done' and self._phase == 'deploy':
            # Cleanup : déployer les troupes restantes avant le combat
            self._cleanup_remaining_troops()

            self._phase = 'combat'
            self._combat_step_count = 0
            self._combat_observer.start_combat()

            if self.verbose:
                print("\n   ⚔️ ═══ PHASE COMBAT ═══")
                print(f"   Héros déployés : {self._hero_manager.num_deployed()}")
                abilities = self._hero_manager.get_available_abilities()
                if abilities:
                    print(f"   Abilities dispo : {abilities}")

            # Première observation combat
            self._update_combat_observation()

            return self._get_obs(), self._get_mask(), shaping, False, {
                'step': self._step_count,
                'phase': 'combat'
            }

        # --- Pendant la phase combat ---
        if self._phase == 'combat':
            self._combat_step_count += 1

            # Vérifier si le combat est terminé
            is_battle_over = self._check_battle_end()

            # Conditions de fin de phase combat
            is_done = (
                is_battle_over
                or action_type == 'done'
                or self._combat_step_count >= MAX_COMBAT_STEPS
                or self._step_count >= MAX_STEPS_PER_EPISODE
            )

            if is_done:
                combat_reward, info = self._finish_episode()
                info['shaping_total'] = sum(self._step_rewards)
                info['combat_reward'] = combat_reward
                info['combat_steps'] = self._combat_step_count
                info['abilities_used'] = self._hero_manager.num_activated()
                return self._get_obs(), self._get_mask(), combat_reward, True, info
            else:
                return self._get_obs(), self._get_mask(), shaping, False, {
                    'step': self._step_count,
                    'combat_step': self._combat_step_count,
                    'phase': 'combat'
                }

        # --- Pendant la phase deploy ---
        is_done = False
        if self._step_count >= MAX_STEPS_PER_EPISODE:
            is_done = True
        # Ne PAS forcer la fin quand remaining=0 — l'agent/heuristique
        # doit atteindre ACTION_DONE pour passer en phase combat.
        # Sinon les sorts en fin de deploy tuent la phase combat.

        if is_done:
            # Pas de phase combat si on force la fin → attente passive
            combat_reward, info = self._finish_episode()
            info['shaping_total'] = sum(self._step_rewards)
            info['combat_reward'] = combat_reward
            return self._get_obs(), self._get_mask(), combat_reward, True, info

        return self._get_obs(), self._get_mask(), shaping, False, {
            'step': self._step_count
        }

    def _check_battle_end(self):
        """
        Vérifie si le combat est terminé.
        
        Deux conditions :
        1. L'écran montre les résultats (fin normale)
        2. Aucune troupe vivante détectée N fois de suite (retraite intelligente)
           → le combat va finir tout seul dans quelques secondes
        """
        # Condition 1 : écran résultats
        state, confidence, _ = self._get_screen_state()
        if state == 'resultats_attaque' and confidence > 0.6:
            return True
        
        # Condition 2 : retraite intelligente (0 troupes consécutifs)
        if self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD:
            if self.verbose:
                print(f"      🏳️ Retraite intelligente : "
                      f"0 troupes depuis {self._no_troops_count} checks")
            return True
        
        return False

    def get_step_rewards(self):
        return self._step_rewards

    # -----------------------------------------------------------------
    #                   FIN D'ÉPISODE
    # -----------------------------------------------------------------

    def _finish_episode(self):
        """Attend la fin du combat et calcule le reward."""
        if self.verbose:
            remaining = int(np.sum(self._remaining_troops))
            print(f"\n   🏁 Épisode terminé !"
                  f" ({self._step_count} steps,"
                  f" {self._combat_step_count} combat,"
                  f" {remaining} restantes,"
                  f" {self._hero_manager.num_activated()} abilities)")

        # Si on est en phase combat, le combat est peut-être déjà fini
        # Sinon, attendre passivement
        result_img = self._wait_for_battle_end()

        if result_img is not None:
            results = read_attack_results(result_img, debug=False)
            stars = results['stars']
            percentage = results['percentage']
            success = results['success']
        else:
            if self.verbose:
                print("   ⚠️  Impossible de lire les résultats")
            stars = 0
            percentage = 0
            success = False

        reward = self._compute_reward(stars, percentage)

        if self.verbose:
            retreat_str = " (🏳️ retraite)" if self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD else ""
            print(f"\n   📊 RÉSULTATS{retreat_str} :")
            print(f"      ⭐ Étoiles    : {stars}/3")
            print(f"      📊 Pourcentage : {percentage}%")
            print(f"      🏆 Reward      : {reward:.0f}")

        if self.verbose:
            print("   🏠 Retour au village...")
        self._return_to_village()

        info = {
            'stars': stars,
            'percentage': percentage,
            'reward': reward,
            'success': success,
            'steps': self._step_count,
            'deploy_steps': self._step_count - self._combat_step_count,
            'combat_steps': self._combat_step_count,
            'troops_remaining': int(np.sum(self._remaining_troops)),
            'abilities_used': self._hero_manager.num_activated(),
            'episode': self._episode_count,
            'early_retreat': self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD,
        }

        return reward, info

    def _wait_for_battle_end(self):
        """
        Attend la fin du combat.
        
        Détection accélérée par seuil bas :
          Si barres vertes <= 2 pendant 3 checks consécutifs → troupes mortes.
          → Surrender (drapeau blanc + confirmation) → résultats en ~5s.
        
        NOTE : les barres vertes diminuent naturellement quand les troupes
        prennent des dégâts (vert → orange). C'est pour ça qu'on utilise
        un seuil très bas (≤2) et non un ratio du pic — une troupe blessée
        (barre orange) est encore vivante et se bat.
        """
        if self.verbose:
            print("   ⏱️  Attente fin de combat...")

        # Si la retraite intelligente a été déclenchée pendant la phase combat,
        # on fait le surrender tout de suite
        surrendered = False
        if self._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD:
            self._surrender()
            surrendered = True
            min_wait = NO_TROOPS_MIN_WAIT
            if self.verbose:
                print(f"   🏳️ Retraite active → attente réduite ({min_wait:.0f}s min)")
        else:
            min_wait = 15.0 if self._phase == 'combat' else 30.0

        start_time = time.time()
        no_troops_consecutive = 0

        while time.time() - start_time < WAIT_BATTLE_MAX:
            elapsed = time.time() - start_time

            # 1. Vérifier l'écran
            state, confidence, img_pil = self._get_screen_state()

            if self.verbose and int(elapsed) % 10 == 0:
                print(f"      📍 {elapsed:.0f}s — écran: {state} ({confidence:.0%})")

            if state == 'resultats_attaque' and elapsed >= min_wait:
                if self.verbose:
                    print(f"   ✅ Combat terminé après {elapsed:.0f}s")
                time.sleep(WAIT_RESULT_SCREEN)
                final_img = self._adb_screenshot()
                return final_img if final_img else img_pil

            # 2. Scanner les barres VERTES uniquement
            #    (les oranges/rouges = troupes blessées OU bâtiments ennemis)
            if img_pil is not None and not surrendered:
                try:
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    from clashai.combat.combat_observer import detect_troop_bars, detect_hero_bars
                    green_pos, _ = detect_troop_bars(img_cv)
                    hero_pos = detect_hero_bars(img_cv)
                    green_count = len(green_pos) + len(hero_pos)

                    if self.verbose:
                        print(f"      🔍 Scan: {len(green_pos)} vertes, "
                              f"{len(hero_pos)} héros "
                              f"→ vivantes={green_count}")

                    if green_count <= GREEN_DEAD_THRESHOLD:
                        no_troops_consecutive += 1
                        if self.verbose:
                            print(f"      💀 Sous le seuil "
                                  f"({no_troops_consecutive}/{NO_TROOPS_CHECKS_THRESHOLD})")
                    else:
                        no_troops_consecutive = 0

                    if no_troops_consecutive >= NO_TROOPS_CHECKS_THRESHOLD:
                        if self.verbose:
                            print(f"   🏳️ Troupes mortes "
                                  f"(vert<={GREEN_DEAD_THRESHOLD} "
                                  f"x{NO_TROOPS_CHECKS_THRESHOLD})")
                        self._surrender()
                        surrendered = True
                        min_wait = NO_TROOPS_MIN_WAIT

                except Exception as e:
                    if self.verbose:
                        print(f"      ⚠️ Scan troupes échoué: {e}")

            time.sleep(WAIT_BATTLE_CHECK)

        state, _, img_pil = self._get_screen_state()
        if state == 'resultats_attaque':
            return img_pil
        return None

    def _compute_reward(self, stars, percentage):
        reward = (stars * REWARD_STAR_BONUS) + percentage
        if stars >= 1:
            reward += REWARD_FIRST_STAR_BONUS
        if stars == 0:
            reward += REWARD_ZERO_STAR_PENALTY
        if stars == 3 and percentage == 100:
            reward += REWARD_THREE_STAR_BONUS
        return float(reward)

    # -----------------------------------------------------------------
    #                  HEURISTIQUE (baseline)
    # -----------------------------------------------------------------

    def get_heuristic_sequence(self):
        """
        Séquence heuristique V3 — entièrement dynamique.
        
        S'adapte automatiquement aux troupes/sorts réellement disponibles.
        
        Phase DEPLOY : troupes seulement (tanks → funnel → ranged → melee → siège → héros)
        Phase COMBAT : sorts (avec screenshot frais à chaque lancer) + abilities héros
        
        Les sorts sont dans la phase combat car :
        - Le SpellCaster fait un screenshot à chaque sort → ciblage précis
        - On voit les troupes se battre → on sait où poser soin/rage/gel
        - Le gel peut cibler les infernos proches des troupes en temps réel
        """
        if self._buildings:
            best_dir = find_best_attack_side(
                self._buildings, verbose=self.verbose
            )
        else:
            best_dir = 0

        center_pos = int(best_dir / 8 * NUM_POSITIONS) % NUM_POSITIONS
        positions = [(center_pos + i - 2) % NUM_POSITIONS for i in range(5)]

        actions = []
        remaining = self._remaining_troops.copy()

        def add(name, pos):
            """Ajoute une action si la troupe est disponible."""
            if name not in TROOP_NAME_TO_IDX:
                return False
            idx = TROOP_NAME_TO_IDX[name]
            if remaining[idx] > 0:
                actions.append(idx * NUM_POSITIONS + pos)
                remaining[idx] -= 1
                return True
            return False

        def add_all(name, pos_list):
            """Déploie toutes les unités d'un type sur les positions données."""
            if name not in TROOP_NAME_TO_IDX:
                return 0
            idx = TROOP_NAME_TO_IDX[name]
            count = 0
            i = 0
            while remaining[idx] > 0:
                p = pos_list[i % len(pos_list)]
                actions.append(idx * NUM_POSITIONS + p)
                remaining[idx] -= 1
                count += 1
                i += 1
            return count

        def add_one_spell(spell_name, pos):
            """Lance UN sort s'il est disponible."""
            return add(spell_name, pos)

        # ============================================================
        # Inventaire dynamique
        # ============================================================
        tanks = [t['name'] for t in TROOP_TYPES 
                 if t['role'] == 'tank' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        ranged = [t['name'] for t in TROOP_TYPES 
                  if t['role'] == 'ranged' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        melee = [t['name'] for t in TROOP_TYPES 
                 if t['role'] == 'melee' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        heroes = [t['name'] for t in TROOP_TYPES 
                  if t['role'] == 'hero' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        sieges = [t['name'] for t in TROOP_TYPES 
                  if t['role'] == 'siege' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
        spells = {}
        for t in TROOP_TYPES:
            if t['role'] == 'spell':
                idx = TROOP_NAME_TO_IDX[t['name']]
                if remaining[idx] > 0:
                    spells[t['name']] = int(remaining[idx])

        total_spells = sum(spells.values())
        sp = 10  # Position action pour les sorts (le SpellCaster override les coords)

        if self.verbose:
            print(f"   📋 Inventaire : "
                  f"{len(tanks)} tanks, {len(ranged)} ranged, "
                  f"{len(melee)} melee, {len(heroes)} héros, "
                  f"{len(sieges)} sièges, {total_spells} sorts {spells}")

        # ============================================================
        #              PHASE DEPLOY : troupes seulement
        # ============================================================

        # 1. TANKS aux extrémités
        for tank_name in tanks:
            idx = TROOP_NAME_TO_IDX[tank_name]
            n = int(remaining[idx])
            if n >= 2:
                add(tank_name, positions[0])
                add(tank_name, positions[4])
                add_all(tank_name, [positions[2]])
            elif n == 1:
                add(tank_name, positions[2])

        actions.append(ACTION_WAIT_LONG)

        # 2. FUNNEL — ranged aux extrémités
        funnel_count = 0
        for r_name in ranged:
            idx = TROOP_NAME_TO_IDX[r_name]
            if remaining[idx] >= 2 and funnel_count < 2:
                add(r_name, positions[0])
                add(r_name, positions[4])
                funnel_count += 1

        actions.append(ACTION_WAIT_SHORT)

        # 3. RANGED en ligne
        for r_name in ranged:
            add_all(r_name, positions[1:4])

        actions.append(ACTION_WAIT_LONG)

        # 4. MELEE + SIÈGE au centre
        for m_name in melee:
            add_all(m_name, [positions[2], positions[1], positions[3]])
        for s_name in sieges:
            add_all(s_name, [positions[2]])

        # 5. HÉROS au centre
        for h_name in heroes:
            add(h_name, positions[2])

        # → DONE : transition vers le combat
        actions.append(ACTION_DONE)

        # ============================================================
        #      PHASE COMBAT : sorts + abilities (screenshot frais)
        #
        #  Chaque wait_combat = screenshot + SpellCaster recalcule
        #  les positions en temps réel. Les sorts sont ciblés sur
        #  ce que l'IA VOIT, pas sur des coordonnées statiques.
        #
        #  Séquence tactique :
        #    1. Observer (les troupes engagent)
        #    2. RAGE (boost DPS pendant l'engagement)
        #    3. Observer (voir les dégâts)
        #    4. GEL sur inferno/eagle (protéger les troupes)
        #    5. Abilities héros (GG en premier = invincibilité)
        #    6. SOIN (les troupes ont pris des dégâts)
        #    7. Observer + alterner RAGE/SOIN restants
        #    8. Abilities héros restantes
        # ============================================================
        from clashai.combat.agent import (ACTION_ABILITY_ROI, ACTION_ABILITY_REINE,
                              ACTION_ABILITY_GG, ACTION_ABILITY_CHAMP, ACTION_ABILITY_PG)

        ABILITY_ORDER = [
            ('grand_gardien', ACTION_ABILITY_GG),
            ('roi', ACTION_ABILITY_ROI),
            ('reine', ACTION_ABILITY_REINE),
            ('championne', ACTION_ABILITY_CHAMP),
            ('prince_gargouille', ACTION_ABILITY_PG),
        ]

        # Séparer les abilities en deux vagues
        wave1_abilities = []  # GG + Roi (défensif, à lancer tôt)
        wave2_abilities = []  # Reine + Champ + PG (offensif, après les sorts)
        for hero_name, ability_action in ABILITY_ORDER:
            if hero_name in heroes:
                if hero_name in ('grand_gardien', 'roi'):
                    wave1_abilities.append(ability_action)
                else:
                    wave2_abilities.append(ability_action)

        # Trier les sorts par priorité tactique
        # Gel = urgent (protéger des infernos), Rage = boost, Soin = sustain
        spell_queue = []
        # D'abord : 1 rage (boost initial)
        if spells.get('rage', 0) > 0:
            spell_queue.append('rage')
            spells['rage'] -= 1
        # Ensuite : tous les gels (stopper les infernos)
        while spells.get('gel', 0) > 0:
            spell_queue.append('gel')
            spells['gel'] -= 1
        # Puis alterner soin/rage
        while any(v > 0 for v in spells.values()):
            for spell_name in ['soin', 'rage']:
                if spells.get(spell_name, 0) > 0:
                    spell_queue.append(spell_name)
                    spells[spell_name] -= 1
                    break
            else:
                # Sorts restants (autres types)
                for spell_name in list(spells.keys()):
                    if spells[spell_name] > 0:
                        spell_queue.append(spell_name)
                        spells[spell_name] -= 1
                        break
                else:
                    break

        # --- Construire la séquence combat ---

        # 1. Observer (les troupes engagent les défenses)
        actions.append(ACTION_WAIT_COMBAT)

        # 2. Premier sort (rage boost) + abilities défensives
        spell_idx = 0
        if spell_idx < len(spell_queue):
            add_one_spell(spell_queue[spell_idx], sp)
            spell_idx += 1

        for ability in wave1_abilities:
            actions.append(ability)

        # 3. Observer les dégâts
        actions.append(ACTION_WAIT_COMBAT)

        # 4. Gel + soin (protéger et soigner)
        spells_this_round = 0
        while spell_idx < len(spell_queue) and spells_this_round < 2:
            add_one_spell(spell_queue[spell_idx], sp + spells_this_round)
            spell_idx += 1
            spells_this_round += 1

        # 5. Observer
        actions.append(ACTION_WAIT_COMBAT)

        # 6. Abilities offensives
        for ability in wave2_abilities:
            actions.append(ability)

        # 7. Sorts restants (avec observe entre chaque pour le ciblage)
        while spell_idx < len(spell_queue):
            actions.append(ACTION_WAIT_COMBAT)
            add_one_spell(spell_queue[spell_idx], sp + (spell_idx % 4))
            spell_idx += 1

        actions.append(ACTION_DONE)

        return actions

    def close(self):
        if self.verbose:
            print(f"\n🎮 Environnement V3 fermé "
                  f"après {self._episode_count} épisodes")


# =============================================================================
#                        TEST
# =============================================================================

if __name__ == "__main__":
    print("ClashEnv V3 — test dry-run\n")

    from clashai.combat.agent import decode_action

    print("1. Test nouvelles actions V3 :")
    test_actions = [
        0, 19, 279, 280, 281, 282,  # V2 actions
        283, 284, 285, 286, 287, 288 # V3 actions (5 abilities + wait_combat)
    ]
    for a in test_actions:
        t, ti, pi = decode_action(a)
        if t == 'deploy':
            name = TROOP_TYPES[ti]['name']
            print(f"   Action {a:3d} → {t} {name} pos {pi}")
        elif t == 'ability':
            hero = HERO_NAMES[ti]
            print(f"   Action {a:3d} → {t} {hero}")
        else:
            print(f"   Action {a:3d} → {t}")

    print("\n2. Test masking par phase :")
    troops = get_initial_troop_counts()

    mask_deploy = compute_action_mask(troops, phase='deploy')
    print(f"   Phase deploy : {int(mask_deploy.sum())} actions valides / {TOTAL_ACTIONS}")

    hero_mask = np.array([1, 1, 0, 0, 0], dtype=np.float32)
    mask_combat = compute_action_mask(troops, phase='combat', hero_ability_mask=hero_mask)
    print(f"   Phase combat : {int(mask_combat.sum())} actions valides / {TOTAL_ACTIONS}")
    print(f"     - Sorts : {int(mask_combat[:280].sum())}")
    print(f"     - Abilities : {int(mask_combat[283:288].sum())}")
    print(f"     - Wait combat : {int(mask_combat[288])}")
    print(f"     - Done : {int(mask_combat[282])}")

    print("\n✅ Test dry-run V3 terminé !")