# scripts/rl/brain.py
# ClashAI Brain — Le cerveau unique du joueur IA.
#
# Un seul programme, un seul joueur, un seul compte.
# Le Brain décide quoi faire à chaque instant, exactement comme un
# humain qui regarde son téléphone :
#
#   "Tiens, je suis au village... je vais checker le chat...
#    Oh, on me demande d'attaquer le #3 en GdC... ok j'y vais...
#    C'est fait, je reviens au village... pas de nouvelles commandes...
#    Bon, je lance une attaque multi pour farm..."
#
# Boucle principale :
#   1. Où je suis ? (CNN écran)
#   2. Y a-t-il une commande urgente ? (chat clan)
#   3. Sinon, qu'est-ce que je fais ? (farm, gdc, attente)
#   4. Exécuter l'action
#   5. Retour au village → recommencer
#
# Usage :
#   python -m clashai.brain
#   python -m clashai.brain --mode farm        (attaques multi seulement)
#   python -m clashai.brain --mode gdc         (GdC seulement, attend les commandes)
#   python -m clashai.brain --mode auto        (tout : farm + GdC + chat)
#   python -m clashai.brain --episodes 50      (farm N attaques puis stop)

import os
import time
import random
import argparse
from datetime import datetime

# Setup paths
from clashai.paths import RL_WEIGHTS_DIR
# (paths centralisés via clashai.paths)


# =============================================================================
#                         CONFIGURATION
# =============================================================================

# Intervalles (secondes)
CHAT_CHECK_INTERVAL = 45       # Vérifier le chat toutes les 45s
IDLE_BETWEEN_ATTACKS = 20      # Pause min entre deux attaques farm
IDLE_BETWEEN_ATTACKS_MAX = 60  # Pause max (aléatoire pour être humain)

# Priorités (plus haut = plus urgent)
PRIORITY_GDC_COMMAND = 100     # Commande GdC du chat → top priorité
PRIORITY_FARM_ATTACK = 10      # Attaque farm → priorité normale
PRIORITY_IDLE = 0              # Rien à faire → attente

# Mode farm : nombre d'attaques avant de checker le chat
ATTACKS_BEFORE_CHAT_CHECK = 2

# Nom du bot
DEFAULT_BOT_NAME = 'mini_pekka'


# =============================================================================
#                         BRAIN
# =============================================================================

class ClashBrain:
    """
    Le cerveau unique de ClashAI.
    
    Gère tout le compte comme un vrai joueur :
    - Attaques farm (multi, pour les ressources)
    - Commandes GdC (depuis le chat du clan)
    - Comportement humain (pauses aléatoires, zooms)
    - Navigation robuste (sait toujours revenir au village)
    
    Futur : gestion du village (améliorations, donations, récolte)
    """

    def __init__(self, mode='auto', bot_name=DEFAULT_BOT_NAME, verbose=True):
        self.mode = mode
        self.bot_name = bot_name
        self.verbose = verbose
        self._running = False

        # Stats
        self._attacks_done = 0
        self._gdc_attacks_done = 0
        self._total_stars = 0
        self._total_destruction = 0
        self._start_time = None

        # Modules (chargés au démarrage)
        self._models = None
        self._env = None
        self._agent = None
        self._chat_monitor = None
        self._gdc_navigator = None
        self._cc_manager = None

        # File d'attente des tâches
        self._task_queue = []

        # Compteurs pour le cycle
        self._attacks_since_chat_check = 0
        self._last_chat_check = 0

    def start(self, max_episodes=None):
        """
        Démarre le Brain. C'est la seule méthode à appeler.
        
        Args:
            max_episodes: nombre max d'attaques farm (None = infini)
        """
        self._running = True
        self._start_time = time.time()

        print(f"\n{'='*60}")
        print("  🧠 ClashAI Brain — Démarrage")
        print(f"  Mode       : {self.mode}")
        print(f"  Bot name   : @{self.bot_name}")
        if max_episodes:
            print(f"  Max attacks: {max_episodes}")
        print(f"  Heure      : {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        # 1. Charger tous les modules
        self._load_modules()

        # 2. Boucle principale
        try:
            self._main_loop(max_episodes)
        except KeyboardInterrupt:
            print("\n\n⛔ Arrêt demandé (Ctrl+C)")
        finally:
            self._shutdown()

    def _load_modules(self):
        """Charge les modèles et initialise tous les modules."""
        print("📦 Chargement des modules...")

        # Modèles de perception
        from clashai.navigation import game_loop
        self._models = game_loop.load_models()

        # Agent V4
        from clashai.combat.agent_v4 import PPOAgentV4
        self._agent = PPOAgentV4()

        # Charger le meilleur checkpoint si disponible
        weights_dir = RL_WEIGHTS_DIR
        best_path = os.path.join(weights_dir, 'agent_v4_best.pth')
        checkpoint_path = os.path.join(weights_dir, 'agent_v4_checkpoint.pth')
        self._use_heuristic = True

        for ckpt_path in [best_path, checkpoint_path]:
            if os.path.exists(ckpt_path):
                try:
                    self._agent.load(ckpt_path)
                    self._use_heuristic = False
                    break
                except RuntimeError as e:
                    print(f"   ⚠️  Checkpoint incompatible ({os.path.basename(ckpt_path)}) : {e}")
                    print("   🧠 Fallback mode heuristique")

        if self._use_heuristic:
            print("   🧠 Mode heuristique (pas de checkpoint compatible)")
        else:
            print("   🤖 Mode RL (checkpoint chargé)")

        # Chat monitor (si mode auto ou gdc)
        if self.mode in ('auto', 'gdc'):
            from clashai.social.clan_chat_monitor import ClanChatMonitor
            self._chat_monitor = ClanChatMonitor(
                bot_name=self.bot_name, verbose=self.verbose
            )

        # GdC navigator (si mode auto ou gdc)
        if self.mode in ('auto', 'gdc'):
            from clashai.navigation.gdc_navigator import GdCNavigator
            self._gdc_navigator = GdCNavigator(
                self._models, verbose=self.verbose
            )

        # Clan Castle Manager (V4.1 — demande de troupes)
        from clashai.social.clan_castle import ClanCastleManager
        self._cc_manager = ClanCastleManager(
            models=self._models,
            verbose=self.verbose,
        )

        # Fonctions de navigation de base
        from clashai.navigation import game_loop as gl
        self._classify_screen = gl.classify_screen
        self._adb_screenshot = gl.adb_screenshot
        self._adb_tap = gl.adb_tap

        print("✅ Tous les modules chargés\n")

    # -----------------------------------------------------------------
    #                    BOUCLE PRINCIPALE
    # -----------------------------------------------------------------

    def _main_loop(self, max_episodes=None):
        """
        Le cœur du Brain. Décide quoi faire à chaque instant.
        
        Cycle :
          1. S'assurer qu'on est au village
          2. Vérifier les commandes du chat (si c'est le moment)
          3. Décider de la prochaine action
          4. Exécuter
          5. Pause humaine
          6. Recommencer
        """
        while self._running:
            # --- Limite d'épisodes ---
            if max_episodes and self._attacks_done >= max_episodes:
                print(f"\n🏁 {max_episodes} attaques terminées")
                break

            # --- 1. Retour au village ---
            if not self._ensure_at_village():
                print("   ⚠️  Impossible de revenir au village, retry...")
                time.sleep(5)
                continue

            # --- 2. Vérifier le chat du clan ---
            if self._should_check_chat():
                commands = self._check_clan_chat()

                # Traiter les commandes par priorité
                for cmd in commands:
                    if cmd['type'] == 'attack':
                        self._task_queue.append({
                            'type': 'gdc_attack',
                            'target': cmd['target'],
                            'priority': PRIORITY_GDC_COMMAND,
                            'original_cmd': cmd,  # Pour mark_executed
                        })
                    elif cmd['type'] == 'stop':
                        print("   🛑 Commande d'arrêt reçue")
                        if self._chat_monitor:
                            self._chat_monitor.mark_executed(cmd)
                        self._running = False
                        return

                # Trier par priorité
                self._task_queue.sort(key=lambda t: t['priority'], reverse=True)

            # --- 3. Décider de la prochaine action ---
            if self._task_queue:
                # Exécuter la tâche la plus urgente
                task = self._task_queue.pop(0)
                self._execute_task(task)
            elif self.mode in ('farm', 'auto'):
                # Pas de tâche urgente → farm
                self._execute_task({
                    'type': 'farm_attack',
                    'priority': PRIORITY_FARM_ATTACK,
                })
            elif self.mode == 'gdc':
                # Mode GdC : attendre les commandes
                if self.verbose:
                    print(f"   ⏳ Attente de commandes GdC... "
                          f"(prochain check dans {CHAT_CHECK_INTERVAL}s)")
                time.sleep(CHAT_CHECK_INTERVAL)
                continue

            # --- 4. Pause humaine ---
            if self._running:
                self._human_pause()

    # -----------------------------------------------------------------
    #                    EXÉCUTION DES TÂCHES
    # -----------------------------------------------------------------

    def _execute_task(self, task):
        """Exécute une tâche (attaque farm ou GdC)."""
        task_type = task['type']

        if task_type == 'farm_attack':
            self._do_farm_attack()

        elif task_type == 'gdc_attack':
            target = task['target']
            original_cmd = task.get('original_cmd')

            # Accusé de réception AVANT l'attaque
            if self._chat_monitor:
                self._send_chat_ack(target, before=True)

            # Attaque
            info = self._do_gdc_attack(target)

            # Marquer comme exécutée
            if original_cmd and self._chat_monitor:
                self._chat_monitor.mark_executed(original_cmd)

            # Accusé de réception APRÈS l'attaque (avec résultat)
            if self._chat_monitor and info:
                self._send_chat_ack(target, before=False, result=info)

    def _send_chat_ack(self, target, before=True, result=None):
        """Envoie un message dans le chat du clan."""
        try:
            # Ouvrir le chat
            if self._chat_monitor.open_chat(
                    self._classify_screen, self._models):
                time.sleep(0.3)

                if before:
                    self._chat_monitor.send_chat_message(
                        f"IA - jattaque le {target}"
                    )
                else:
                    stars = result.get('stars', 0)
                    pct = result.get('percentage', 0)
                    self._chat_monitor.send_chat_message(
                        f"IA - {target} fait {stars}e {pct}pct"
                    )

                # Fermer le chat
                self._chat_monitor.close_chat()
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Erreur envoi chat: {e}")

    def _do_farm_attack(self):
        """Exécute une attaque farm (multi classique)."""
        self._attacks_done += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  ⚔️  Attaque farm #{self._attacks_done}")
            print(f"  {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

        # V4.1: demander des troupes CC avant l'attaque
        self._request_cc_troops()

        info = self._run_attack_episode()

        if info:
            stars = info.get('stars', 0)
            pct = info.get('percentage', 0)
            self._total_stars += stars
            self._total_destruction += pct
            self._attacks_since_chat_check += 1

            if self.verbose:
                avg_dest = self._total_destruction / max(self._attacks_done, 1)
                print(f"\n   📊 Farm #{self._attacks_done}: "
                      f"{stars}⭐ {pct}% | "
                      f"Moyenne: {avg_dest:.1f}%")

    def _do_gdc_attack(self, target_number):
        """
        Exécute une attaque GdC sur une cible spécifique.
        
        Returns:
            info: dict avec les résultats, ou None
        """
        self._gdc_attacks_done += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  🏰 Attaque GdC — Cible #{target_number}")
            print(f"  {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

        # V4.1: demander des troupes CC avant l'attaque
        self._request_cc_troops()

        if self._gdc_navigator is None:
            print("   ❌ GdC navigator non initialisé")
            return None

        # Naviguer vers la cible
        success = self._gdc_navigator.attack_target(target_number)

        if not success:
            print(f"   ❌ Navigation vers cible #{target_number} échouée")
            return None

        # L'agent attaque
        info = self._run_attack_episode()

        if info:
            stars = info.get('stars', 0)
            pct = info.get('percentage', 0)
            if self.verbose:
                print(f"\n   📊 GdC #{target_number}: {stars}⭐ {pct}%")

        return info

    def _request_cc_troops(self):
        """
        Demande des troupes de château de clan si le cooldown est passé.
        V4.1: appelé automatiquement avant chaque attaque.
        """
        if self._cc_manager is None:
            return

        try:
            if self._cc_manager._cooldown_ready():
                # S'assurer qu'on est au village
                if not self._ensure_at_village():
                    return
                self._cc_manager.request_if_needed(
                    self._adb_screenshot, self._adb_tap
                )
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Erreur demande CC: {e}")

    def _run_attack_episode(self):
        """
        Exécute un épisode d'attaque complet avec l'agent V4.
        Utilisé pour le farm ET la GdC.
        
        Returns:
            info: dict avec les résultats, ou None si échec
        """
        from clashai.combat.environment_v4 import ClashEnvV4
        from clashai.combat.action_space import MAX_STEPS_SAFETY

        try:
            env = ClashEnvV4(models=self._models, verbose=self.verbose)
            obs, mask = env.reset()
            grid, vector = obs

            if self._use_heuristic:
                # Mode heuristique
                actions = env.get_heuristic_sequence()
                for action in actions:
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break
            else:
                # Mode RL
                for step in range(MAX_STEPS_SAFETY):
                    action, _, _ = self._agent.select_action(grid, vector, mask)
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break

            env.close()
            return info

        except Exception as e:
            print(f"   ❌ Erreur pendant l'attaque : {e}")
            import traceback
            traceback.print_exc()
            return None

    # -----------------------------------------------------------------
    #                    CHAT & NAVIGATION
    # -----------------------------------------------------------------

    def _should_check_chat(self):
        """Détermine s'il faut vérifier le chat maintenant."""
        if self.mode == 'farm':
            return False  # Pas de chat en mode farm pur

        if self._chat_monitor is None:
            return False

        # Vérifier après N attaques ou après un intervalle de temps
        now = time.time()
        time_since_check = now - self._last_chat_check

        if self._attacks_since_chat_check >= ATTACKS_BEFORE_CHAT_CHECK:
            return True
        if time_since_check >= CHAT_CHECK_INTERVAL:
            return True

        return False

    def _check_clan_chat(self):
        """
        Ouvre le chat, lit les commandes, ferme le chat.
        Comme un joueur qui jette un œil au chat entre deux attaques.
        
        Returns:
            commands: liste de commandes détectées
        """
        if self.verbose:
            print("\n   💬 Vérification du chat clan...")

        self._last_chat_check = time.time()
        self._attacks_since_chat_check = 0

        # Ouvrir le chat
        if not self._chat_monitor.open_chat(self._classify_screen, self._models):
            if self.verbose:
                print("   ⚠️  Impossible d'ouvrir le chat")
            return []

        time.sleep(0.5)

        # Lire les commandes
        img = self._adb_screenshot()
        commands = []
        if img is not None:
            commands = self._chat_monitor.check_once(img)

        # Fermer le chat
        self._chat_monitor.close_chat()

        if commands and self.verbose:
            print(f"   📨 {len(commands)} commande(s) trouvée(s)")

        return commands

    def _ensure_at_village(self):
        """
        S'assure qu'on est au village. Navigate si nécessaire.
        
        Returns:
            success: bool
        """
        for attempt in range(15):
            img = self._adb_screenshot()
            if img is None:
                time.sleep(1)
                continue

            state, conf = self._classify_screen(img, self._models)

            if state == 'village_home':
                return True

            # Navigation contextuelle
            if state == 'resultats_attaque':
                # Chercher le bouton vert "Rentrer"
                _img_cv = __import__('cv2').cvtColor(
                    __import__('numpy').array(img), 
                    __import__('cv2').COLOR_RGB2BGR
                )
                for btn_y in [800, 760, 840, 720]:
                    self._adb_tap(960, btn_y)
                    time.sleep(0.3)
                time.sleep(1.5)
            elif state == 'chat_clan':
                self._adb_tap(1400, 400)
                time.sleep(0.5)
                self._adb_tap(960, 400)
                time.sleep(1.5)
            elif state in ('gdc_ally', 'gdc_enemy', 'gdc_ended'):
                try:
                    from clashai.navigation.calibrate_ui import get_position
                    self._adb_tap(*get_position('gdc_return_home'))
                except ImportError:
                    self._adb_tap(80, 780)
                time.sleep(1.5)
            elif state == 'profil':
                try:
                    from clashai.navigation.calibrate_ui import get_position
                    self._adb_tap(*get_position('close_profil'))
                except ImportError:
                    self._adb_tap(1270, 90)
                time.sleep(0.5)
                self._adb_tap(1800, 500)
                time.sleep(1.5)
            elif state == 'menu_boutique':
                try:
                    from clashai.navigation.calibrate_ui import get_position
                    self._adb_tap(*get_position('close_menu'))
                except ImportError:
                    self._adb_tap(1340, 95)
                time.sleep(1.5)
            elif state == 'popup_offre':
                try:
                    from clashai.navigation.calibrate_ui import get_position
                    self._adb_tap(*get_position('close_popup'))
                except ImportError:
                    self._adb_tap(1300, 100)
                time.sleep(1.5)
            elif state == 'chargement':
                time.sleep(3)
            else:
                self._adb_tap(960, 400)
                time.sleep(1.5)

        return False

    # -----------------------------------------------------------------
    #                    COMPORTEMENT HUMAIN
    # -----------------------------------------------------------------

    def _human_pause(self):
        """Pause entre les actions, comme un vrai joueur."""
        wait = random.uniform(IDLE_BETWEEN_ATTACKS, IDLE_BETWEEN_ATTACKS_MAX)

        if self.verbose:
            print(f"\n   😴 Pause ({wait:.0f}s)...")

        elapsed = 0
        while elapsed < wait and self._running:
            action = random.choices(
                ['wait', 'zoom', 'scroll'],
                weights=[0.6, 0.2, 0.2], k=1
            )[0]

            if action == 'zoom':
                try:
                    from clashai.navigation.zoom_control import zoom_in, zoom_out
                    fn = random.choice([zoom_in, zoom_out])
                    fn(scrolls=random.randint(2, 4))
                except ImportError:
                    pass
                pause = random.uniform(1.5, 3.0)

            elif action == 'scroll':
                import subprocess
                x1 = random.randint(400, 1500)
                y1 = random.randint(200, 600)
                dx, dy = random.randint(-120, 120), random.randint(-80, 80)
                try:
                    subprocess.run(
                        ["adb", "shell",
                         f"input swipe {x1} {y1} {x1+dx} {y1+dy} "
                         f"{random.randint(200, 400)}"],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass
                pause = random.uniform(2.0, 4.0)

            else:
                pause = random.uniform(2.0, 5.0)

            time.sleep(pause)
            elapsed += pause

    # -----------------------------------------------------------------
    #                    SHUTDOWN & STATS
    # -----------------------------------------------------------------

    def _shutdown(self):
        """Arrêt propre et affichage des stats."""
        self._running = False
        elapsed = time.time() - self._start_time if self._start_time else 0

        print(f"\n{'='*60}")
        print("  🧠 ClashAI Brain — Arrêt")
        print(f"{'='*60}")
        print(f"   Durée totale    : {elapsed/60:.1f} minutes")
        print(f"   Attaques farm   : {self._attacks_done}")
        print(f"   Attaques GdC    : {self._gdc_attacks_done}")
        print(f"   Étoiles totales : {self._total_stars}")
        if self._attacks_done > 0:
            avg = self._total_destruction / self._attacks_done
            print(f"   Destruction moy : {avg:.1f}%")
        print(f"{'='*60}\n")


# =============================================================================
#                            MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClashAI Brain — IA autonome pour Clash of Clans"
    )
    parser.add_argument(
        '--mode', type=str, default='auto',
        choices=['farm', 'gdc', 'auto'],
        help="farm=attaques multi, gdc=attend commandes clan, auto=tout"
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help="Nombre max d'attaques farm (défaut: infini)"
    )
    parser.add_argument(
        '--bot-name', type=str, default=DEFAULT_BOT_NAME,
        help="Nom du bot pour les commandes clan"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Moins de logs"
    )

    args = parser.parse_args()

    brain = ClashBrain(
        mode=args.mode,
        bot_name=args.bot_name,
        verbose=not args.quiet,
    )
    brain.start(max_episodes=args.episodes)