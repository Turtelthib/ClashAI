# scripts/rl/brain.py
# ClashAI Brain — The single brain of the AI player.
#
# One program, one player, one account.
# The Brain decides what to do at every moment, exactly like a
# human looking at their phone:
#
# "Hey, I'm at the village... let me check the chat...
# Oh, I've been asked to attack #3 in CW... ok let's go...
# Done, back to the village... no new commands...
# Alright, launching a multiplayer attack to farm..."
#
# Main loop:
# 1. Where am I? (screen CNN)
# 2. Is there an urgent command? (clan chat)
# 3. Otherwise, what do I do? (farm, cw, wait)
# 4. Execute the action
# 5. Return to village → repeat
#
# Usage:
# python -m clashai.brain
# python -m clashai.brain --mode farm (multiplayer attacks only)
# python -m clashai.brain --mode gdc (CW only, waits for commands)
# python -m clashai.brain --mode auto (everything: farm + CW + chat)
# python -m clashai.brain --episodes 50 (farm N attacks then stop)

import os
import time
import random
import argparse
from datetime import datetime

# Setup paths
from clashai.paths import RL_WEIGHTS_DIR
# (paths centralized via clashai.paths)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Brain orchestrator constants re-imported from clashai/config/brain.py (Phase A).
from clashai.config import (
    CHAT_CHECK_INTERVAL,
    IDLE_BETWEEN_ATTACKS, IDLE_BETWEEN_ATTACKS_MAX,
    PRIORITY_GDC_COMMAND, PRIORITY_FARM_ATTACK, PRIORITY_IDLE,
    ATTACKS_BEFORE_CHAT_CHECK,
    DEFAULT_BOT_NAME,
)  # noqa: E402


# =============================================================================
# BRAIN
# =============================================================================

class ClashBrain:
    """
    The single brain of ClashAI.

    Manages the whole account like a real player:
    - Farm attacks (multiplayer, for resources)
    - CW commands (from clan chat)
    - Human-like behavior (random pauses, zooms)
    - Robust navigation (always knows how to return to the village)

    Future: village management (upgrades, donations, harvesting)
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

        # Modules (loaded at startup)
        self._models = None
        self._env = None
        self._agent = None
        self._chat_monitor = None
        self._gdc_navigator = None
        self._cc_manager = None

        # Task queue
        self._task_queue = []

        # Cycle counters
        self._attacks_since_chat_check = 0
        self._last_chat_check = 0

    def start(self, max_episodes=None):
        """
        Starts the Brain. This is the only method to call.

        Args:
            max_episodes: max number of farm attacks (None = infinite)
        """
        self._running = True
        self._start_time = time.time()

        print(f"\n{'='*60}")
        print(" ClashAI Brain — Démarrage")
        print(f" Mode : {self.mode}")
        print(f" Bot name : @{self.bot_name}")
        if max_episodes:
            print(f" Max attacks: {max_episodes}")
        print(f" Heure : {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        # 1. Load all modules
        self._load_modules()

        # 2. Main loop
        try:
            self._main_loop(max_episodes)
        except KeyboardInterrupt:
            print("\n\nArrêt demandé (Ctrl+C)")
        finally:
            self._shutdown()

    def _load_modules(self):
        """Loads models and initializes all modules."""
        print("Loading modules...")

        # Perception models
        from clashai.navigation import game_loop
        self._models = game_loop.load_models()

        # Agent V4
        from clashai.combat.agent_v4 import PPOAgentV4
        self._agent = PPOAgentV4()

        # Load best checkpoint if available
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
                    print(f" WARNING: Checkpoint incompatible ({os.path.basename(ckpt_path)}) : {e}")
                    print(" Fallback mode heuristique")

        if self._use_heuristic:
            print(" Mode heuristique (pas de checkpoint compatible)")
        else:
            print(" Mode RL (checkpoint chargé)")

        # Chat monitor (if auto or gdc mode)
        if self.mode in ('auto', 'gdc'):
            from clashai.social.clan_chat_monitor import ClanChatMonitor
            self._chat_monitor = ClanChatMonitor(
                bot_name=self.bot_name, verbose=self.verbose
            )

        # GdC navigator (if auto or gdc mode)
        if self.mode in ('auto', 'gdc'):
            from clashai.navigation.gdc_navigator import GdCNavigator
            self._gdc_navigator = GdCNavigator(
                self._models, verbose=self.verbose
            )

        # Clan Castle Manager (V4.1 — troop request)
        from clashai.social.clan_castle import ClanCastleManager
        self._cc_manager = ClanCastleManager(
            models=self._models,
            verbose=self.verbose,
        )

        # Basic navigation functions
        from clashai.navigation import game_loop as gl
        self._classify_screen = gl.classify_screen
        self._adb_screenshot = gl.adb_screenshot
        self._adb_tap = gl.adb_tap

        print("Tous les modules chargés\n")

    # -----------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------

    def _main_loop(self, max_episodes=None):
        """
        The heart of the Brain. Decides what to do at every moment.

        Cycle:
          1. Make sure we are at the village
          2. Check chat commands (if it is time)
          3. Decide the next action
          4. Execute
          5. Human pause
          6. Repeat
        """
        while self._running:
            # --- Episode limit ---
            if max_episodes and self._attacks_done >= max_episodes:
                print(f"\n{max_episodes} attacks completed")
                break

            # --- 1. Return to village ---
            if not self._ensure_at_village():
                print(" WARNING: Unable to return to village, retry...")
                time.sleep(5)
                continue

            # --- 2. Check clan chat ---
            if self._should_check_chat():
                commands = self._check_clan_chat()

                # Process commands by priority
                for cmd in commands:
                    if cmd['type'] == 'attack':
                        self._task_queue.append({
                            'type': 'gdc_attack',
                            'target': cmd['target'],
                            'priority': PRIORITY_GDC_COMMAND,
                            'original_cmd': cmd,
                        })
                    elif cmd['type'] == 'stop':
                        print(" Stop command received")
                        if self._chat_monitor:
                            self._chat_monitor.mark_executed(cmd)
                        self._running = False
                        return

                # Sort by priority
                self._task_queue.sort(key=lambda t: t['priority'], reverse=True)

            # --- 3. Decide the next action ---
            if self._task_queue:
                # Execute the most urgent task
                task = self._task_queue.pop(0)
                self._execute_task(task)
            elif self.mode in ('farm', 'auto'):
                # No urgent task → farm
                self._execute_task({
                    'type': 'farm_attack',
                    'priority': PRIORITY_FARM_ATTACK,
                })
            elif self.mode == 'gdc':
                # CW mode: wait for commands
                if self.verbose:
                    print(f"  Waiting for CW commands... "
                          f"(next check in {CHAT_CHECK_INTERVAL}s)")
                time.sleep(CHAT_CHECK_INTERVAL)
                continue

            # --- 4. Human pause ---
            if self._running:
                self._human_pause()

    # -----------------------------------------------------------------
    # TASK EXECUTION
    # -----------------------------------------------------------------

    def _execute_task(self, task):
        """Executes a task (farm attack or CW attack)."""
        task_type = task['type']

        if task_type == 'farm_attack':
            self._do_farm_attack()

        elif task_type == 'gdc_attack':
            target = task['target']
            original_cmd = task.get('original_cmd')

            # Acknowledgement BEFORE the attack
            if self._chat_monitor:
                self._send_chat_ack(target, before=True)

            # Attack
            info = self._do_gdc_attack(target)

            # Mark as executed
            if original_cmd and self._chat_monitor:
                self._chat_monitor.mark_executed(original_cmd)

            # Acknowledgement AFTER the attack (with result)
            if self._chat_monitor and info:
                self._send_chat_ack(target, before=False, result=info)

    def _send_chat_ack(self, target, before=True, result=None):
        """Sends a message in the clan chat."""
        try:
            # Open the chat
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

                # Close the chat
                self._chat_monitor.close_chat()
        except Exception as e:
            if self.verbose:
                print(f" WARNING: Erreur envoi chat: {e}")

    def _do_farm_attack(self):
        """Executes a farm attack (classic multiplayer)."""
        self._attacks_done += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Attaque farm #{self._attacks_done}")
            print(f" {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

        # V4.1: request CC troops before the attack
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
                print(f"\n Farm #{self._attacks_done}: "
                      f"{stars}* {pct}% | "
                      f"Average: {avg_dest:.1f}%")

    def _do_gdc_attack(self, target_number):
        """
        Executes a CW attack on a specific target.

        Returns:
            info: dict with results, or None
        """
        self._gdc_attacks_done += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f" Attaque GdC — Cible #{target_number}")
            print(f" {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

        # V4.1: request CC troops before the attack
        self._request_cc_troops()

        if self._gdc_navigator is None:
            print(" ERROR: GdC navigator not initialized")
            return None

        # Navigate to the target
        success = self._gdc_navigator.attack_target(target_number)

        if not success:
            print(f" ERROR: Navigation vers cible #{target_number} échouée")
            return None

        # Agent attacks
        info = self._run_attack_episode()

        if info:
            stars = info.get('stars', 0)
            pct = info.get('percentage', 0)
            if self.verbose:
                print(f"\n GdC #{target_number}: {stars}* {pct}%")

        return info

    def _request_cc_troops(self):
        """
        Requests clan castle troops if the cooldown has passed.
        V4.1: called automatically before each attack.
        """
        if self._cc_manager is None:
            return

        try:
            if self._cc_manager._cooldown_ready():
                # Make sure we are at the village
                if not self._ensure_at_village():
                    return
                self._cc_manager.request_if_needed(
                    self._adb_screenshot, self._adb_tap
                )
        except Exception as e:
            if self.verbose:
                print(f" WARNING: Erreur demande CC: {e}")

    def _run_attack_episode(self):
        """
        Executes a complete attack episode with agent V4.
        Used for both farm AND CW.

        Returns:
            info: dict with results, or None on failure
        """
        from clashai.combat.environment_v4 import ClashEnvV4
        from clashai.combat.action_space import MAX_STEPS_SAFETY

        try:
            env = ClashEnvV4(models=self._models, verbose=self.verbose)
            obs, mask = env.reset()
            grid, vector = obs

            if self._use_heuristic:
                # Heuristic mode
                actions = env.get_heuristic_sequence()
                for action in actions:
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break
            else:
                # RL mode
                for step in range(MAX_STEPS_SAFETY):
                    action, _, _ = self._agent.select_action(grid, vector, mask)
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break

            env.close()
            return info

        except Exception as e:
            print(f" ERROR: Erreur pendant l'attaque : {e}")
            import traceback
            traceback.print_exc()
            return None

    # -----------------------------------------------------------------
    # CHAT & NAVIGATION
    # -----------------------------------------------------------------

    def _should_check_chat(self):
        """Determines whether the chat should be checked now."""
        if self.mode == 'farm':
            return False

        if self._chat_monitor is None:
            return False

        # Check after N attacks or after a time interval
        now = time.time()
        time_since_check = now - self._last_chat_check

        if self._attacks_since_chat_check >= ATTACKS_BEFORE_CHAT_CHECK:
            return True
        if time_since_check >= CHAT_CHECK_INTERVAL:
            return True

        return False

    def _check_clan_chat(self):
        """
        Opens the chat, reads commands, closes the chat.
        Like a player glancing at the chat between attacks.

        Returns:
            commands: list of detected commands
        """
        if self.verbose:
            print("\n  Vérification du chat clan...")

        self._last_chat_check = time.time()
        self._attacks_since_chat_check = 0

        # Open the chat
        if not self._chat_monitor.open_chat(self._classify_screen, self._models):
            if self.verbose:
                print(" WARNING: Unable to open chat")
            return []

        time.sleep(0.5)

        # Read commands
        img = self._adb_screenshot()
        commands = []
        if img is not None:
            commands = self._chat_monitor.check_once(img)

        # Close the chat
        self._chat_monitor.close_chat()

        if commands and self.verbose:
            print(f"  {len(commands)} commande(s) trouvée(s)")

        return commands

    def _ensure_at_village(self):
        """
        Makes sure we are at the village. Navigates if necessary.

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

            # Contextual navigation
            if state == 'resultats_attaque':
                # Look for the green "Return" button
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
    # HUMAN BEHAVIOR
    # -----------------------------------------------------------------

    def _human_pause(self):
        """Pause between actions, like a real player."""
        wait = random.uniform(IDLE_BETWEEN_ATTACKS, IDLE_BETWEEN_ATTACKS_MAX)

        if self.verbose:
            print(f"\n  Pause ({wait:.0f}s)...")

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
                    from clashai.paths import ADB_DEVICE as _ADB_DEV
                    subprocess.run(
                        ["adb", "-s", _ADB_DEV, "shell",
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
    # SHUTDOWN & STATS
    # -----------------------------------------------------------------

    def _shutdown(self):
        """Clean shutdown and stats display."""
        self._running = False
        elapsed = time.time() - self._start_time if self._start_time else 0

        print(f"\n{'='*60}")
        print(" ClashAI Brain — Arrêt")
        print(f"{'='*60}")
        print(f" Total duration : {elapsed/60:.1f} minutes")
        print(f" Farm attacks : {self._attacks_done}")
        print(f" CW attacks : {self._gdc_attacks_done}")
        print(f" Total stars : {self._total_stars}")
        if self._attacks_done > 0:
            avg = self._total_destruction / self._attacks_done
            print(f" Avg destruction : {avg:.1f}%")
        print(f"{'='*60}\n")


# =============================================================================
# MAIN
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