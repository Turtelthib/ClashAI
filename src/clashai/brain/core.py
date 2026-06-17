# clashai/brain/core.py
# BrainCoreMixin — lifecycle: init, module loading, start, shutdown.

import os
import time
from datetime import datetime

from clashai.paths import RL_WEIGHTS_DIR
from clashai.config import DEFAULT_BOT_NAME


class BrainCoreMixin:
    """Lifecycle + module loading for ClashBrain."""

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

        # V5.1: assemble the agent scheduler + swappable Brain.
        # Each agent wraps an already-loaded module (no double init).
        from clashai.agents import (
            AgentScheduler, CombatAgent, GdCAgent, ClanCastleAgent, ChatAgent,
        )
        from clashai.brain.interface import HeuristicBrain

        self._scheduler = AgentScheduler()

        # Combat (farm) — always available (also farms loot for war modes).
        self._scheduler.register(CombatAgent(
            models=self._models, agent=self._agent,
            use_heuristic=self._use_heuristic, verbose=self.verbose,
        ))

        # Clan castle — always (CC troops help farm + war).
        self._scheduler.register(ClanCastleAgent(
            manager=self._cc_manager, models=self._models,
            screenshot_fn=self._adb_screenshot, tap_fn=self._adb_tap,
            verbose=self.verbose,
        ))

        # War + chat only in auto/gdc modes.
        if self.mode in ('auto', 'gdc'):
            gdc_agent = GdCAgent(
                models=self._models, navigator=self._gdc_navigator,
                agent=self._agent, use_heuristic=self._use_heuristic,
                verbose=self.verbose,
            )
            self._scheduler.register(gdc_agent)
            self._scheduler.register(ChatAgent(
                monitor=self._chat_monitor, models=self._models,
                on_attack=gdc_agent.enqueue_target,
                on_stop=lambda: setattr(self, '_running', False),
                verbose=self.verbose,
            ))

        self._brain = HeuristicBrain(self._scheduler)

        print("Tous les modules chargés\n")

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
