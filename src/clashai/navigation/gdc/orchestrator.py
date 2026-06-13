# clashai/navigation/gdc/orchestrator.py
# GdCOrchestrator — chat monitor → CW navigation → V4 agent → return home.

import os

from clashai.navigation.gdc.navigator import GdCNavigator


class GdCOrchestrator:
    """
    Orchestrates a complete CW attack:
    chat monitor → CW navigation → V3 agent → return to village.

    Usage:
        orchestrator = GdCOrchestrator(models)
        orchestrator.run() # Infinite loop: monitors chat and attacks
    """

    def __init__(self, models, bot_name='mini_pekka', verbose=True):
        self.models = models
        self.verbose = verbose

        from clashai.social.clan_chat_monitor import ClanChatMonitor
        self._chat_monitor = ClanChatMonitor(bot_name=bot_name, verbose=verbose)
        self._navigator = GdCNavigator(models, verbose=verbose)

    def handle_command(self, command):
        """
        Executes a command received from the chat.

        Args:
            command: dict {'type': 'attack', 'target': 3, ...}
        """
        if command['type'] == 'attack':
            target = command['target']
            if self.verbose:
                print(f"\nCommand received: attack #{target} in CW")

            success = self._navigator.attack_target(target)

            if success:
                # We are in phase_attaque → launch the V3 agent
                self._run_attack()
            else:
                if self.verbose:
                    print(f" ERROR: Navigation to target #{target} failed")

            # Return to village in all cases
            self._navigator.return_to_village()

        elif command['type'] == 'status':
            if self.verbose:
                print(" Status requested (no action)")

    def _run_attack(self):
        """
        Launches the V4 agent for an attack from phase_attaque.
        Delegates the deploy/combat loop to the shared SSOT runner.
        """
        if self.verbose:
            print("\n Launching V4 attack...")

        from clashai.combat.agent_v4 import PPOAgentV4
        from clashai.combat.episode_runner import run_attack_episode
        from clashai.paths import WEIGHTS_DIR

        # Load the best checkpoint if available (SSOT: <root>/weights/rl).
        agent = PPOAgentV4()
        weights_dir = os.path.join(WEIGHTS_DIR, 'rl')
        use_heuristic = True
        for ckpt_path in (os.path.join(weights_dir, 'agent_v4_best.pth'),
                          os.path.join(weights_dir, 'agent_v4_checkpoint.pth')):
            if os.path.exists(ckpt_path):
                try:
                    agent.load(ckpt_path)
                    use_heuristic = False
                    break
                except RuntimeError:
                    if self.verbose:
                        print(" WARNING: Incompatible checkpoint, heuristic mode")

        info = run_attack_episode(
            self.models,
            agent=None if use_heuristic else agent,
            use_heuristic=use_heuristic,
            verbose=self.verbose,
        )
        if self.verbose and info:
            print(f"\n CW result: {info.get('stars', '?')}* {info.get('percentage', '?')}%")

    def run(self, monitor_interval=30):
        """
        Main loop: monitors the chat and executes commands.

        Args:
            monitor_interval: seconds between each chat check
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print(" ClashAI GdC Orchestrator")
            print(f" Bot: @{self._chat_monitor.bot_name}")
            print(f" Interval: {monitor_interval}s")
            print(f"{'='*50}\n")

        self._chat_monitor.monitor_loop(
            classify_screen_fn=self._navigator._classify_screen,
            models=self.models,
            callback=self.handle_command,
            interval=monitor_interval,
        )
