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
        """
        if self.verbose:
            print("\n Launching V4 attack...")

        try:
            from clashai.combat.environment_v4 import ClashEnvV4
            from clashai.combat.agent_v4 import PPOAgentV4
            from clashai.combat.action_space import MAX_STEPS_SAFETY

            env = ClashEnvV4(models=self.models, verbose=self.verbose)

            # Agent: load the best checkpoint.
            # SSOT: weights live at <root>/weights (clashai.paths.WEIGHTS_DIR),
            # not under src/ — use the canonical constant.
            agent = PPOAgentV4()
            from clashai.paths import WEIGHTS_DIR
            weights_dir = os.path.join(WEIGHTS_DIR, 'rl')
            best_path = os.path.join(weights_dir, 'agent_v4_best.pth')
            checkpoint_path = os.path.join(weights_dir, 'agent_v4_checkpoint.pth')

            heuristic_mode = True
            for ckpt_path in [best_path, checkpoint_path]:
                if os.path.exists(ckpt_path):
                    try:
                        agent.load(ckpt_path)
                        heuristic_mode = False
                        break
                    except RuntimeError:
                        if self.verbose:
                            print(f" WARNING: Incompatible checkpoint, heuristic mode")

            # Reset (resumes from phase_attaque)
            obs, mask = env.reset()
            grid, vector = obs

            # Heuristic or RL depending on whether a checkpoint exists
            heuristic_mode = not os.path.exists(best_path) and not os.path.exists(checkpoint_path)

            if heuristic_mode:
                actions = env.get_heuristic_sequence()
                for action in actions:
                    obs, mask, reward, done, info = env.step(action)
                    if done:
                        break
            else:
                for step in range(MAX_STEPS_SAFETY):
                    action, _, _ = agent.select_action(grid, vector, mask)
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break

            if self.verbose:
                stars = info.get('stars', '?')
                pct = info.get('percentage', '?')
                print(f"\n CW result: {stars}* {pct}%")

            env.close()

        except Exception as e:
            if self.verbose:
                print(f" ERROR: V3 attack error: {e}")
                import traceback
                traceback.print_exc()

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
