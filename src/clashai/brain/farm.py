# clashai/brain/farm.py
# BrainFarmMixin — farm attacks, CC troop request, attack episode runner.

from datetime import datetime


class BrainFarmMixin:
    """Farm (multiplayer) attacks + the shared attack-episode runner."""

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
