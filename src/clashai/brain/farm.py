# clashai/brain/farm.py
# BrainFarmMixin — farm attacks, CC troop request, attack episode runner.

from datetime import datetime


class BrainFarmMixin:
    """Farm (multiplayer) attacks + the shared attack-episode runner.

    [DEAD-CODE-V5.1] All methods below are superseded by CombatAgent +
    ClanCastleAgent (V5.1 scheduler migration) — remove in Étape B.
    (grep "DEAD-CODE-V5.1" to find every method to delete.)
    """

    def _do_farm_attack(self):  # [DEAD-CODE-V5.1] → CombatAgent
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

    def _request_cc_troops(self):  # [DEAD-CODE-V5.1] → ClanCastleAgent
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

    def _run_attack_episode(self):  # [DEAD-CODE-V5.1] → combat.episode_runner
        """
        Executes a complete attack episode with agent V4. Used for farm AND CW.
        Delegates to the shared SSOT runner (also used by CombatAgent, V5.1).

        Returns:
            info: dict with results, or None on failure
        """
        from clashai.combat.episode_runner import run_attack_episode
        return run_attack_episode(
            self._models,
            agent=self._agent,
            use_heuristic=self._use_heuristic,
            verbose=self.verbose,
        )
