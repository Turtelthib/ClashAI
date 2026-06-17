# clashai/brain/war.py
# BrainWarMixin — Clan War (GdC) attacks on a specific target.

from datetime import datetime


class BrainWarMixin:
    """CW attacks driven by a target number (from clan chat commands).

    [DEAD-CODE-V5.1] superseded by GdCAgent (V5.1 scheduler migration) —
    remove in Étape B. (grep "DEAD-CODE-V5.1".)
    """

    def _do_gdc_attack(self, target_number):  # [DEAD-CODE-V5.1] → GdCAgent
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
