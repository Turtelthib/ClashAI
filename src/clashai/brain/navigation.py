# clashai/brain/navigation.py
# BrainNavigationMixin — "always get back to the village" + human-like pauses.

import random
import time

from clashai.config import IDLE_BETWEEN_ATTACKS, IDLE_BETWEEN_ATTACKS_MAX


class BrainNavigationMixin:
    """Robust return-to-village navigation + human behavior between actions."""

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
