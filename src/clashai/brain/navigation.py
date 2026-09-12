# clashai/brain/navigation.py
# BrainNavigationMixin — "always get back to the village" + human-like pauses.

import random
import time

from clashai.config import IDLE_BETWEEN_ATTACKS, IDLE_BETWEEN_ATTACKS_MAX
from clashai.perception.ui_buttons import find_button

# Backends qui lisent l'ÉCRAN PHYSIQUE, pas la fenêtre : si l'émulateur est
# minimisé ou masqué, ils rendent une image du bureau — et le bot agit dessus.
SCREEN_REGION_BACKENDS = ('mss', 'dxcam')


def navigation_diagnosis(no_capture, attempts, states, backend=None):
    """Pourquoi on n'a pas atteint le village. Message destiné à l'utilisateur.

    ⚠️ Sans ça, le bot annonçait « Unable to return to village » — un problème
    de NAVIGATION — alors qu'il n'avait reçu AUCUNE image (émulateur minimisé,
    ADB déconnecté). On cherchait le défaut au mauvais endroit.
    """
    if no_capture >= attempts:
        return ("aucune capture d'écran (0 image sur "
                f"{attempts} essais) — ce n'est PAS un problème de navigation. "
                "Vérifie que l'émulateur tourne et n'est pas MINIMISÉ (derrière "
                "une autre fenêtre, c'est bon), ou qu'un appareil ADB est "
                "connecté (`adb devices`).")

    loading_heavy = states and states.count('chargement') > len(states) / 2
    if backend in SCREEN_REGION_BACKENDS and loading_heavy:
        return (f"backend « {backend} » + écran vu comme « chargement » en "
                "boucle : c'est la signature d'une capture du BUREAU au lieu du "
                "jeu. L'émulateur est probablement minimisé. "
                "Voir TROUBLESHOOTING « Capture fenêtre émulateur occluded ».")

    vus = ', '.join(sorted(set(states))) if states else 'aucun'
    return (f"village jamais atteint en {attempts} essais "
            f"(écrans vus : {vus})")


class BrainNavigationMixin:
    """Robust return-to-village navigation + human behavior between actions."""

    def _ensure_at_village(self):
        """
        Makes sure we are at the village. Navigates if necessary.

        Returns:
            success: bool
        """
        attempts = 15
        no_capture = 0
        states = []
        self.last_navigation_error = None

        for attempt in range(attempts):
            img = self._adb_screenshot()
            if img is None:
                no_capture += 1
                time.sleep(1)
                continue

            state, conf = self._classify_screen(img, self._models)
            states.append(state)

            if state == 'village_home':
                return True

            # Contextual navigation.
            # Tout passe par find_button(key, screenshot=img) : le CNN UI cherche
            # le bouton RÉELLEMENT à l'écran et, s'il ne trouve rien d'assez sûr,
            # find_button retombe sur la position calibrée. Plus de try/except
            # ImportError ni de coordonnées de secours en dur : find_button rend
            # toujours une position.
            if state == 'resultats_attaque':
                # Le bouton "Rentrer" est une classe du CNN. Avant, on tapait 4
                # hauteurs à l'aveugle en espérant tomber dessus.
                self._adb_tap(*find_button('return_home', screenshot=img))
                time.sleep(1.5)
            elif state == 'chat_clan':
                self._adb_tap(*find_button('chat_close_tap', screenshot=img))
                time.sleep(0.5)
                self._adb_tap(960, 400)
                time.sleep(1.5)
            elif state in ('gdc_ally', 'gdc_enemy', 'gdc_ended'):
                self._adb_tap(*find_button('gdc_return_home', screenshot=img))
                time.sleep(1.5)
            elif state == 'profil':
                self._adb_tap(*find_button('close_profil', screenshot=img))
                time.sleep(0.5)
                self._adb_tap(1800, 500)
                time.sleep(1.5)
            elif state == 'menu_boutique':
                self._adb_tap(*find_button('close_menu', screenshot=img))
                time.sleep(1.5)
            elif state == 'popup_offre':
                self._adb_tap(*find_button('close_popup', screenshot=img))
                time.sleep(1.5)
            elif state == 'chargement':
                time.sleep(3)
            else:
                self._adb_tap(960, 400)
                time.sleep(1.5)

        backend = None
        try:
            from clashai.perception.screen_capture import get_capture
            backend = get_capture().backend
        except Exception:
            pass
        self.last_navigation_error = navigation_diagnosis(
            no_capture, attempts, states, backend)
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
