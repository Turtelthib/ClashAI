# clashai/village/collector.py
# Récolte des ressources du village via le CNN UI (V5.2, increment 1).
#
# Quand un collecteur (mine d'or, collecteur d'élixir, foreuse d'élixir noir) est
# plein, une petite icône flottante apparaît au-dessus (classes CNN
# recolter_or / recolter_elixir / recolter_elixir_noire).
#
# ⚠️ Méca CoC : taper UNE icône de récolte en récolte automatiquement d'autres.
# Donc on ne tape PAS toutes les positions détectées d'un coup — après le 1er tap
# les autres icônes disparaissent, et taper leur ancienne position taperait un
# bâtiment au hasard. On boucle : capture FRAÎCHE → tape UNE icône → attend →
# re-scanne, jusqu'à ce qu'il n'y ait plus rien. Robuste aux deux cas (un tap
# vide tout → 1 passe ; un tap vide un seul → plusieurs passes).
#
# On réutilise le détecteur déjà branché (ui_buttons.set_detector au démarrage)
# pour ne pas recharger le modèle YOLO ; fallback : une instance dédiée.

import time

# Classes CNN des icônes de récolte.
RESOURCE_CLASSES = ('recolter_or', 'recolter_elixir', 'recolter_elixir_noire')

# Délai après un tap : laisse la récolte se propager + l'icône disparaître avant
# la capture suivante (sinon on re-détecte une icône déjà partie).
PASS_DELAY = 0.6

# Garde-fou : nombre max de passes (un village se vide en 1-2 taps en pratique).
MAX_PASSES = 8

# Rayon (px ADB) sous lequel deux détections successives = "la même icône" → si
# rien n'a bougé après un tap, on arrête (évite de re-taper une icône bloquée).
_SAME_ICON_PX = 40


class VillageCollector:
    """Récolte les ressources visibles à l'écran du village."""

    def __init__(self, detector=None, verbose=True):
        self._detector = detector
        self.verbose = verbose

    def _get_detector(self):
        """Détecteur branché en priorité, sinon instance dédiée (lazy)."""
        if self._detector is None:
            from clashai.perception.ui_buttons import get_detector
            self._detector = get_detector()
            if self._detector is None:
                from clashai.perception.ui_detector import UIDetector
                self._detector = UIDetector(verbose=self.verbose)
        return self._detector

    @staticmethod
    def _best_icon(raw):
        """Meilleure icône de récolte (conf max, toutes classes), ou None."""
        cands = []
        for cls in RESOURCE_CLASSES:
            cands.extend(raw.get(cls, []))
        return max(cands, key=lambda d: d.conf) if cands else None

    def collect(self, screenshot_fn, tap_fn) -> int:
        """Récolte en boucle avec re-scan entre chaque tap.

        Args:
            screenshot_fn: () -> PIL.Image | None (frame courante du village).
            tap_fn:        (x, y) -> None (tap ADB).

        Returns:
            Nombre de taps effectués (0 si rien à récolter / pas de frame).
        """
        detector = self._get_detector()
        taps = 0
        last = None

        for _ in range(MAX_PASSES):
            img = screenshot_fn()
            if img is None:
                break
            icon = self._best_icon(detector.detect_raw(img))
            if icon is None:
                break
            # Rien n'a bougé depuis le tap précédent → icône bloquée, on arrête.
            if last is not None and abs(icon.x - last[0]) < _SAME_ICON_PX \
                    and abs(icon.y - last[1]) < _SAME_ICON_PX:
                break
            tap_fn(icon.x, icon.y)
            last = (icon.x, icon.y)
            taps += 1
            time.sleep(PASS_DELAY)

        if self.verbose and taps:
            from clashai.config.logging import pp
            pp(f" Village : récolte effectuée ({taps} tap(s))", tag='ok')

        return taps
