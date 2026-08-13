# clashai/village/collector.py
# Récolte des ressources du village via le CNN UI (V5.2, increment 1).
#
# Quand un collecteur (mine d'or, collecteur d'élixir, foreuse d'élixir noir) est
# plein, une petite icône flottante apparaît au-dessus. Le CNN UI les détecte
# toutes d'un coup (classes recolter_or / recolter_elixir / recolter_elixir_noire)
# → on tape chaque icône détectée. Aucune dépense, action sûre et idempotente.
#
# On réutilise le détecteur déjà branché (ui_buttons.set_detector au démarrage)
# pour ne pas recharger le modèle YOLO ; fallback : une instance dédiée.

import time

# Classes CNN des icônes de récolte (une icône = un tap).
RESOURCE_CLASSES = ('recolter_or', 'recolter_elixir', 'recolter_elixir_noire')

# Petit délai entre deux taps (laisse l'animation de récolte se jouer).
TAP_DELAY = 0.3


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

    def collect(self, screenshot_fn, tap_fn) -> int:
        """Détecte puis tape toutes les icônes de récolte présentes.

        Args:
            screenshot_fn: () -> PIL.Image | None (frame courante du village).
            tap_fn:        (x, y) -> None (tap ADB).

        Returns:
            Nombre d'icônes tapées (0 si rien à récolter / pas de frame).
        """
        img = screenshot_fn()
        if img is None:
            return 0

        raw = self._get_detector().detect_raw(img)

        taps = 0
        by_class = {}
        for cls in RESOURCE_CLASSES:
            dets = raw.get(cls, [])
            by_class[cls] = len(dets)
            for d in dets:
                tap_fn(d.x, d.y)
                taps += 1
                time.sleep(TAP_DELAY)

        if self.verbose and taps:
            from clashai.config.logging import pp
            detail = ', '.join(f"{n}×{c.replace('recolter_', '')}"
                               for c, n in by_class.items() if n)
            pp(f" Village : {taps} ressources récoltées ({detail})", tag='ok')

        return taps
