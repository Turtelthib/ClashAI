# clashai/village/upgrader.py
# Exécuteur d'améliorations de bâtiments (V5.2, increment 2) — les "mains".
#
# La DÉCISION (quoi améliorer) reviendra au LLM (V5.3). Ici on fournit le GESTE
# + les capteurs proactifs, conçus comme des outils appelables :
#   - free_builders(img)  -> int | None            (via nombre_ouvrier)
#   - resources(img)      -> {or, elixir, ...}      (via compteur_*)
#   - upgrade_building(...) -> UpgradeResult        (le flux complet)
#
# Flux upgrade_building :
#   1. constructeur libre ? (0 → no_builder, gating proactif fiable)
#   2. tap bâtiment → cherche le bouton `ameliorer` (absent → not_upgradeable)
#   3. tap `ameliorer` → écran de confirmation : lit prix + ressources
#   4. DÉCISION d'affordabilité → `confirmer` (ok) ou `annuler`
#
# ⚠️ SÛR PAR DÉFAUT : sans décision d'affordabilité prouvée, on ANNULE (jamais de
# tap `confirmer` à l'aveugle → pas de pop-up "acheter des gemmes"). La décision
# vient soit d'un `resource_type` + prix lisible, soit d'un `confirm_decider`
# fourni par l'appelant (LLM / démo).
#
# Boutons `ameliorer` / `confirmer` / `annuler` : classes déjà dans le CNN UI
# (124). Prix : classe `prix_upgrade` à AJOUTER au dataset (lue comme les
# compteurs) ; d'ici là le prix est None → décision déférée (annulation sûre).

import time
from dataclasses import dataclass, field

# Confiance minimale pour agir sur un bouton détecté.
from clashai.perception.ui_buttons import DETECTOR_MIN_CONFIDENCE as _MIN_CONF

# Classes CNN de l'écran de confirmation (à ajouter au dataset) :
#   - confirmer_upgrade : LE bouton de confirmation avec le prix (labo + bâtiment),
#     distinct du `ameliorer` du menu et du `confirmer` générique. On le TAPE.
#   - prix_upgrade      : box serrée sur le CHIFFRE du prix. On la LIT (digit CNN).
CONFIRM_CLASS = 'confirmer_upgrade'
PRICE_CLASS = 'prix_upgrade'

# Tap neutre pour fermer un menu de bâtiment (ciel en haut, sans bouton).
from clashai.config import ADB_HEIGHT, ADB_WIDTH  # noqa: E402
_NEUTRAL_TAP = (ADB_WIDTH // 2, int(ADB_HEIGHT * 0.15))

# Délais laissant l'UI s'ouvrir (menu bâtiment / écran de confirmation).
_D_MENU = 0.8
_D_CONFIRM = 1.0
_D_TAP = 0.4


@dataclass
class UpgradeResult:
    """Télémétrie d'une tentative d'upgrade (retour d'outil pour le LLM)."""
    status: str                       # ok | no_builder | not_upgradeable |
    #                                   cant_afford | need_decision | error
    price: int = None
    resources: dict = field(default_factory=dict)
    builders: tuple = None            # (libres, total)


class VillageUpgrader:
    """Ouvre un bâtiment et lance son amélioration si les conditions sont là."""

    def __init__(self, detector=None, reader=None, verbose=True):
        self._detector = detector
        self._reader = reader
        self.verbose = verbose

    def _det(self):
        if self._detector is None:
            from clashai.perception.ui_buttons import get_detector
            self._detector = get_detector()
            if self._detector is None:
                from clashai.perception.ui_detector import UIDetector
                self._detector = UIDetector(verbose=False)
        return self._detector

    def _rdr(self):
        if self._reader is None:
            from clashai.perception.widget_reader import WidgetReader
            self._reader = WidgetReader(detector=self._det())
        return self._reader

    # ---- capteurs (outils LLM) --------------------------------------------

    def free_builders(self, screenshot_pil):
        """Nombre de constructeurs libres, ou None si illisible."""
        b = self._rdr().read_builders(screenshot_pil)
        return b[0] if b else None

    def resources(self, screenshot_pil) -> dict:
        return self._rdr().read_resources(screenshot_pil)

    # ---- exécuteur ---------------------------------------------------------

    def upgrade_building(self, target_xy, screenshot_fn, tap_fn,
                         resource_type=None, confirm_decider=None) -> UpgradeResult:
        """Tente d'améliorer le bâtiment à `target_xy`. Voir statuts en tête."""
        detector, reader = self._det(), self._rdr()

        # 1. constructeur libre ? (gating proactif fiable)
        img = screenshot_fn()
        builders = reader.read_builders(img) if img is not None else None
        if builders is not None and builders[0] <= 0:
            return self._log(UpgradeResult('no_builder', builders=builders))

        # 2. ouvrir le bâtiment → chercher le bouton `ameliorer`
        tap_fn(*target_xy)
        time.sleep(_D_MENU)
        img = screenshot_fn()
        hit = self._find(detector, img, 'ameliorer')
        if hit is None:
            tap_fn(*_NEUTRAL_TAP)          # referme le menu bâtiment
            return self._log(UpgradeResult('not_upgradeable', builders=builders))
        tap_fn(int(hit[0]), int(hit[1]))
        time.sleep(_D_CONFIRM)

        # 3. écran de confirmation : prix + ressources
        img = screenshot_fn()
        resources = reader.read_resources(img) if img is not None else {}
        price = reader.read_widget_number(img, PRICE_CLASS) if img is not None else None

        # 4. décision d'affordabilité — la ressource qui paie est lue à l'écran
        # (couleur de l'icône du prix) si l'appelant ne l'impose pas.
        if resource_type is None and img is not None:
            resource_type = reader.read_price_resource(img)
        decision = self._decide(price, resources, resource_type, confirm_decider)
        if decision is True:
            conf = self._find(detector, img, CONFIRM_CLASS)
            if conf is not None:
                tap_fn(int(conf[0]), int(conf[1]))
                time.sleep(_D_TAP)
            return self._log(UpgradeResult('ok', price=price, resources=resources,
                                           builders=builders))

        # sinon on annule (sûr) : décision False (pas les moyens) ou None (inconnu)
        self._cancel(detector, img, tap_fn)
        status = 'cant_afford' if decision is False else 'need_decision'
        return self._log(UpgradeResult(status, price=price, resources=resources,
                                       builders=builders))

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _find(detector, img, name):
        """(x, y, conf) d'un bouton CNN si détecté ≥ seuil, sinon None."""
        if img is None:
            return None
        hit = detector.detect(img).get(name)
        if hit is not None and hit[2] >= _MIN_CONF:
            return hit
        return None

    def _cancel(self, detector, img, tap_fn):
        """Ferme l'écran de confirmation via `annuler`, sinon tap neutre."""
        hit = self._find(detector, img, 'annuler')
        if hit is not None:
            tap_fn(int(hit[0]), int(hit[1]))
        else:
            tap_fn(*_NEUTRAL_TAP)
        time.sleep(_D_TAP)

    @staticmethod
    def _decide(price, resources, resource_type, confirm_decider):
        """True = confirmer, False = pas les moyens, None = indécidable (→ annule).

        Priorité au décideur fourni (LLM). Sinon, ne confirme QUE si on peut
        prouver l'affordabilité (prix lu + ressource cible connue).
        """
        if confirm_decider is not None:
            return bool(confirm_decider(price, resources))
        if price is None or resource_type is None:
            return None                     # pas de preuve → annulation sûre
        have = resources.get(resource_type)
        if have is None:
            return None
        return have >= price

    def _log(self, result):
        if self.verbose:
            from clashai.config.logging import pp
            pp(f" Upgrade: {result.status}"
               + (f" (prix {result.price})" if result.price else ""), tag='ok')
        return result
