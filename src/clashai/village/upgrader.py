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
# Toutes les classes nécessaires sont dans le CNN UI : `ameliorer`, `annuler`,
# `confirmer_upgrade` (le bouton de confirmation avec prix) et `prix_upgrade`
# (le chiffre, lu par le digit CNN). Flux validé de bout en bout en réel.

import time
from dataclasses import dataclass, field

from clashai.config import ADB_HEIGHT, ADB_WIDTH
from clashai.perception.ui_buttons import DETECTOR_MIN_CONFIDENCE as _MIN_CONF

# _MIN_CONF = confiance minimale pour agir sur un bouton détecté.

# `confirmer_upgrade` = LE bouton de confirmation avec le prix (labo + bâtiment),
# distinct du `ameliorer` du menu et du `confirmer` générique. On le TAPE ; le
# prix, lui, est lu par WidgetReader.read_price_number (qui porte le garde-fou
# anti-confusion avec les compteurs du HUD).
CONFIRM_CLASS = 'confirmer_upgrade'

# Tap neutre pour fermer un menu de bâtiment (ciel en haut, sans bouton).
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

    def __init__(self, detector=None, reader=None, verbose=True, debug_dir=None):
        self._detector = detector
        self._reader = reader
        self.verbose = verbose
        # Si renseigné, chaque étape y dépose une capture ANNOTÉE (boîtes + classes)
        # → on voit exactement l'écran que le bot regarde quand un bouton manque.
        self._debug_dir = debug_dir

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
        # On lit AUSSI les ressources ICI : l'écran de confirmation assombrit le
        # village, ce qui rend les compteurs illisibles (constaté en run réel —
        # `ressources={}` alors qu'ils étaient bien détectés). Le solde ne bouge
        # pas entre les deux écrans, donc la lecture du village fait foi.
        img = screenshot_fn()
        self._dump(img, 'upgrade_1_village')
        builders = reader.read_builders(img) if img is not None else None
        resources = reader.read_resources(img) if img is not None else {}
        if builders is not None and builders[0] <= 0:
            return self._log(UpgradeResult('no_builder', builders=builders,
                                           resources=resources))

        # 2. ouvrir le bâtiment → chercher le bouton `ameliorer`
        tap_fn(*target_xy)
        time.sleep(_D_MENU)
        img = screenshot_fn()
        self._dump(img, 'upgrade_2_menu_batiment')
        hit = self._find(detector, img, 'ameliorer')
        if hit is None:
            tap_fn(*_NEUTRAL_TAP)          # referme le menu bâtiment
            return self._log(UpgradeResult('not_upgradeable', builders=builders,
                                           resources=resources))
        tap_fn(int(hit[0]), int(hit[1]))
        time.sleep(_D_CONFIRM)

        # 3-4. écran de confirmation : prix + décision (étape partagée avec le labo)
        return self._log(self.confirm_step(
            screenshot_fn, tap_fn, resources=resources, builders=builders,
            resource_type=resource_type, confirm_decider=confirm_decider))

    def confirm_step(self, screenshot_fn, tap_fn, resources=None, builders=None,
                     resource_type=None, confirm_decider=None) -> UpgradeResult:
        """Écran de confirmation → décision → `confirmer_upgrade` ou `annuler`.

        Partagé par l'upgrade de bâtiment ET la recherche au labo : les deux
        aboutissent au MÊME écran (`prix_upgrade` + `confirmer_upgrade`), donc
        le garde-fou anti-gemmes n'existe qu'à un seul endroit.

        `resources` doit être lu AVANT l'ouverture du pop-up (il assombrit le
        village et rend les compteurs illisibles).
        """
        detector, reader = self._det(), self._rdr()
        resources = {} if resources is None else resources

        img = screenshot_fn()
        self._dump(img, 'upgrade_3_confirmation')
        # read_price_number, pas read_widget_number : il écarte une détection de
        # prix qui recouvrirait un compteur du HUD (sinon on lirait le solde
        # comme prix → achat confirmé à tort).
        price = reader.read_price_number(img) if img is not None else None

        # ⚠️ PRIORITAIRE : un prix écrit en ROUGE = le JEU signale un solde
        # insuffisant. Signal autoritatif, indépendant de la lecture des chiffres
        # (elle peut perdre un digit et sous-estimer le prix). Il prime même sur
        # un `confirm_decider` qui dirait oui : taper `confirmer` ici ouvrirait le
        # pop-up « acheter des gemmes ».
        if img is not None and reader.price_is_red(img) is True:
            self._cancel(detector, img, tap_fn)
            return UpgradeResult('cant_afford', price=price, resources=resources,
                                 builders=builders)

        # la ressource qui paie est lue à l'écran (couleur de l'icône du prix)
        # si l'appelant ne l'impose pas.
        if resource_type is None and img is not None:
            resource_type = reader.read_price_resource(img)
        decision = self._decide(price, resources, resource_type, confirm_decider)

        if decision is True:
            conf = self._find(detector, img, CONFIRM_CLASS)
            if conf is not None:
                tap_fn(int(conf[0]), int(conf[1]))
                time.sleep(_D_TAP)
            return UpgradeResult('ok', price=price, resources=resources,
                                 builders=builders)

        # sinon on annule (sûr) : décision False (pas les moyens) ou None (inconnu)
        self._cancel(detector, img, tap_fn)
        status = 'cant_afford' if decision is False else 'need_decision'
        return UpgradeResult(status, price=price, resources=resources,
                             builders=builders)

    # ---- helpers -----------------------------------------------------------

    def _dump(self, img, step):
        """Dépose la capture de l'étape : ANNOTÉE + BRUTE (debug_dir seulement).

        La brute compte autant que l'annotée : les boîtes et étiquettes dessinées
        par `annotate` **recouvrent les pixels** (un cadre magenta sur le prix a
        déjà faussé une mesure de couleur et fait perdre un chiffre au moment de
        diagnostiquer). Toute mesure de pixels doit se faire sur `*_raw.png`.
        """
        if not self._debug_dir or img is None:
            return
        import os
        os.makedirs(self._debug_dir, exist_ok=True)
        # `step` porte déjà son contexte (le labo passe 'lab_…') : pas de
        # préfixe en dur, sinon on obtient des 'upgrade_lab_1_village.png'.
        path = os.path.join(self._debug_dir, f'{step}.png')
        try:
            img.save(os.path.join(self._debug_dir, f'{step}_raw.png'))
            self._det().annotate(img, path)
            if self.verbose:
                from clashai.config.logging import pp
                pp(f"   capture annotée -> {path}", tag='ok')
        except Exception as e:
            print(f"WARNING: dump '{step}' impossible ({e})")

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
