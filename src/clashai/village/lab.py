# clashai/village/lab.py
# Recherches au laboratoire (V5.2, increment 3) — les "mains" du labo.
#
# Comme pour les bâtiments, la DÉCISION (quelle troupe améliorer) reviendra au
# LLM (V5.3) ; ici on fournit les capteurs + le geste, en outils appelables :
#   - is_free(img)          -> bool | None      (via `place_labo`, "1/1" = libre)
#   - find_lab(img, models) -> (x, y) | None    (via le CNN bâtiments)
#   - list_candidates(...)  -> [LabCandidate]   (troupes améliorables + prix)
#   - research(...)         -> UpgradeResult    (le flux complet)
#
# Flux :
#   1. labo libre ? (`place_labo` = "1/1" → libre ; "0/1" → recherche en cours)
#   2. localiser le laboratoire dans le village (CNN bâtiments, pas de coords
#      en dur) puis le taper
#   3. taper `rechercher` → écran « Choisissez les troupes à améliorer »
#   4. lister les cartes AMÉLIORABLES : le CNN de la barre de troupes les nomme
#      (80 classes) et sa saturation HSV sépare couleur (améliorable) / gris
#      ("Laboratoire de niveau X nécessaire")
#   5. choisir (politique injectée — le LLM plus tard), taper la carte
#   6. écran de confirmation → `VillageUpgrader.confirm_step` (PARTAGÉ) : même
#      lecture de prix, même décision d'affordabilité, même garde-fou
#      anti-gemmes que l'upgrade de bâtiment.
#
# Limite v1 assumée : seules les cartes visibles à l'ouverture sont considérées
# (pas de scroll horizontal). Le scroll est un item séparé de la ROADMAP.

import time
from dataclasses import dataclass

from clashai.config import ADB_HEIGHT, ADB_WIDTH

SEARCH_BUTTON = 'rechercher'     # ouvre le menu du labo
LAB_BUILDING = 'laboratoire'     # classe du CNN bâtiments

# Délais laissant l'UI s'ouvrir.
_D_MENU = 1.0
_D_GRID = 1.0
_D_TAP = 0.4

# Le prix est écrit en bas de la carte : on lit cette bande.
_PRICE_BAND_TOP = 0.62      # fraction de la hauteur de carte où commence le prix


@dataclass
class LabCandidate:
    """Une carte de troupe améliorable sur l'écran du labo."""
    name: str                 # nom CNN ('barbare', 'dragon', 'soin'…)
    x: int                    # centre de la carte (ADB)
    y: int
    price: int = None         # prix lu sur la carte (None si illisible)
    conf: float = 0.0

    def __repr__(self):
        p = format(self.price, ',').replace(',', ' ') if self.price else '?'
        return f"<{self.name} @({self.x},{self.y}) prix={p}>"


class VillageLab:
    """Lance une recherche au laboratoire."""

    def __init__(self, detector=None, reader=None, upgrader=None,
                 verbose=True, debug_dir=None):
        self._detector = detector
        self._reader = reader
        self._upgrader = upgrader
        self.verbose = verbose
        self._debug_dir = debug_dir

    # ---- dépendances (lazy, injectables pour les tests) --------------------

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

    def _upg(self):
        if self._upgrader is None:
            from clashai.village.upgrader import VillageUpgrader
            self._upgrader = VillageUpgrader(
                detector=self._det(), reader=self._rdr(),
                verbose=self.verbose, debug_dir=self._debug_dir)
        return self._upgrader

    # ---- capteurs (outils LLM) --------------------------------------------

    def is_free(self, screenshot_pil):
        """True si aucune recherche ne tourne, None si illisible.

        `place_labo` = "N/M" avec N = places DISPONIBLES (même convention que
        les ouvriers) : "1/1" → labo libre, "0/1" → recherche en cours.
        """
        ratio = self._rdr().read_labs(screenshot_pil)
        return None if ratio is None else ratio[0] >= 1

    @staticmethod
    def find_lab(screenshot_pil, models):
        """(x, y) ADB du laboratoire via le CNN bâtiments, ou None."""
        if not models:
            return None
        from clashai.navigation.game_loop import analyze_village
        try:
            buildings = analyze_village(screenshot_pil, models)
        except Exception:
            return None
        labs = [b for b in (buildings or []) if b.get('class') == LAB_BUILDING]
        if not labs:
            return None
        best = max(labs, key=lambda b: b.get('confidence', 0.0))
        cx, cy = best['center']
        iw, ih = screenshot_pil.size
        return (int(cx * ADB_WIDTH / iw), int(cy * ADB_HEIGHT / ih))

    def list_candidates(self, screenshot_pil, models) -> list:
        """Cartes AMÉLIORABLES de l'écran labo (nom + position + prix).

        Le CNN de la barre de troupes nomme les vignettes ; son test de
        saturation HSV (`is_grayed`) écarte les cartes grises « Laboratoire de
        niveau X nécessaire ». Une carte non reconnue est simplement ignorée.
        """
        bar = (models or {}).get('troop_bar_detector')
        if bar is None:
            return []
        try:
            dets = bar.detect(screenshot_pil)
        except Exception:
            return []

        iw, ih = screenshot_pil.size
        sx, sy = ADB_WIDTH / iw, ADB_HEIGHT / ih
        out = []
        for d in dets:
            if d.get('is_grayed') or d.get('no_tap'):
                continue                      # gris = labo trop bas / indispo
            price = self._read_card_price(screenshot_pil, d['bbox'])
            cx, cy = d['center']
            out.append(LabCandidate(
                name=d['name'], x=int(cx * sx), y=int(cy * sy),
                price=price, conf=d.get('conf', 0.0),
            ))
        out.sort(key=lambda c: (c.y, c.x))    # ordre de lecture
        return out

    def _read_card_price(self, screenshot_pil, bbox):
        """Prix écrit en bas de la carte (None si illisible)."""
        from clashai.perception import digit_reader
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h < 20:
            return None
        top = y1 + int(h * _PRICE_BAND_TOP)
        crop = screenshot_pil.crop((x1, top, x2, y2))
        if crop.size[0] < 6 or crop.size[1] < 6:
            return None
        n, _ = digit_reader.read_widget_number(crop)
        return n

    # ---- confirmation visuelle --------------------------------------------

    def annotate_scan(self, screenshot_pil, models, out_path):
        """Dépose une capture annotée des CARTES du labo. Renvoie out_path|None.

        Le `annotate()` du détecteur UI dessine les boutons ; ici ce qui compte
        ce sont les vignettes de troupes. On dessine donc les détections du CNN
        barre de troupes : **vert** = améliorable (avec le prix lu), **gris** =
        non améliorable (labo trop bas). Voir d'un coup d'œil ce que le bot voit
        évite de deviner quand une carte manque ou qu'un prix est illisible.
        """
        bar = (models or {}).get('troop_bar_detector')
        if bar is None or screenshot_pil is None:
            return None
        try:
            import os

            import cv2
            import numpy as np

            dets = bar.detect(screenshot_pil)
            img = cv2.cvtColor(np.asarray(screenshot_pil.convert('RGB')),
                               cv2.COLOR_RGB2BGR)
            for d in dets:
                x1, y1, x2, y2 = d['bbox']
                grayed = d.get('is_grayed')
                color = (150, 150, 150) if grayed else (60, 200, 60)   # BGR
                if grayed:
                    label = f"{d['name']} (grise)"
                else:
                    price = self._read_card_price(screenshot_pil, d['bbox'])
                    shown = format(price, ',').replace(',', ' ') if price else '?'
                    label = f"{d['name']} {shown}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6, 2)
                ty = max(th + 4, y1 - 4)
                cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2),
                              color, -1)
                cv2.putText(img, label, (x1 + 2, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

            d = os.path.dirname(out_path)
            if d:
                os.makedirs(d, exist_ok=True)
            # la brute aussi : les boîtes dessinées recouvrent les pixels, donc
            # toute mesure de couleur/chiffre doit se faire sur `*_raw.png`.
            base, ext = os.path.splitext(out_path)
            screenshot_pil.save(f'{base}_raw{ext or ".png"}')
            cv2.imwrite(out_path, img)
            return out_path
        except Exception as e:
            print(f"WARNING: annotation du scan impossible ({e})")
            return None

    # ---- exécuteur ---------------------------------------------------------

    def research(self, screenshot_fn, tap_fn, models,
                 choose=None, confirm_decider=None):
        """Lance une recherche au labo. Renvoie un `UpgradeResult`.

        Args:
            choose: (candidats) -> LabCandidate | None. Défaut : le moins cher
                    dont le prix est lisible. Le LLM branchera sa politique ici.
            confirm_decider: (prix, ressources) -> bool, transmis à confirm_step.

        Statuts : ok | busy | lab_not_found | menu_not_found |
                  nothing_upgradable | cant_afford | need_decision
        """
        from clashai.village.upgrader import UpgradeResult
        upg = self._upg()
        detector, reader = self._det(), self._rdr()

        # 1. labo occupé ? + ressources lues sur l'écran clair (le pop-up
        #    assombrit le village et rend les compteurs illisibles).
        img = screenshot_fn()
        upg._dump(img, 'lab_1_village')
        if img is None:
            return self._log(UpgradeResult('lab_not_found'))
        resources = reader.read_resources(img)
        if self.is_free(img) is False:
            return self._log(UpgradeResult('busy', resources=resources))

        # 2. localiser puis ouvrir le laboratoire
        lab_xy = self.find_lab(img, models)
        if lab_xy is None:
            return self._log(UpgradeResult('lab_not_found', resources=resources))
        tap_fn(*lab_xy)
        time.sleep(_D_MENU)

        # 3. bouton `rechercher` → grille des troupes
        img = screenshot_fn()
        upg._dump(img, 'lab_2_menu')
        hit = upg._find(detector, img, SEARCH_BUTTON)
        if hit is None:
            tap_fn(ADB_WIDTH // 2, int(ADB_HEIGHT * 0.15))   # referme
            return self._log(UpgradeResult('menu_not_found', resources=resources))
        tap_fn(int(hit[0]), int(hit[1]))
        time.sleep(_D_GRID)

        # 4. candidats améliorables
        img = screenshot_fn()
        if self._debug_dir and img is not None:
            import os
            self.annotate_scan(
                img, models, os.path.join(self._debug_dir, 'lab_3_grille.png'))
        candidates = self.list_candidates(img, models) if img is not None else []
        pick = (choose or self._cheapest)(candidates) if candidates else None
        if pick is None:
            upg._cancel(detector, img, tap_fn)
            return self._log(UpgradeResult('nothing_upgradable',
                                           resources=resources))
        if self.verbose:
            from clashai.config.logging import pp
            pp(f" Labo : {len(candidates)} améliorables -> {pick}", tag='ok')

        # 5. taper la carte → écran de confirmation PARTAGÉ avec les bâtiments
        tap_fn(pick.x, pick.y)
        time.sleep(_D_TAP)
        return self._log(upg.confirm_step(
            screenshot_fn, tap_fn, resources=resources,
            confirm_decider=confirm_decider))

    @staticmethod
    def _cheapest(candidates):
        """Politique par défaut : le moins cher dont le prix est lisible."""
        priced = [c for c in candidates if c.price]
        return min(priced, key=lambda c: c.price) if priced else None

    def _log(self, result):
        if self.verbose:
            from clashai.config.logging import pp
            pp(f" Labo: {result.status}"
               + (f" (prix {result.price})" if result.price else ""), tag='ok')
        return result
