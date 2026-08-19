# clashai/perception/widget_reader.py
# Lecture de nombres dans les widgets d'interface (V5.2).
#
# Pattern unifié : le CNN UI LOCALISE le widget, le digit CNN LIT le nombre dedans.
#   - ressources  : compteur_or / compteur_elixir / compteur_elixir_noire
#   - ouvriers    : nombre_ouvrier  ("1/6" = libres/total)
#   - prix upgrade : prix_upgrade (écran de confirmation)
#
# On réutilise digit_reader — offline, rapide, zéro dépendance externe — mais par
# son chemin WIDGET (composantes connexes), PAS celui des badges de troupes :
# voir le bloc « Widgets d'UI : chiffres faux » de TROUBLESHOOTING.md.
#
# Validé en conditions réelles (18 août 2026) : or, élixir, élixir noir, ouvriers
# et labo lus correctement. Un widget non détecté ou une lecture refusée par le
# garde-fou rend None / {} — jamais un chiffre deviné.

from clashai.config import ADB_HEIGHT, ADB_WIDTH
from clashai.perception import digit_reader

# clé logique -> classe CNN UI du compteur
RESOURCE_CLASSES = {
    'or': 'compteur_or',
    'elixir': 'compteur_elixir',
    'elixir_noire': 'compteur_elixir_noire',
}
BUILDERS_CLASS = 'nombre_ouvrier'   # "N/M" = ouvriers libres / total
LAB_CLASS = 'place_labo'            # "0/1" = labo libre, "1/1" = labo occupé
PRICE_CLASS = 'prix_upgrade'        # le chiffre du prix (écran de confirmation)

# Marge autour du widget avant lecture (px image).
_CROP_PAD = 4

# ── Quelle ressource paie un prix ? (par COULEUR, pas par classe CNN) ────────
# L'icône (goutte/pièce) à droite du prix apparaît partout dans le jeu : en faire
# une classe CNN obligerait à la labéliser sur TOUS les écrans (même visuel = même
# classe) pour un gain nul.
#
# ⚠️ En RGB ça ne marche pas : le gris sombre de l'UI (63,58,56) tombe à distance
# 54 de l'élixir noir (43,34,46) → un panneau gris était pris pour de l'élixir
# noir, d'où un faux « pas les moyens ». On classe donc en **HSV**, où les trois
# ressources occupent des zones franches et où gris/blanc/vert sont exclus par
# construction (saturation trop basse, ou teinte hors plage).
#
# Mesuré sur capture réelle (OpenCV : H 0-179, S 0-255, V 0-255) :
#   pièce d'or      H 27  S 171  V 255      goutte élixir   H 150  S 193  V 128
#   élixir noir     H 143 S  60  V  46      bouton vert     H  39  S  47  V 248
#   ombre jaunâtre  H 10  S 161  V 140      panneau gris    H   9  S  28  V  63
_RESOURCE_HSV = {
    #            H_min H_max  S_min  V_min  V_max
    'or':          (15,  33,   110,   110,   255),
    'elixir':      (130, 168,  110,    96,   255),
    'elixir_noire':(120, 170,   35,     0,    95),
}
# Minimum de pixels matchés pour conclure (sinon None → décision déférée).
_MIN_COLOR_PIXELS = 20


class WidgetReader:
    """Lit les nombres des widgets d'UI via CNN UI (localise) + digit CNN (lit)."""

    def __init__(self, detector=None):
        self._detector = detector

    def _get_detector(self):
        """Détecteur branché en priorité, sinon instance dédiée (lazy)."""
        if self._detector is None:
            from clashai.perception.ui_buttons import get_detector
            self._detector = get_detector()
            if self._detector is None:
                from clashai.perception.ui_detector import UIDetector
                self._detector = UIDetector(verbose=False)
        return self._detector

    def _widget_crop(self, screenshot_pil, det, pad=_CROP_PAD):
        """Crop image du widget. det.x/y/w/h sont en ADB → reconvertis en px image."""
        img_w, img_h = screenshot_pil.size
        sx, sy = img_w / ADB_WIDTH, img_h / ADB_HEIGHT
        cx, cy = det.x * sx, det.y * sy
        w, h = det.w * sx, det.h * sy
        x1 = max(0, int(cx - w / 2) - pad)
        y1 = max(0, int(cy - h / 2) - pad)
        x2 = min(img_w, int(cx + w / 2) + pad)
        y2 = min(img_h, int(cy + h / 2) + pad)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        return screenshot_pil.crop((x1, y1, x2, y2))

    # ---- nombre d'un widget nommé (localisé par le CNN UI) -----------------

    def read_widget_number(self, screenshot_pil, class_name):
        """Localise la classe CNN `class_name` puis lit l'entier dedans. int|None."""
        dets = self._get_detector().detect_raw(screenshot_pil).get(class_name)
        if not dets:
            return None
        crop = self._widget_crop(screenshot_pil, dets[0])
        if crop is None:
            return None
        n, _ = digit_reader.read_widget_number(crop)
        return n

    # ---- ressources --------------------------------------------------------

    def read_resources(self, screenshot_pil) -> dict:
        """{'or': N, 'elixir': N, 'elixir_noire': N} — seulement les lus."""
        out = {}
        for key, cls in RESOURCE_CLASSES.items():
            n = self.read_widget_number(screenshot_pil, cls)
            if n is not None:
                out[key] = n
        return out

    # ---- widgets "N/M" (ouvriers, labo) ------------------------------------

    def _read_ratio(self, screenshot_pil, class_name):
        """Lit un widget "N/M" → (N, M), ou None si absent/illisible.

        Délègue à `digit_reader.read_widget_ratio` : glyphes cohérents (les
        icônes blanches du crop — épée, visage d'ouvrier — sont écartées), puis
        premier et dernier glyphe (le '/' n'est pas une classe du CNN).
        """
        dets = self._get_detector().detect_raw(screenshot_pil).get(class_name)
        if not dets:
            return None
        crop = self._widget_crop(screenshot_pil, dets[0])
        if crop is None:
            return None
        ratio, _ = digit_reader.read_widget_ratio(crop)
        return ratio

    def read_builders(self, screenshot_pil):
        """(libres, total) depuis `nombre_ouvrier` ("1/6"), ou None si illisible."""
        return self._read_ratio(screenshot_pil, BUILDERS_CLASS)

    def read_labs(self, screenshot_pil):
        """(N, M) depuis `place_labo`, où **N = places DISPONIBLES**.

        Même convention que les ouvriers ("1/6" = 1 libre sur 6) : "1/1" = labo
        libre, "0/1" = recherche en cours. `VillageLab.is_free()` s'appuie
        dessus ; ici on rend le ratio brut.
        """
        return self._read_ratio(screenshot_pil, LAB_CLASS)

    # ---- quelle ressource paie le prix ? (couleur de l'icône) --------------

    @staticmethod
    def classify_resource_color(crop_pil):
        """'or' | 'elixir' | 'elixir_noire' | None depuis un crop d'icône.

        Compte les pixels tombant dans la zone HSV de chaque ressource ; la plus
        représentée gagne. Le fond (bouton vert, panneau gris, blanc, ombres) ne
        tombe dans aucune zone → None, et la décision d'achat est déférée.
        """
        import cv2
        import numpy as np

        rgb = np.asarray(crop_pil.convert('RGB'), dtype=np.uint8)
        if rgb.size == 0:
            return None
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        h, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        best, best_count = None, 0
        for name, (h0, h1, s0, v0, v1) in _RESOURCE_HSV.items():
            m = (h >= h0) & (h <= h1) & (sat >= s0) & (val >= v0) & (val <= v1)
            count = int(m.sum())
            if count > best_count:
                best, best_count = name, count
        return best if best_count >= _MIN_COLOR_PIXELS else None

    # ---- prix d'upgrade (avec garde-fou anti-confusion) --------------------

    @staticmethod
    def _overlaps(a, b):
        """True si deux détections se recouvrent (boîtes centrées x/y, w/h)."""
        ax1, ax2 = a.x - a.w / 2, a.x + a.w / 2
        ay1, ay2 = a.y - a.h / 2, a.y + a.h / 2
        bx1, bx2 = b.x - b.w / 2, b.x + b.w / 2
        by1, by2 = b.y - b.h / 2, b.y + b.h / 2
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return False
        smaller = min(max(1.0, a.w * a.h), max(1.0, b.w * b.h))
        return inter / smaller > 0.5      # l'un est majoritairement dans l'autre

    def _price_det(self, screenshot_pil):
        """Détection `prix_upgrade` VALIDÉE, ou None.

        ⚠️ `prix_upgrade` et `compteur_*` sont tous « des chiffres blancs ». Si le
        CNN prenait un compteur du HUD pour le prix, on lirait le SOLDE comme
        prix → `solde >= prix` trivialement vrai → achat confirmé à tort. On
        rejette donc toute détection de prix qui recouvre un compteur de
        ressource (les deux sont visibles en même temps sur l'écran de
        confirmation). Sans preuve, pas de décision : l'upgrader annule.
        """
        raw = self._get_detector().detect_raw(screenshot_pil)
        prices = raw.get(PRICE_CLASS) or []
        counters = [d for cls in RESOURCE_CLASSES.values() for d in raw.get(cls, [])]
        for p in prices:                       # déjà triées par confiance ↓
            if not any(self._overlaps(p, c) for c in counters):
                return p
        return None

    def read_price_number(self, screenshot_pil):
        """Montant du prix d'upgrade (int) ou None — garde-fou compris."""
        det = self._price_det(screenshot_pil)
        if det is None:
            return None
        crop = self._widget_crop(screenshot_pil, det)
        if crop is None:
            return None
        n, _ = digit_reader.read_widget_number(crop)
        return n

    def price_is_red(self, screenshot_pil):
        """True si le prix est écrit en ROUGE, None si indéterminable.

        Dans CoC, un prix rouge = **le jeu lui-même signale un solde
        insuffisant**. C'est un signal AUTORITATIF, indépendant de toute lecture
        de chiffre : il reste juste même si la segmentation perd un digit. On
        s'en sert pour refuser l'achat sans dépendre du montant lu — taper
        `confirmer` dans ce cas ouvrirait le pop-up « acheter des gemmes ».
        """
        import cv2
        import numpy as np

        from clashai.perception import digit_reader as dr

        det = self._price_det(screenshot_pil)
        if det is None:
            return None
        crop = self._widget_crop(screenshot_pil, det)
        if crop is None:
            return None
        bgr = cv2.cvtColor(np.asarray(crop.convert('RGB')), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red = ((val > dr.RED_V_MIN) & (sat > dr.RED_S_MIN)
               & ((h >= dr.RED_H_HIGH) | (h <= dr.RED_H_LOW)))
        white = (val > dr.V_MIN) & (sat < dr.S_MAX)
        n_red, n_white = int(red.sum()), int(white.sum())
        if n_red + n_white < _MIN_COLOR_PIXELS:
            return None
        return n_red > n_white

    def read_price_resource(self, screenshot_pil):
        """Ressource dans laquelle est libellé `prix_upgrade`, ou None.

        L'icône est collée à DROITE du nombre → on échantillonne cette bande.
        Repli sur la box du prix elle-même si le label la contient déjà.
        """
        det = self._price_det(screenshot_pil)
        if det is None:
            return None

        img_w, img_h = screenshot_pil.size
        sx, sy = img_w / ADB_WIDTH, img_h / ADB_HEIGHT
        cx, cy, w, h = det.x * sx, det.y * sy, det.w * sx, det.h * sy
        y1 = max(0, int(cy - h / 2))
        y2 = min(img_h, int(cy + h / 2))

        # 1. bande à droite du nombre. Largeur 2.5x la hauteur du texte : l'icône
        # est décalée d'un petit espace et la couvrir ENTIÈREMENT compte (une
        # bande trop courte n'attrapait qu'un liseré de la pièce, et c'est le
        # gris du fond qui l'emportait).
        rx1 = min(img_w, int(cx + w / 2))
        rx2 = min(img_w, rx1 + max(12, int(h * 2.5)))
        if rx2 - rx1 >= 4 and y2 - y1 >= 4:
            hit = self.classify_resource_color(
                screenshot_pil.crop((rx1, y1, rx2, y2)))
            if hit is not None:
                return hit

        # 2. repli : la box du prix (si l'icône y est incluse)
        crop = self._widget_crop(screenshot_pil, det)
        return self.classify_resource_color(crop) if crop is not None else None

    def _first(self, screenshot_pil, class_name):
        """Meilleure détection d'une classe CNN, ou None."""
        dets = self._get_detector().detect_raw(screenshot_pil).get(class_name)
        return dets[0] if dets else None
