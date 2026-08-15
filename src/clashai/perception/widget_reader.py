# clashai/perception/widget_reader.py
# Lecture de nombres dans les widgets d'interface (V5.2).
#
# Pattern unifié : le CNN UI LOCALISE le widget, le digit CNN LIT le nombre dedans.
#   - ressources  : compteur_or / compteur_elixir / compteur_elixir_noire
#   - ouvriers    : nombre_ouvrier  ("1/6" = libres/total)
#   - prix upgrade : lu dans un crop fourni (écran de confirmation)
#
# On réutilise digit_reader (segment_glyphs + DigitCNN) — offline, rapide, zéro
# dépendance externe. Ces classes CNN UI (compteur_*, nombre_ouvrier) arrivent
# avec le prochain re-train du modèle ; d'ici là read_* rend simplement None /
# {} (widget non détecté) sans lever.
#
# ⚠️ À VALIDER sur de vrais crops une fois les classes ajoutées : la police des
# ressources diffère des badges de troupes → si la segmentation cale, on ajoute
# quelques samples et on re-train le digit CNN (même pipeline).

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

# Marge autour du widget avant lecture (px image).
_CROP_PAD = 4


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
        n, _ = digit_reader.read_number(crop, drop_leading_x=False)
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

        On coupe le crop en deux au milieu (le '/' est centré) et on lit chaque
        moitié — évite que le digit CNN bute sur le '/'.
        """
        dets = self._get_detector().detect_raw(screenshot_pil).get(class_name)
        if not dets:
            return None
        crop = self._widget_crop(screenshot_pil, dets[0])
        if crop is None:
            return None
        w, h = crop.size
        a, _ = digit_reader.read_number(crop.crop((0, 0, w // 2, h)))
        b, _ = digit_reader.read_number(crop.crop((w // 2, 0, w, h)))
        if a is None or b is None:
            return None
        return (a, b)

    def read_builders(self, screenshot_pil):
        """(libres, total) depuis `nombre_ouvrier` ("1/6"), ou None si illisible."""
        return self._read_ratio(screenshot_pil, BUILDERS_CLASS)

    def read_labs(self, screenshot_pil):
        """(N, M) depuis `place_labo` ("0/1" = labo libre, "1/1" = occupé).

        Sémantique exacte (quel nombre = libre) à figer en incr. 3 (agent labo) ;
        ici on rend le ratio brut.
        """
        return self._read_ratio(screenshot_pil, LAB_CLASS)

    # ---- nombre générique (ex. prix d'upgrade) -----------------------------

    def read_number_in_box(self, screenshot_pil, bbox):
        """Lit un entier dans une bbox image (x1,y1,x2,y2). (int|None)."""
        crop = screenshot_pil.crop(bbox)
        n, _ = digit_reader.read_number(crop, drop_leading_x=False)
        return n
