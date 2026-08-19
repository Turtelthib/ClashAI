"""WidgetReader : « CNN localise -> digit CNN lit ».

Le digit CNN (read_widget_number / read_widget_ratio) est monkeypatche : on teste
la LOGIQUE (localisation par classe, scaling ADB->image, lecture "N/M"), pas la
lecture de pixels. Aucun poids, aucun GPU.

Le garde-fou d'homogenite (`_glyphs_consistent`) est teste a part : c'est lui qui
garantit qu'une boite mal cadree rend None plutot qu'un montant faux.
"""

import numpy as np
from PIL import Image

from clashai.perception import digit_reader, widget_reader
from clashai.perception.ui_detector import Detection


class _FakeDetector:
    def __init__(self, raw):
        self._raw = raw

    def detect_raw(self, img):
        return self._raw


def _img(w=1920, h=1080):
    return Image.new('RGB', (w, h))


def _det(cls, x, y, w=120, h=40):
    return Detection(cls, 0, x, y, w, h, 0.9)


# ---------------------------------------------------------------------------
# Ressources
# ---------------------------------------------------------------------------

def test_read_resources_reads_each_present_counter(monkeypatch):
    raw = {
        'compteur_or': [_det('compteur_or', 1700, 60)],
        'compteur_elixir': [_det('compteur_elixir', 1700, 180)],
    }
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    vals = iter([(10620483, 0.9), (5573747, 0.9)])
    monkeypatch.setattr(digit_reader, 'read_widget_number', lambda *a, **k: next(vals))

    out = reader.read_resources(_img())
    assert out == {'or': 10620483, 'elixir': 5573747}   # elixir_noire absent -> ignoré


def test_read_resources_empty_when_nothing_detected():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_resources(_img()) == {}


def test_read_resources_skips_unreadable_counter(monkeypatch):
    """Lecture refusée (garde-fou d'homogénéité) -> la ressource est omise,
    jamais devinée."""
    raw = {'compteur_or': [_det('compteur_or', 1700, 60)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_widget_number', lambda *a, **k: (None, 0.0))
    assert reader.read_resources(_img()) == {}


# ---------------------------------------------------------------------------
# Ouvriers ("N/M" -> (libres, total))
# ---------------------------------------------------------------------------

def test_read_builders_splits_free_over_total(monkeypatch):
    raw = {'nombre_ouvrier': [_det('nombre_ouvrier', 300, 60, w=80)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_widget_ratio', lambda *a, **k: ((1, 6), 0.9))
    assert reader.read_builders(_img()) == (1, 6)


def test_read_builders_none_when_widget_absent():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_builders(_img()) is None


def test_read_labs_splits_ratio(monkeypatch):
    raw = {'place_labo': [_det('place_labo', 820, 60, w=70)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_widget_ratio', lambda *a, **k: ((0, 1), 0.9))
    assert reader.read_labs(_img()) == (0, 1)   # "0/1" = labo libre


def test_read_labs_none_when_absent():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_labs(_img()) is None


def test_read_builders_none_when_unreadable(monkeypatch):
    raw = {'nombre_ouvrier': [_det('nombre_ouvrier', 300, 60, w=80)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_widget_ratio', lambda *a, **k: (None, 0.0))
    assert reader.read_builders(_img()) is None


# ---------------------------------------------------------------------------
# Segmentation par composantes connexes : isoler le nombre DANS une boite large
#
# La boite du CNN reste large (l'icone distingue nombre_ouvrier de place_labo),
# donc c'est le lecteur qui doit ecarter contour de cadre, epee, reflets.
# Valeurs des composantes mesurees sur village_principal.png.
# ---------------------------------------------------------------------------

def _blob(x, y, w, h, fill=0.7):
    """Image binaire d'une composante de taux de remplissage donne."""
    img = np.zeros((y + h + 5, x + w + 5), dtype=np.uint8)
    img[y:y + int(h * fill), x:x + w] = 255      # bloc plein -> fill ~= `fill`
    return img


def _spans_of(mask):
    """Passe un masque binaire dans le filtre de composantes du digit_reader."""
    from PIL import Image as _I
    rgb = np.dstack([mask] * 3)
    return digit_reader._widget_spans(_I.fromarray(rgb))[0]


def test_components_keep_digits_and_drop_frame_outline():
    """Un contour de cadre (grande boite quasi vide, fill 0.09) est ecarte ;
    les chiffres compacts alignes sont gardes."""
    mask = np.zeros((120, 240), dtype=np.uint8)
    # contour du cadre : rectangle creux 130x60 -> fill tres bas
    mask[39:99, 101:231] = 255
    mask[43:95, 105:227] = 0
    # trois glyphes compacts alignes (le "5/5" reel : x=124/147/170, y=50, h=28)
    for x in (124, 147, 170):
        mask[50:78, x:x + 20] = 255
    spans = _spans_of(mask)
    assert len(spans) == 3, f"attendu les 3 glyphes, obtenu {spans}"
    assert [s[0] for s in spans] == [124, 147, 170]     # ordre gauche->droite


def test_components_drop_flat_icon_and_thin_stroke():
    """L'epee (ratio h/w 0.64) et un trait fin (ratio 12.5) sont ecartes."""
    mask = np.zeros((120, 240), dtype=np.uint8)
    mask[62:103, 56:120] = 255        # epee : 64x41 -> ratio 0.64
    mask[15:40, 47:49] = 255          # trait : 2x25 -> ratio 12.5
    for x in (140, 177):              # deux chiffres
        mask[51:77, x:x + 11] = 255
    spans = _spans_of(mask)
    assert [s[0] for s in spans] == [140, 177]


# ---------------------------------------------------------------------------
# Garde-fou : plutot NE RIEN lire qu'un chiffre faux
# ---------------------------------------------------------------------------

def _glyph(h, w=12):
    return np.zeros((h, w), dtype=np.uint8)


def test_guard_accepts_homogeneous_glyphs():
    """Boite serree -> tous les glyphes a la meme hauteur (cas reel: 7x h=24)."""
    assert digit_reader._glyphs_consistent([_glyph(24) for _ in range(7)])


def test_guard_accepts_small_height_jitter():
    """L'antialiasing fait varier de 1-2 px : ca doit rester accepte."""
    assert digit_reader._glyphs_consistent([_glyph(24), _glyph(25), _glyph(26)])


def test_guard_rejects_icon_contaminated_glyphs():
    """Boite large : icone/contour captes -> hauteurs disparates (cas reel:
    place_labo lisait '222' au lieu de '1/1'). Doit etre refuse."""
    assert not digit_reader._glyphs_consistent(
        [_glyph(h) for h in (49, 56, 64, 70, 80, 86, 92)])


def test_guard_rejects_empty():
    assert not digit_reader._glyphs_consistent([])


# ---------------------------------------------------------------------------
# Ressource d'un prix, par COULEUR de l'icone (pas de classe CNN dediee)
# ---------------------------------------------------------------------------

def _patch(rgb, size=(30, 30), bg=None):
    """Carre de couleur `rgb`, optionnellement sur un fond `bg` (moitie droite)."""
    im = Image.new('RGB', size, bg or rgb)
    if bg is not None:
        im.paste(Image.new('RGB', (size[0] // 2, size[1]), rgb), (0, 0))
    return im


def test_classify_resource_color_recognises_each_resource():
    """Couleurs REELLES mesurees sur une capture de jeu (pas des valeurs devinees)."""
    clf = widget_reader.WidgetReader.classify_resource_color
    assert clf(_patch((255, 237, 84))) == 'or'            # piece d'or
    assert clf(_patch((128, 31, 128))) == 'elixir'        # goutte d'elixir
    assert clf(_patch((43, 34, 46))) == 'elixir_noire'    # goutte d'elixir noir


def test_classify_resource_color_ignores_green_button_background():
    """L'icone rose sur le bouton vert de confirmation -> 'elixir'."""
    clf = widget_reader.WidgetReader.classify_resource_color
    assert clf(_patch((128, 31, 128), bg=(218, 248, 136))) == 'elixir'


def test_classify_resource_color_rejects_ui_decoys():
    """Aucune ressource -> None (decision deferee, jamais devinee).

    REGRESSION : le panneau gris fonce (63,58,56) etait pris pour de l'elixir
    noir en RGB (distance 54 de (43,34,46)) -> faux 'cant_afford' alors que
    l'upgrade etait payable en or. La classification HSV l'ecarte.
    """
    clf = widget_reader.WidgetReader.classify_resource_color
    assert clf(_patch((63, 58, 56))) is None       # panneau gris  <-- le bug
    assert clf(_patch((218, 248, 136))) is None    # bouton vert
    assert clf(_patch((112, 109, 70))) is None     # ombre jaunatre
    assert clf(_patch((250, 250, 250))) is None    # blanc
    assert clf(_patch((20, 20, 20))) is None       # noir


def test_read_price_resource_none_when_price_widget_absent():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_price_resource(_img()) is None


# ---------------------------------------------------------------------------
# Garde-fou : un compteur du HUD pris pour le prix
#
# `prix_upgrade` et `compteur_*` sont tous « des chiffres blancs », et les deux
# sont visibles en meme temps sur l'ecran de confirmation. Confondre les deux
# ferait lire le SOLDE comme prix -> `solde >= prix` trivialement vrai -> achat
# confirme a tort. Le prix qui recouvre un compteur doit donc etre rejete.
# ---------------------------------------------------------------------------

def test_price_overlapping_a_counter_is_rejected(monkeypatch):
    """Faux positif : `prix_upgrade` detecte pile sur le compteur d'or."""
    raw = {
        'prix_upgrade': [_det('prix_upgrade', 1700, 60, w=200, h=40)],
        'compteur_or': [_det('compteur_or', 1700, 60, w=200, h=40)],   # meme boite
    }
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_widget_number',
                        lambda *a, **k: (2742878, 0.9))
    assert reader.read_price_number(_img()) is None      # refuse -> decision deferee
    assert reader.read_price_resource(_img()) is None


def test_genuine_price_away_from_counters_is_kept(monkeypatch):
    """Vrai prix (centre du popup) : les compteurs du HUD ne le genent pas."""
    raw = {
        'prix_upgrade': [_det('prix_upgrade', 960, 950, w=200, h=40)],
        'compteur_or': [_det('compteur_or', 1700, 60, w=200, h=40)],
    }
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_widget_number',
                        lambda *a, **k: (1380000, 0.9))
    assert reader.read_price_number(_img()) == 1380000


def test_price_falls_back_to_a_non_overlapping_candidate(monkeypatch):
    """Si la meilleure detection recouvre un compteur, on prend la suivante."""
    raw = {
        'prix_upgrade': [
            _det('prix_upgrade', 1700, 60, w=200, h=40),    # sur le compteur
            _det('prix_upgrade', 960, 950, w=200, h=40),    # le vrai prix
        ],
        'compteur_or': [_det('compteur_or', 1700, 60, w=200, h=40)],
    }
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    det = reader._price_det(_img())
    assert det is not None and (det.x, det.y) == (960, 950)


# ---------------------------------------------------------------------------
# Scaling ADB -> pixels image (le détecteur rend des coords ADB 1920x1080)
# ---------------------------------------------------------------------------

def test_widget_crop_scales_adb_to_image_pixels():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    img = _img(960, 540)                          # moitié de l'ADB
    det = _det('x', 1000, 500, w=100, h=40)       # centre ADB (1000,500)
    crop = reader._widget_crop(img, det)
    assert crop is not None
    # sx=0.5 -> centre px (500,250), w=50,h=20 (+2*4 de pad) => ~58x28
    assert 54 <= crop.size[0] <= 62
    assert 24 <= crop.size[1] <= 32
