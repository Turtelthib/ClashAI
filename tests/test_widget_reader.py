"""WidgetReader : « CNN localise -> digit CNN lit ».

Le digit CNN (digit_reader.read_number) est monkeypatche : on teste la LOGIQUE
(localisation par classe, scaling ADB->image, split "N/M" des ouvriers), pas la
lecture de pixels. Aucun poids, aucun GPU.
"""

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
    monkeypatch.setattr(digit_reader, 'read_number', lambda *a, **k: next(vals))

    out = reader.read_resources(_img())
    assert out == {'or': 10620483, 'elixir': 5573747}   # elixir_noire absent -> ignoré


def test_read_resources_empty_when_nothing_detected():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_resources(_img()) == {}


def test_read_resources_skips_unreadable_counter(monkeypatch):
    raw = {'compteur_or': [_det('compteur_or', 1700, 60)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_number', lambda *a, **k: (None, 0.0))
    assert reader.read_resources(_img()) == {}


# ---------------------------------------------------------------------------
# Ouvriers ("N/M" -> (libres, total))
# ---------------------------------------------------------------------------

def test_read_builders_splits_free_over_total(monkeypatch):
    raw = {'nombre_ouvrier': [_det('nombre_ouvrier', 300, 60, w=80)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    halves = iter([(1, 0.9), (6, 0.9)])          # moitié gauche puis droite
    monkeypatch.setattr(digit_reader, 'read_number', lambda *a, **k: next(halves))
    assert reader.read_builders(_img()) == (1, 6)


def test_read_builders_none_when_widget_absent():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_builders(_img()) is None


def test_read_labs_splits_ratio(monkeypatch):
    raw = {'place_labo': [_det('place_labo', 820, 60, w=70)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    halves = iter([(0, 0.9), (1, 0.9)])          # "0/1" = labo libre
    monkeypatch.setattr(digit_reader, 'read_number', lambda *a, **k: next(halves))
    assert reader.read_labs(_img()) == (0, 1)


def test_read_labs_none_when_absent():
    reader = widget_reader.WidgetReader(detector=_FakeDetector({}))
    assert reader.read_labs(_img()) is None


def test_read_builders_none_when_unreadable(monkeypatch):
    raw = {'nombre_ouvrier': [_det('nombre_ouvrier', 300, 60, w=80)]}
    reader = widget_reader.WidgetReader(detector=_FakeDetector(raw))
    monkeypatch.setattr(digit_reader, 'read_number', lambda *a, **k: (None, 0.0))
    assert reader.read_builders(_img()) is None


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
