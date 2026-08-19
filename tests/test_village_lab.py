"""VillageLab : recherches au laboratoire (V5.2, increment 3).

Detecteur UI, lecteur de widgets, CNN barre de troupes et I/O sont injectes :
aucun ADB, aucun poids, aucun GPU. On couvre le gating (labo occupe, labo
introuvable, menu absent), la selection des cartes ameliorables (couleur vs
gris) et le fait que la confirmation est bien DELEGUEE a l'etape partagee avec
l'upgrade de batiment (donc le garde-fou anti-gemmes s'applique aussi ici).
"""

import pytest

from clashai.village.lab import LabCandidate, VillageLab
from clashai.village.upgrader import UpgradeResult


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)


class _FakeDetector:
    def __init__(self, buttons=None):
        self._b = buttons or {}

    def detect(self, img):
        return dict(self._b)

    def detect_raw(self, img):
        return {}


class _FakeReader:
    def __init__(self, labs=None, resources=None):
        self._labs = labs
        self._r = resources or {}

    def read_labs(self, img):
        return self._labs

    def read_resources(self, img):
        return dict(self._r)


class _FakeBar:
    """CNN barre de troupes : cartes du labo (grises = non ameliorables)."""

    def __init__(self, cards):
        self._cards = cards

    def detect(self, img):
        out = []
        for name, cx, grayed in self._cards:
            out.append({
                'name': name, 'bbox': (cx - 40, 500, cx + 40, 620),
                'center': (cx, 560), 'conf': 0.9,
                'is_grayed': grayed, 'no_tap': False,
            })
        return out


class _FakeUpgrader:
    """Capture l'appel a confirm_step (l'etape PARTAGEE avec les batiments)."""

    def __init__(self, result=None):
        self.confirm_calls = 0
        self.cancelled = 0
        self._result = result or UpgradeResult('ok', price=500000)

    def confirm_step(self, screenshot_fn, tap_fn, **kw):
        self.confirm_calls += 1
        self.kw = kw
        return self._result

    def _dump(self, img, step):
        pass

    def _find(self, detector, img, name):
        hit = detector.detect(img).get(name)
        return hit if hit and hit[2] >= 0.60 else None

    def _cancel(self, detector, img, tap_fn):
        self.cancelled += 1


class _Img:
    size = (1920, 1080)


def _lab(cards=None, labs=(1, 1), buttons=None, upgrader=None, resources=None):
    return VillageLab(
        detector=_FakeDetector(buttons if buttons is not None
                               else {'rechercher': (800, 700, 0.9)}),
        reader=_FakeReader(labs=labs, resources=resources or {'elixir': 9_000_000}),
        upgrader=upgrader or _FakeUpgrader(),
        verbose=False,
    ), {'troop_bar_detector': _FakeBar(cards or [])}


def _run(lab, models, taps, **kw):
    return lab.research(lambda: _Img(), lambda x, y: taps.append((x, y)),
                        models, **kw)


# ---------------------------------------------------------------------------
# Capteur : place_labo "N/M", N = places disponibles
# ---------------------------------------------------------------------------

def test_is_free_true_when_a_slot_is_available():
    lab, _ = _lab(labs=(1, 1))
    assert lab.is_free(_Img()) is True


def test_is_free_false_when_research_running():
    lab, _ = _lab(labs=(0, 1))
    assert lab.is_free(_Img()) is False


def test_is_free_none_when_unreadable():
    lab, _ = _lab(labs=None)
    assert lab.is_free(_Img()) is None


# ---------------------------------------------------------------------------
# Selection des cartes : couleur = ameliorable, gris = labo trop bas
# ---------------------------------------------------------------------------

def test_list_candidates_keeps_only_coloured_cards(monkeypatch):
    cards = [('barbare', 300, False), ('pekka', 600, True), ('dragon', 900, False)]
    lab, models = _lab(cards)
    monkeypatch.setattr(VillageLab, '_read_card_price', lambda self, i, b: 1000)
    names = [c.name for c in lab.list_candidates(_Img(), models)]
    assert names == ['barbare', 'dragon']       # pekka grise -> ecartee


def test_list_candidates_empty_without_troop_bar_model():
    lab, _ = _lab([('barbare', 300, False)])
    assert lab.list_candidates(_Img(), {'troop_bar_detector': None}) == []


def test_cheapest_policy_picks_lowest_readable_price():
    cands = [LabCandidate('dragon', 1, 1, 3_700_000),
             LabCandidate('gobelin', 2, 2, 500_000),
             LabCandidate('inconnu', 3, 3, None)]      # prix illisible -> ignore
    assert VillageLab._cheapest(cands).name == 'gobelin'


def test_cheapest_returns_none_when_no_price_readable():
    assert VillageLab._cheapest([LabCandidate('x', 1, 1, None)]) is None


# ---------------------------------------------------------------------------
# Flux : gating puis delegation de la confirmation
# ---------------------------------------------------------------------------

def test_busy_lab_stops_before_touching_anything(monkeypatch):
    taps = []
    lab, models = _lab([('barbare', 300, False)], labs=(0, 1))
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: (500, 500)))
    r = _run(lab, models, taps)
    assert r.status == 'busy'
    assert taps == []


def test_lab_not_found_reports_and_does_nothing(monkeypatch):
    taps = []
    lab, models = _lab([('barbare', 300, False)])
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: None))
    assert _run(lab, models, taps).status == 'lab_not_found'
    assert taps == []


def test_menu_not_found_when_search_button_absent(monkeypatch):
    taps = []
    lab, models = _lab([('barbare', 300, False)], buttons={})
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: (500, 500)))
    r = _run(lab, models, taps)
    assert r.status == 'menu_not_found'
    assert taps[0] == (500, 500)                 # labo tape, puis refermeture


def test_nothing_upgradable_when_all_cards_greyed(monkeypatch):
    taps = []
    upg = _FakeUpgrader()
    lab, models = _lab([('pekka', 600, True)], upgrader=upg)
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: (500, 500)))
    r = _run(lab, models, taps)
    assert r.status == 'nothing_upgradable'
    assert upg.confirm_calls == 0 and upg.cancelled == 1


def test_research_taps_the_pick_then_delegates_confirmation(monkeypatch):
    taps = []
    upg = _FakeUpgrader(UpgradeResult('ok', price=500_000))
    cards = [('dragon', 900, False), ('gobelin', 300, False)]
    lab, models = _lab(cards, upgrader=upg)
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: (500, 500)))
    prices = {'dragon': 3_700_000, 'gobelin': 500_000}
    monkeypatch.setattr(
        VillageLab, '_read_card_price',
        lambda self, img, bbox: prices['gobelin' if bbox[0] < 500 else 'dragon'])

    r = _run(lab, models, taps)

    assert r.status == 'ok'
    assert upg.confirm_calls == 1, "la confirmation DOIT passer par l'etape partagee"
    # le gobelin (moins cher) est tape, pas le dragon
    assert (300, 560) in taps and (900, 560) not in taps


def test_explicit_choice_overrides_the_default_policy(monkeypatch):
    taps = []
    upg = _FakeUpgrader()
    lab, models = _lab([('dragon', 900, False), ('gobelin', 300, False)],
                       upgrader=upg)
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: (500, 500)))
    monkeypatch.setattr(VillageLab, '_read_card_price', lambda self, i, b: 1000)

    _run(lab, models, taps,
         choose=lambda cands: next(c for c in cands if c.name == 'dragon'))

    assert (900, 560) in taps and (300, 560) not in taps


def test_resources_are_read_before_opening_the_popup(monkeypatch):
    """Le pop-up assombrit le village : le solde doit venir de l'ecran clair."""
    taps = []
    upg = _FakeUpgrader()
    lab, models = _lab([('gobelin', 300, False)], upgrader=upg,
                       resources={'elixir': 4_242_000})
    monkeypatch.setattr(VillageLab, 'find_lab', staticmethod(lambda i, m: (500, 500)))
    monkeypatch.setattr(VillageLab, '_read_card_price', lambda self, i, b: 1000)

    _run(lab, models, taps)

    assert upg.kw['resources'] == {'elixir': 4_242_000}


# ---------------------------------------------------------------------------
# Confirmation visuelle : ne doit jamais casser le flux
# ---------------------------------------------------------------------------

def test_annotate_scan_is_a_noop_without_the_troop_bar_model(tmp_path):
    lab, _ = _lab()
    out = lab.annotate_scan(_Img(), {'troop_bar_detector': None},
                            str(tmp_path / 'x.png'))
    assert out is None


def test_annotate_scan_is_a_noop_without_screenshot(tmp_path):
    lab, models = _lab([('barbare', 300, False)])
    assert lab.annotate_scan(None, models, str(tmp_path / 'x.png')) is None
