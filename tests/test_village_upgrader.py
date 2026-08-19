"""VillageUpgrader : le flux d'upgrade + gating proactif + sécurité anti-gemmes.

Détecteur (boutons CNN) et lecteur (ouvriers/ressources/prix) sont injectés :
aucun ADB, aucun poids. On couvre chaque statut de sortie + la garantie SÛRE
(jamais de tap `confirmer` sans décision prouvée).
"""

import pytest

from clashai.village.upgrader import VillageUpgrader


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Neutralise les time.sleep de l'exécuteur (tests instantanés)."""
    import time
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)


class _FakeDetector:
    """detect() -> {nom_bouton: (x, y, conf)}."""
    def __init__(self, buttons):
        self._b = buttons

    def detect(self, img):
        return dict(self._b)

    def detect_raw(self, img):
        return {}


class _FakeReader:
    def __init__(self, builders=None, resources=None, price=None,
                 price_resource=None, price_red=None):
        self._b = builders
        self._r = resources or {}
        self._p = price
        self._pr = price_resource
        self._red = price_red

    def read_builders(self, img):
        return self._b

    def read_resources(self, img):
        return dict(self._r)

    def read_price_number(self, img):
        return self._p

    def read_price_resource(self, img):
        return self._pr

    def price_is_red(self, img):
        return self._red


def _run(detector, reader, taps, **kw):
    up = VillageUpgrader(detector=detector, reader=reader, verbose=False)
    return up.upgrade_building(
        (900, 500), lambda: object(), lambda x, y: taps.append((x, y)), **kw
    )


# ---------------------------------------------------------------------------
# Gating proactif
# ---------------------------------------------------------------------------

def test_no_builder_stops_before_touching_the_village():
    taps = []
    r = _run(_FakeDetector({}), _FakeReader(builders=(0, 6)), taps)
    assert r.status == 'no_builder'
    assert r.builders == (0, 6)
    assert taps == []                    # rien tapé : on n'a pas ouvert le bâtiment


def test_not_upgradeable_when_no_ameliorer_button():
    taps = []
    # constructeur libre mais pas de bouton `ameliorer` au menu du bâtiment.
    r = _run(_FakeDetector({}), _FakeReader(builders=(1, 6)), taps)
    assert r.status == 'not_upgradeable'
    assert taps[0] == (900, 500)         # bâtiment tapé
    assert len(taps) == 2                # + tap neutre pour refermer le menu


# ---------------------------------------------------------------------------
# Écran de confirmation : décision d'affordabilité
# ---------------------------------------------------------------------------

def test_need_decision_when_price_unknown_cancels_safely():
    """Prix illisible (classe prix_upgrade pas encore là) + aucun décideur ->
    on ANNULE (jamais de confirmer à l'aveugle)."""
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9),
                         'annuler': (300, 950, 0.9)})
    r = _run(det, _FakeReader(builders=(1, 6), price=None), taps)
    assert r.status == 'need_decision'
    assert (700, 950) not in taps        # confirmer JAMAIS tapé
    assert (300, 950) in taps            # annuler tapé


def test_cant_afford_cancels():
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9),
                         'annuler': (300, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=5000, resources={'elixir': 100})
    r = _run(det, reader, taps, resource_type='elixir')
    assert r.status == 'cant_afford'
    assert (700, 950) not in taps
    assert (300, 950) in taps


def test_ok_when_affordable_via_resource_type():
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=5000, resources={'elixir': 999999})
    r = _run(det, reader, taps, resource_type='elixir')
    assert r.status == 'ok'
    assert r.price == 5000
    assert (700, 950) in taps            # confirmer tapé


def test_ok_when_confirm_decider_says_yes():
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=None)
    r = _run(det, reader, taps, confirm_decider=lambda price, res: True)
    assert r.status == 'ok'
    assert (700, 950) in taps


def test_price_resource_read_from_screen_makes_affordability_autonomous():
    """Sans resource_type impose, la ressource est lue a l'ecran (couleur de
    l'icone) -> l'upgrade se decide tout seul."""
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=1380000,
                         resources={'elixir': 7646344},
                         price_resource='elixir')
    r = _run(det, reader, taps)                 # aucun resource_type fourni
    assert r.status == 'ok'
    assert (700, 950) in taps


def test_price_resource_read_from_screen_can_refuse():
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9),
                         'annuler': (300, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=1380000,
                         resources={'elixir': 100},
                         price_resource='elixir')
    r = _run(det, reader, taps)
    assert r.status == 'cant_afford'
    assert (700, 950) not in taps


def test_builders_unreadable_does_not_block_the_flow():
    """read_builders=None (classe pas encore là) -> on n'infère pas 0, on tente."""
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9)})
    reader = _FakeReader(builders=None, price=None)
    r = _run(det, reader, taps, confirm_decider=lambda p, res: True)
    assert r.status == 'ok'              # pas de faux 'no_builder'


# ---------------------------------------------------------------------------
# Garde-fou AUTORITATIF : un prix ROUGE = le jeu dit qu'on ne peut pas payer
#
# Signal independant de la lecture des chiffres (qui peut perdre un digit et
# SOUS-estimer le prix). Taper `confirmer` dans ce cas ouvrirait le pop-up
# "acheter des gemmes" -> il prime sur tout le reste.
# ---------------------------------------------------------------------------

def test_red_price_forces_cant_afford_even_if_digits_look_affordable():
    """Prix lu 380 000 (un zero perdu) alors que le vrai prix est 3 800 000 :
    le rouge sauve la mise."""
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9),
                         'annuler': (300, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=380_000,
                         resources={'elixir': 3_394_748},
                         price_resource='elixir', price_red=True)
    r = _run(det, reader, taps, resource_type='elixir')
    assert r.status == 'cant_afford'
    assert (700, 950) not in taps        # confirmer JAMAIS tape
    assert (300, 950) in taps


def test_red_price_overrides_an_explicit_confirm_decider():
    """Meme un decideur qui dit OUI ne doit pas passer outre le rouge."""
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9),
                         'annuler': (300, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=1000, price_red=True)
    r = _run(det, reader, taps, confirm_decider=lambda p, res: True)
    assert r.status == 'cant_afford'
    assert (700, 950) not in taps


def test_non_red_price_does_not_block():
    """Prix blanc (payable) : le garde-fou rouge ne se declenche pas."""
    taps = []
    det = _FakeDetector({'ameliorer': (500, 900, 0.9),
                         'confirmer_upgrade': (700, 950, 0.9)})
    reader = _FakeReader(builders=(1, 6), price=5000,
                         resources={'elixir': 999_999}, price_red=False)
    r = _run(det, reader, taps, resource_type='elixir')
    assert r.status == 'ok'
    assert (700, 950) in taps
