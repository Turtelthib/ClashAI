"""DonationManager : dons de troupes au clan (V5.2, increment 4).

Detecteur UI, CNN barre de troupes et I/O sont injectes : aucun ADB, aucun poids.

L'invariant CRITIQUE teste ici : l'onglet `dons_gemme` (don paye en GEMMES) ne
doit JAMAIS etre tape, et rien n'est donne tant que l'onglet GRATUIT
(`dons_normaux`) n'a pas ete trouve ET selectionne.
"""

import numpy as np
import pytest
from PIL import Image

from clashai.perception.ui_detector import Detection
from clashai.social.donations import GEM_TAB, DonationManager


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)


class _FakeDetector:
    """detect_raw -> {classe: [Detection]} ; detect -> {classe: (x,y,conf)}."""

    def __init__(self, raw=None, buttons=None):
        self._raw = raw or {}
        self._b = buttons or {}

    def detect_raw(self, img):
        return self._raw

    def detect(self, img):
        return dict(self._b)

    def annotate(self, img, path):
        return path


class _FakeBar:
    def __init__(self, cards):
        self._cards = cards          # [(nom, cx, grayed)]

    def detect(self, img):
        return [{'name': n, 'bbox': (cx - 30, 500, cx + 30, 600),
                 'center': (cx, 550), 'conf': 0.9,
                 'is_grayed': g, 'no_tap': False}
                for n, cx, g in self._cards]


def _img(colour=(200, 60, 60)):
    """Image unie SATUREE -> les crops de boutons ne seront pas vus comme grises."""
    return Image.new('RGB', (1920, 1080), colour)


def _det(cls, x, y, w=120, h=50, conf=0.9):
    return Detection(cls, 0, x, y, w, h, conf)


def _mgr(raw=None, buttons=None):
    return DonationManager(detector=_FakeDetector(raw, buttons), verbose=False)


# ---------------------------------------------------------------------------
# Boutons `donner` : les demandes deja satisfaites sont grisees
# ---------------------------------------------------------------------------

def test_active_donate_buttons_are_returned_top_to_bottom():
    raw = {'donner': [_det('donner', 500, 800), _det('donner', 500, 300)]}
    got = _mgr(raw).find_donate_buttons(_img())
    assert [y for _x, y, _c in got] == [300, 800]      # de haut en bas


def test_greyed_donate_button_is_skipped():
    """Demande deja satisfaite (ex. 45/45) -> bouton desature -> ignore."""
    raw = {'donner': [_det('donner', 500, 300)]}
    grey = Image.new('RGB', (1920, 1080), (128, 128, 128))   # saturation nulle
    assert _mgr(raw).find_donate_buttons(grey) == []


def test_low_confidence_button_is_skipped():
    raw = {'donner': [_det('donner', 500, 300, conf=0.30)]}
    assert _mgr(raw).find_donate_buttons(_img()) == []


# ---------------------------------------------------------------------------
# SECURITE : jamais l'onglet GEMMES
# ---------------------------------------------------------------------------

def test_nothing_is_donated_without_the_free_tab():
    """Onglet gratuit introuvable -> on sort sans rien donner."""
    taps = []
    mgr = _mgr(buttons={})            # ni dons_normaux ni fermer
    status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        {'troop_bar_detector': _FakeBar([('barbare', 900, False)])})
    assert status == 'free_tab_not_found'
    assert given == 0
    assert (900, 550) not in taps     # aucune carte donnee


def test_gem_tab_is_never_tapped():
    """Meme detecte, `dons_gemme` ne doit jamais recevoir de tap."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9),
                        GEM_TAB: (900, 200, 0.95),      # plus confiant !
                        'fermer': (1500, 100, 0.9)})
    mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        {'troop_bar_detector': _FakeBar([])})
    assert (700, 200) in taps, "l'onglet GRATUIT doit etre selectionne"
    assert (900, 200) not in taps, "l'onglet GEMMES ne doit JAMAIS etre tape"


def test_free_tab_is_selected_before_any_card():
    """L'onglet gratuit est tape AVANT la premiere carte."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        {'troop_bar_detector': _FakeBar([('barbare', 900, False)])}, max_taps=1)
    assert taps.index((700, 200)) < taps.index((900, 550))


# ---------------------------------------------------------------------------
# Selection des cartes
# ---------------------------------------------------------------------------

def test_greyed_cards_are_not_donatable():
    mgr = _mgr()
    models = {'troop_bar_detector': _FakeBar(
        [('barbare', 900, False), ('pekka', 1100, True)])}
    names = [n for n, _x, _y in mgr.list_donatable(_img(), models)]
    assert names == ['barbare']


def test_cards_left_of_the_button_are_ignored():
    """Les icones de la DEMANDE (panneau de chat, a gauche) ne sont pas des
    cartes du pop-up."""
    mgr = _mgr()
    models = {'troop_bar_detector': _FakeBar(
        [('gobelin', 200, False), ('barbare', 900, False)])}
    names = [n for n, _x, _y in mgr.list_donatable(_img(), models, min_x=500)]
    assert names == ['barbare']


def test_no_troop_bar_model_means_no_cards():
    assert _mgr().list_donatable(_img(), {'troop_bar_detector': None}) == []


def test_donation_stops_when_nothing_left_to_give():
    """Le pop-up se vide -> on arrete, sans consommer tous les max_taps."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        {'troop_bar_detector': _FakeBar([])}, max_taps=6)
    assert (status, given) == ('nothing_to_give', 0)


# ---------------------------------------------------------------------------
# Donner CE QUI EST DEMANDE
#
# Envoyer 6 barbares a quelqu'un qui reclame des ballons ne rend service a
# personne. Les vignettes de la demande sont dans le panneau de chat, au-dessus
# du bouton ; le bloc est borne par le bouton precedent.
# ---------------------------------------------------------------------------

class _FakeBarXY:
    """Cartes positionnees librement : [(nom, cx, cy, grayed)]."""

    def __init__(self, cards):
        self._cards = cards

    def detect(self, img):
        return [{'name': n, 'bbox': (cx - 30, cy - 30, cx + 30, cy + 30),
                 'center': (cx, cy), 'conf': 0.9,
                 'is_grayed': g, 'no_tap': False}
                for n, cx, cy, g in self._cards]


def test_read_request_reads_icons_of_its_own_block():
    """Le bloc est borne par le bouton precedent -> pas de melange entre
    demandes empilees."""
    models = {'troop_bar_detector': _FakeBarXY([
        ('ballon', 300, 200, False),      # demande 1 (au-dessus du bouton 1)
        ('sapeur', 380, 200, False),
        ('barbare', 300, 700, False),     # demande 2
    ])}
    mgr = _mgr()
    assert mgr.read_request(_img(), models, button_y=400, block_top=0) == {
        'ballon', 'sapeur'}
    assert mgr.read_request(_img(), models, button_y=900, block_top=400) == {
        'barbare'}


def test_read_request_ignores_icons_outside_the_chat_panel():
    """Les cartes du pop-up (a droite) ne sont pas ce qui est demande."""
    models = {'troop_bar_detector': _FakeBarXY([('dragon', 1400, 200, False)])}
    assert _mgr().read_request(_img(), models, button_y=400) == set()


def test_only_requested_troops_are_donated():
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    models = {'troop_bar_detector': _FakeBarXY([
        ('barbare', 900, 550, False), ('ballon', 1100, 550, False)])}
    status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        models, max_taps=2, wanted={'ballon'})
    assert status == 'ok' and given == 2
    assert (1100, 550) in taps           # le ballon demande
    assert (900, 550) not in taps        # PAS le barbare


def test_nothing_given_when_we_have_none_of_the_requested_troops():
    """Mieux vaut ne rien donner que n'importe quoi."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    models = {'troop_bar_detector': _FakeBarXY([('barbare', 900, 550, False)])}
    status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        models, wanted={'ballon', 'sapeur'})
    assert (status, given) == ('no_match', 0)
    assert (900, 550) not in taps


def test_unreadable_request_falls_back_to_donating_anything():
    """Demande sans icone lisible -> on reste utile plutot que bloque."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    models = {'troop_bar_detector': _FakeBarXY([('barbare', 900, 550, False)])}
    status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        models, max_taps=1, wanted=set())
    assert status == 'ok' and given == 1


# ---------------------------------------------------------------------------
# Politique par defaut : REPARTIR entre les troupes proposees
#
# L'ancienne ("toujours la 1ere carte") martelait une seule troupe : sur une
# demande "ballon + sorciere", elle n'envoyait que des ballons tant que le jeu
# ne les grisait pas. Cas signale par l'utilisateur.
# ---------------------------------------------------------------------------

def test_donations_are_spread_across_the_offered_troops():
    """ballon + sorciere proposes -> les deux recoivent des dons, pas un seul."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    models = {'troop_bar_detector': _FakeBarXY([
        ('ballon', 900, 550, False), ('sorciere', 1100, 550, False)])}
    status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        models, max_taps=4)
    assert status == 'ok' and given == 4
    assert taps.count((900, 550)) == 2, "le ballon doit recevoir sa part"
    assert taps.count((1100, 550)) == 2, "la sorciere aussi"


def test_least_donated_policy_alternates():
    """Fonction pure : on sert toujours la troupe la moins donnee."""
    cards = [('ballon', 900, 550), ('sorciere', 1100, 550)]
    pick = DonationManager._least_donated
    assert pick(cards, {})[0] == 'ballon'        # aucun don -> ordre de lecture
    assert pick(cards, {'ballon': 1})[0] == 'sorciere'
    assert pick(cards, {'ballon': 1, 'sorciere': 1})[0] == 'ballon'
    assert pick(cards, {'ballon': 2, 'sorciere': 1})[0] == 'sorciere'


def test_single_troop_does_not_loop_forever():
    """Une seule troupe proposee et rien qui bouge -> on s'arrete vite."""
    taps = []
    mgr = _mgr(buttons={'dons_normaux': (700, 200, 0.9), 'fermer': (1500, 100, 0.9)})
    models = {'troop_bar_detector': _FakeBarXY([('barbare', 900, 550, False)])}
    _status, given = mgr.donate_to_request(
        (400, 300), lambda: _img(), lambda x, y: taps.append((x, y)),
        models, max_taps=30)
    assert given <= 4, f"stagnation non detectee : {given} taps"
