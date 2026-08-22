"""Le `world` expose les boutons visibles (CNN UI a cadence reduite).

C'est ce que lira le futur LocalLLMBrain : "voila l'ecran, voila les boutons
disponibles, que fais-tu ?". Sans ca, il devrait declencher une detection a
chaque decision.
"""

import time

from clashai.agents.world import WORLD_KEYS, build_world


class _PT:
    """Double du PerceptionThread : rend un etat fige."""

    def __init__(self, state, fresh=True):
        self._state = state
        self._fresh = fresh

    def is_fresh(self, max_age_s=2.0):
        return self._fresh

    def get_latest(self):
        return dict(self._state)


def _state(**kw):
    base = {'screen_state': 'village_home', 'screen_conf': 0.99,
            'buildings': [], 'troop_bar': [], 'troop_positions': {},
            'buttons': {}, 'buttons_ts': 0.0, 'timestamp': time.time()}
    base.update(kw)
    return base


def test_world_always_carries_the_button_keys():
    """Meme sans perception : cles presentes, valeurs neutres (jamais None.get)."""
    w = build_world()
    assert w['buttons'] == {}
    assert w['buttons_age_s'] is None
    assert all(k in w for k in WORLD_KEYS)


def test_buttons_are_exposed_with_their_age():
    pt = _PT(_state(buttons={'attaquer': (157, 955, 0.94)},
                    buttons_ts=time.time() - 1.5))
    w = build_world({'perception_thread': pt})
    assert w['buttons'] == {'attaquer': (157, 955, 0.94)}
    assert 1.0 < w['buttons_age_s'] < 5.0


def test_never_detected_yet_means_age_none():
    """buttons_ts = 0 -> on ne pretend pas que la detection est 'toute fraiche'."""
    w = build_world({'perception_thread': _PT(_state(buttons_ts=0.0))})
    assert w['buttons_age_s'] is None


def test_stale_perception_does_not_leak_buttons():
    """Cache perime -> world neutre, pas de boutons fantomes."""
    pt = _PT(_state(buttons={'attaquer': (1, 2, 0.9)}), fresh=False)
    w = build_world({'perception_thread': pt})
    assert w['buttons'] == {}


def test_missing_buttons_key_is_tolerated():
    """Un PerceptionThread plus ancien (sans 'buttons') ne casse pas le world."""
    st = _state()
    del st['buttons']
    del st['buttons_ts']
    w = build_world({'perception_thread': _PT(st)})
    assert w['buttons'] == {} and w['buttons_age_s'] is None


# ---------------------------------------------------------------------------
# Valeurs LUES a l'ecran (ressources, ouvriers, labo)
# ---------------------------------------------------------------------------

def test_readings_are_exposed():
    pt = _PT(_state(readings={'resources': {'or': 4261458},
                              'builders': {'libres': 5, 'total': 5},
                              'lab_libre': True}))
    w = build_world({'perception_thread': pt})
    assert w['readings']['resources']['or'] == 4261458
    assert w['readings']['lab_libre'] is True


def test_readings_default_to_empty_not_none():
    """Une cle absente = non lisible. Jamais de valeur devinee, jamais de None
    qui ferait planter un .get() en aval."""
    assert build_world()['readings'] == {}
    w = build_world({'perception_thread': _PT(_state())})
    assert w['readings'] == {}


# ---------------------------------------------------------------------------
# _count_widgets : compter au seuil d'ACTION (increment 5.3.1)
# ---------------------------------------------------------------------------

class _Det:
    def __init__(self, conf):
        self.conf = conf


def _count(raw):
    from clashai.perception.perception_thread import PerceptionThread
    return PerceptionThread._count_widgets(raw)


def test_collectors_are_counted_per_resource():
    got = _count({'recolter_or': [_Det(0.9)] * 5,
                  'recolter_elixir': [_Det(0.8)] * 6,
                  'recolter_elixir_noire': [_Det(0.76)] * 3})
    assert got['recoltes'] == {'or': 5, 'elixir': 6, 'elixir_noire': 3}


def test_weak_detections_do_not_inflate_the_count():
    """Un nombre annonce au cerveau doit valoir ce sur quoi on AGIRAIT : on
    compte au seuil d'action (0.60), pas au seuil d'inference (0.40)."""
    got = _count({'recolter_or': [_Det(0.9), _Det(0.45), _Det(0.7)]})
    assert got['recoltes'] == {'or': 2}


def test_a_resource_with_nothing_to_collect_is_absent():
    got = _count({'recolter_or': [_Det(0.2)]})
    assert 'recoltes' not in got


def test_pending_donations_are_counted():
    got = _count({'donner': [_Det(0.9), _Det(0.8)]})
    assert got['dons_en_attente'] == 2


def test_no_donation_button_means_no_key():
    assert 'dons_en_attente' not in _count({})


def test_counting_an_empty_screen_is_empty_not_zeroed():
    """Clef absente = « rien vu ici », jamais « il y en a zero » : c'est ce qui
    permet au cerveau de ne pas commenter un ecran ou la question ne se pose pas."""
    assert _count({}) == {}
