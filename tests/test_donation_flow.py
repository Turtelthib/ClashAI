"""Parcours complet des dons : ouvrir le chat -> servir -> refermer (5.3.3a).

Extrait de la demo validee en V5.2. L'invariant ajoute ici : une fois ouvert,
le chat est TOUJOURS referme, meme si un don plante — sinon le bot reprendrait
sa boucle avec le chat par-dessus le village.
"""

import pytest

from clashai.social.donation_flow import DEFAULT_MAX_REQUESTS, donate_visible_requests


class _Monitor:
    def __init__(self, opens=True):
        self.opens = opens
        self.opened = 0
        self.closed = 0

    def open_chat(self, classify_fn, models):
        self.opened += 1
        return self.opens

    def close_chat(self):
        self.closed += 1


class _Manager:
    def __init__(self, buttons, statuses=None, boom_on=None):
        self._buttons = buttons
        self._statuses = statuses or {}
        self._boom_on = boom_on
        self.calls = []

    def find_donate_buttons(self, img):
        return list(self._buttons)

    def donate_to_request(self, xy, screenshot_fn, tap_fn, models, wanted=None):
        self.calls.append((xy, wanted))
        if xy == self._boom_on:
            raise ValueError('ADB perdu en plein don')
        return self._statuses.get(xy, ('ok', 4))


def _run(manager, monitor=None, image=object(), **kw):
    monitor = monitor or _Monitor()
    out = donate_visible_requests(manager, monitor, lambda: image,
                                  lambda x, y: None, lambda img, m: ('x', 1),
                                  models={}, **kw)
    return out, monitor


def test_every_visible_request_is_served_then_the_chat_is_closed():
    mgr = _Manager([(500, 300, .9), (500, 700, .9)])
    out, monitor = _run(mgr)
    assert [xy for xy, _w in mgr.calls] == [(500, 300), (500, 700)]
    assert out['demandes_vues'] == 2 and out['demandes_servies'] == 2
    assert out['dons'] == 8
    assert monitor.opened == 1 and monitor.closed == 1


def test_wanted_troops_are_passed_to_every_request():
    mgr = _Manager([(500, 300, .9), (500, 700, .9)])
    out, _m = _run(mgr, wanted={'ballon', 'yeti'})
    assert all(w == {'ballon', 'yeti'} for _xy, w in mgr.calls)
    assert out['troupes_voulues'] == ['ballon', 'yeti']


def test_no_wanted_means_free_choice():
    mgr = _Manager([(500, 300, .9)])
    out, _m = _run(mgr)
    assert mgr.calls[0][1] is None and out['troupes_voulues'] is None


def test_max_requests_limits_how_many_are_served():
    mgr = _Manager([(500, y, .9) for y in (100, 200, 300, 400)])
    out, _m = _run(mgr, max_requests=2)
    assert len(mgr.calls) == 2
    assert out['demandes_vues'] == 4


def test_the_default_limit_is_applied():
    mgr = _Manager([(500, y, .9) for y in range(0, 1000, 100)])
    _run(mgr)
    assert len(mgr.calls) == DEFAULT_MAX_REQUESTS


def test_statuses_are_reported_and_only_ok_counts_as_served():
    mgr = _Manager([(500, 300, .9), (500, 700, .9)],
                   statuses={(500, 700): ('no_match', 0)})
    out, _m = _run(mgr)
    assert out['statuts'] == ['ok', 'no_match']
    assert out['demandes_servies'] == 1 and out['dons'] == 4


def test_no_request_still_closes_the_chat():
    out, monitor = _run(_Manager([]))
    assert out['demandes_vues'] == 0 and out['dons'] == 0
    assert monitor.closed == 1


def test_no_screenshot_gives_nothing_and_closes():
    mgr = _Manager([(500, 300, .9)])
    out, monitor = _run(mgr, image=None)
    assert mgr.calls == [] and out['demandes_vues'] == 0
    assert monitor.closed == 1


def test_a_chat_that_does_not_open_raises_and_donates_nothing():
    mgr = _Manager([(500, 300, .9)])
    monitor = _Monitor(opens=False)
    with pytest.raises(RuntimeError, match='chat de clan'):
        _run(mgr, monitor=monitor)
    assert mgr.calls == []


# ---------------------------------------------------------------------------
# INVARIANT : le chat est referme meme si un don plante
# ---------------------------------------------------------------------------

def test_the_chat_is_closed_even_when_a_donation_crashes():
    mgr = _Manager([(500, 300, .9), (500, 700, .9)], boom_on=(500, 300))
    monitor = _Monitor()
    with pytest.raises(ValueError):
        _run(mgr, monitor=monitor)
    assert monitor.closed == 1
