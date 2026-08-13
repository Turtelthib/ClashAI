"""VillageAgent + VillageCollector : recolte des ressources via le CNN UI.

Reflete la demo offline de agents/village_agent.py (bloc __main__, laisse intact).
Aucun ADB / GPU / poids : detecteur et I/O sont injectes.

Point cle : la recolte RE-SCANNE entre chaque tap (meca CoC : taper une icone en
recolte d'autres). Le detecteur fake est donc sequence par frames — chaque
detect_raw() pop la frame suivante, simulant la disparition des icones.

Rappel des priorites : chat 30 > gdc 25 > clan_castle 20 > village 15 > combat 10.
"""

from conftest import no_hardware

from clashai.agents.combat_agent import CombatAgent
from clashai.agents.village_agent import VillageAgent
from clashai.perception.ui_detector import Detection
from clashai.village.collector import VillageCollector


class _FakeDetector:
    """detect_raw() sequence par frames : {classe: [(x, y), ...]} par passe.

    La derniere frame est repetee si on scanne plus que prevu (garde-fou boucle).
    """

    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def detect_raw(self, img):
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return {cls: [Detection(cls, 0, x, y, 40, 40, 0.9) for (x, y) in pts]
                for cls, pts in frame.items()}


def _collector(frames):
    return VillageCollector(detector=_FakeDetector(frames), verbose=False)


def _agent(frames, taps):
    return VillageAgent(
        collector=_collector(frames),
        screenshot_fn=lambda: object(),          # frame non-None
        tap_fn=lambda x, y: taps.append((x, y)),
    )


# ---------------------------------------------------------------------------
# VillageCollector : la boucle re-scan
# ---------------------------------------------------------------------------

def test_tap_clears_all_stops_after_one_tap():
    """Meca CoC courante : un tap vide tout -> la 2e passe ne voit plus rien."""
    taps = []
    frames = [{'recolter_or': [(100, 400), (200, 400), (300, 400)]}, {}]
    n = _collector(frames).collect(lambda: object(), lambda x, y: taps.append((x, y)))
    assert n == 1
    assert taps == [(100, 400)]          # une seule icone tapee, pas les 3


def test_tap_clears_one_loops_until_empty():
    """Cas ou chaque tap ne vide qu'une icone -> plusieurs passes."""
    taps = []
    frames = [
        {'recolter_or': [(100, 400)]},
        {'recolter_elixir': [(500, 400)]},
        {'recolter_elixir_noire': [(900, 400)]},
        {},
    ]
    n = _collector(frames).collect(lambda: object(), lambda x, y: taps.append((x, y)))
    assert n == 3
    assert taps == [(100, 400), (500, 400), (900, 400)]


def test_stuck_icon_breaks_the_loop():
    """Si la meme icone reapparait au meme endroit (tap sans effet), on arrete."""
    taps = []
    frames = [{'recolter_or': [(100, 400)]}]      # jamais vide -> repetee
    n = _collector(frames).collect(lambda: object(), lambda x, y: taps.append((x, y)))
    assert n == 1                                  # tapee une fois, puis stop
    assert taps == [(100, 400)]


def test_collect_noop_on_empty_village():
    taps = []
    n = _collector([{}]).collect(lambda: object(), lambda x, y: taps.append((x, y)))
    assert n == 0
    assert taps == []


def test_collect_noop_when_screenshot_is_none():
    taps = []
    frames = [{'recolter_or': [(100, 400)]}]
    n = _collector(frames).collect(lambda: None, lambda x, y: taps.append((x, y)))
    assert n == 0
    assert taps == []


def test_collect_ignores_non_resource_classes():
    """Seules les classes recolter_* declenchent un tap."""
    taps = []
    frames = [{'attaquer': [(500, 500)], 'recolter_or': [(100, 400)]}, {}]
    n = _collector(frames).collect(lambda: object(), lambda x, y: taps.append((x, y)))
    assert n == 1
    assert taps == [(100, 400)]


# ---------------------------------------------------------------------------
# VillageAgent : cycle world -> can_run -> pick -> run
# ---------------------------------------------------------------------------

def test_village_not_picked_when_away_from_home(scheduler):
    scheduler.register(_agent([{'recolter_or': [(100, 400)]}, {}], []))
    assert scheduler.pick({'on_village_home': False}) is None


def test_village_run_collects_then_goes_on_cooldown(scheduler):
    taps = []
    agent = _agent([{'recolter_or': [(100, 400)]}, {'recolter_or': [(200, 400)]}, {}], taps)
    scheduler.register(agent)

    result = scheduler.run(scheduler.pick({'on_village_home': True}))

    assert result.ok
    assert result.data['collected'] == 2
    assert taps == [(100, 400), (200, 400)]
    # Cooldown pose -> plus eligible immediatement.
    assert scheduler.pick({'on_village_home': True}) is None


# ---------------------------------------------------------------------------
# Priorite : clan_castle (20) > village (15) > combat (10)
# ---------------------------------------------------------------------------

def test_village_preempts_combat(scheduler, village_farm):
    scheduler.register(_agent([{'recolter_or': [(100, 400)]}, {}], []))
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))
    assert scheduler.pick(village_farm).name == 'village'


def test_combat_is_default_once_village_on_cooldown(scheduler, village_farm):
    village = _agent([{'recolter_or': [(100, 400)]}, {}], [])
    scheduler.register(village)
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))

    scheduler.run(scheduler.pick(village_farm))
    assert scheduler.pick(village_farm).name == 'combat'
