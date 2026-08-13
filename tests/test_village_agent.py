"""VillageAgent + VillageCollector : recolte des ressources via le CNN UI.

Reflete la demo offline de agents/village_agent.py (bloc __main__, laisse intact).
Aucun ADB / GPU / poids : detecteur et I/O sont injectes.

Rappel des priorites : chat 30 > gdc 25 > clan_castle 20 > village 15 > combat 10.
"""

from conftest import no_hardware

from clashai.agents.combat_agent import CombatAgent
from clashai.agents.village_agent import VillageAgent
from clashai.perception.ui_detector import Detection
from clashai.village.collector import VillageCollector


class _FakeDetector:
    """detect_raw() figee : `spec` = {classe: nb d'icones}."""

    def __init__(self, spec):
        self._spec = spec

    def detect_raw(self, img):
        out = {}
        for cls, n in self._spec.items():
            out[cls] = [Detection(cls, 0, 100 * (i + 1), 400, 40, 40, 0.9)
                        for i in range(n)]
        return out


def _agent(spec, taps):
    collector = VillageCollector(detector=_FakeDetector(spec), verbose=False)
    return VillageAgent(
        collector=collector,
        screenshot_fn=lambda: object(),          # frame non-None
        tap_fn=lambda x, y: taps.append((x, y)),
    )


# ---------------------------------------------------------------------------
# VillageCollector : detecte -> tape chaque icone
# ---------------------------------------------------------------------------

def test_collect_taps_every_resource_icon():
    taps = []
    collector = VillageCollector(
        detector=_FakeDetector({'recolter_or': 2, 'recolter_elixir': 1}),
        verbose=False,
    )
    n = collector.collect(lambda: object(), lambda x, y: taps.append((x, y)))
    assert n == 3
    assert taps == [(100, 400), (200, 400), (100, 400)]


def test_collect_noop_on_empty_village():
    taps = []
    collector = VillageCollector(detector=_FakeDetector({}), verbose=False)
    assert collector.collect(lambda: object(), lambda x, y: taps.append((x, y))) == 0
    assert taps == []


def test_collect_noop_when_screenshot_is_none():
    """Pas de frame -> 0 recolte, aucun tap, aucune erreur."""
    taps = []
    collector = VillageCollector(
        detector=_FakeDetector({'recolter_or': 5}), verbose=False,
    )
    assert collector.collect(lambda: None, lambda x, y: taps.append((x, y))) == 0
    assert taps == []


def test_collect_ignores_non_resource_classes():
    """Seules les classes recolter_* declenchent un tap."""
    taps = []
    collector = VillageCollector(
        detector=_FakeDetector({'attaquer': 1, 'recolter_or': 1}), verbose=False,
    )
    assert collector.collect(lambda: object(), lambda x, y: taps.append((x, y))) == 1
    assert taps == [(100, 400)]


# ---------------------------------------------------------------------------
# VillageAgent : cycle world -> can_run -> pick -> run
# ---------------------------------------------------------------------------

def test_village_not_picked_when_away_from_home(scheduler):
    scheduler.register(_agent({'recolter_or': 1}, []))
    assert scheduler.pick({'on_village_home': False}) is None


def test_village_run_collects_then_goes_on_cooldown(scheduler):
    taps = []
    agent = _agent({'recolter_or': 2}, taps)
    scheduler.register(agent)

    result = scheduler.run(scheduler.pick({'on_village_home': True}))

    assert result.ok
    assert result.data['collected'] == 2
    assert len(taps) == 2
    # Cooldown pose -> plus eligible immediatement.
    assert scheduler.pick({'on_village_home': True}) is None


# ---------------------------------------------------------------------------
# Priorite : clan_castle (20) > village (15) > combat (10)
# ---------------------------------------------------------------------------

def test_village_preempts_combat(scheduler, village_farm):
    scheduler.register(_agent({'recolter_or': 1}, []))
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))
    assert scheduler.pick(village_farm).name == 'village'


def test_combat_is_default_once_village_on_cooldown(scheduler, village_farm):
    village = _agent({'recolter_or': 1}, [])
    scheduler.register(village)
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))

    # Une recolte -> village en cooldown -> combat reprend le sol.
    scheduler.run(scheduler.pick(village_farm))
    assert scheduler.pick(village_farm).name == 'combat'
